"""Restoration-specific CNN/Mamba hybrid U-Net.

The shallow encoder deliberately remains convolutional at H and H/2.  Local
four-direction SS2D is introduced only at H/4, where its sequence length is
manageable, and the decoder retains all four spatial skips.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .context_unet import _group_count
from .fusion_unet import IdentityResidualHead, PhysicsPyramid, ZeroGatedFusion
from .mamba_unet import SS2D


class ConvStage(nn.Sequential):
    """Two 3x3 convolutional refinements, optionally downsampling first."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        groups = _group_count(out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class HybridMambaBlock(nn.Module):
    """SS2D residual block with an optional parallel local convolution path."""

    def __init__(self, dim: int, d_state: int = 16, use_local_branch: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state)
        self.mamba_scale = nn.Parameter(torch.full((dim,), 1e-3))
        self.use_local_branch = use_local_branch
        if use_local_branch:
            self.local_norm = nn.GroupNorm(_group_count(dim), dim)
            self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
            self.pointwise = nn.Conv2d(dim, dim, 1, bias=False)
            self.local_scale = nn.Parameter(torch.full((dim,), 1e-3))
        else:
            self.register_parameter("local_scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to channel-first features."""
        scanned = self.ss2d(self.norm(x.permute(0, 2, 3, 1).contiguous()))
        scanned = scanned.permute(0, 3, 1, 2).contiguous()
        output = x + scanned * self.mamba_scale.view(1, -1, 1, 1)
        if self.use_local_branch:
            local = self.pointwise(self.depthwise(self.local_norm(x)))
            output = output + local * self.local_scale.view(1, -1, 1, 1)
        return output


class ECAResidual(nn.Module):
    """Efficient channel attention behind a small residual layer scale."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.scale = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.adaptive_avg_pool2d(x, 1).flatten(2).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights)).transpose(1, 2).unsqueeze(-1)
        return x + (x * weights) * self.scale.view(1, -1, 1, 1)


class HybridMambaStage(nn.Module):
    """Checkpointed hybrid blocks followed by residual ECA."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        use_local_branch: bool = False,
        use_checkpoint: bool = True,
        use_eca: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [HybridMambaBlock(dim, d_state, use_local_branch) for _ in range(depth)]
        )
        self.eca = ECAResidual(dim) if use_eca else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.eca(x)


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pad and partition a BHWC tensor into square windows."""
    batch, height, width, channels = x.shape
    pad_h = (-height) % window_size
    pad_w = (-width) % window_size
    padded = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
    padded_h, padded_w = height + pad_h, width + pad_w
    windows = padded.view(
        batch,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
        channels,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size**2, channels)
    return windows, (batch, height, width, padded_h, padded_w, channels)


def window_reverse(
    windows: torch.Tensor, metadata: tuple[int, ...], window_size: int
) -> torch.Tensor:
    """Restore windows returned by :func:`window_partition` to BHWC."""
    batch, height, width, padded_h, padded_w, channels = metadata
    x = windows.view(
        batch,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, channels)
    return x[:, :height, :width, :].contiguous()


class WindowAttentionBlock(nn.Module):
    """Pre-norm window attention and MLP with residual layer scales."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.attention_scale = nn.Parameter(torch.full((dim,), 1e-3))
        self.mlp_scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply non-overlapping window attention to channel-first features."""
        channel_last = x.permute(0, 2, 3, 1).contiguous()
        windows, metadata = window_partition(channel_last, self.window_size)
        normalized = self.norm1(windows)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        windows = windows + attended * self.attention_scale
        windows = windows + self.mlp(self.norm2(windows)) * self.mlp_scale
        restored = window_reverse(windows, metadata, self.window_size)
        return restored.permute(0, 3, 1, 2).contiguous()


class GatedDecoderBlock(nn.Module):
    """Project, gate, and fuse a decoder feature with its encoder skip."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.decoder_project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.skip_project = nn.Conv2d(skip_channels, out_channels, 1, bias=False)
        self.gate = nn.Conv2d(out_channels * 2, out_channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)
        self.fuse = ConvStage(out_channels * 2, out_channels)
        self.last_gate: torch.Tensor | None = None

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        decoder = self.decoder_project(decoder)
        skip = self.skip_project(skip)
        gate = torch.sigmoid(self.gate(torch.cat((decoder, skip), dim=1)))
        self.last_gate = gate
        return self.fuse(torch.cat((decoder, skip * gate), dim=1))


class HybridMambaUNet(nn.Module):
    """Hybrid restoration U-Net with optional local, attention, and physics paths."""

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (64, 128, 256, 384, 512),
        d_state: int = 16,
        use_local_branch: bool = True,
        use_attention: bool = True,
        use_physics: bool = False,
        use_checkpoint: bool = True,
        window_size: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        if len(widths) != 5:
            raise ValueError("widths must contain the H, H/2, H/4, H/8, and H/16 widths")
        expected_channels = 7 if use_physics else 3
        if in_channels != expected_channels:
            raise ValueError(
                f"use_physics={use_physics} requires in_channels={expected_channels}, got {in_channels}"
            )
        self.in_channels = in_channels
        self.use_physics = use_physics
        c0, c1, c2, c3, c4 = tuple(widths)

        self.stem = ConvStage(3, c0)
        self.enc_half = ConvStage(c0, c1, stride=2)
        self.down_quarter = ConvStage(c1, c2, stride=2)
        self.enc_quarter = HybridMambaStage(c2, 2, d_state, use_local_branch, use_checkpoint)
        self.down_eighth = ConvStage(c2, c3, stride=2)
        self.enc_eighth = HybridMambaStage(c3, 2, d_state, use_local_branch, use_checkpoint)
        self.down_sixteenth = ConvStage(c3, c4, stride=2)
        self.bottleneck_in = HybridMambaStage(
            c4, 1, d_state, use_local_branch, use_checkpoint, use_eca=False
        )
        self.attention = (
            WindowAttentionBlock(c4, window_size, num_heads, 2.0)
            if use_attention
            else nn.Identity()
        )
        self.bottleneck_out = HybridMambaStage(c4, 2, d_state, use_local_branch, use_checkpoint)

        if use_physics:
            physics_widths = (16, 32, 64, 128, 256)
            self.physics = PhysicsPyramid(physics_widths)
            self.fusions = nn.ModuleList(
                [
                    ZeroGatedFusion(image, prior)
                    for image, prior in zip(widths, physics_widths, strict=True)
                ]
            )

        self.dec_eighth = GatedDecoderBlock(c4, c3, c3)
        self.dec_quarter = GatedDecoderBlock(c3, c2, c2)
        self.dec_half = GatedDecoderBlock(c2, c1, c1)
        self.dec_full = GatedDecoderBlock(c1, c0, c0)
        self.head = IdentityResidualHead(c0)

    def _fuse(self, index: int, image: torch.Tensor, priors: list[torch.Tensor]) -> torch.Tensor:
        return self.fusions[index](image, priors[index]) if self.use_physics else image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            channels = x.shape[1] if x.ndim >= 2 else "unknown"
            raise ValueError(f"HybridMambaUNet expects {self.in_channels} channels, got {channels}")
        rgb = x[:, :3]
        priors = self.physics(x[:, 3:]) if self.use_physics else []

        full = self._fuse(0, self.stem(rgb), priors)
        half = self._fuse(1, self.enc_half(full), priors)
        quarter = self._fuse(2, self.enc_quarter(self.down_quarter(half)), priors)
        eighth = self._fuse(3, self.enc_eighth(self.down_eighth(quarter)), priors)
        bottleneck = self.down_sixteenth(eighth)
        bottleneck = self.bottleneck_out(self.attention(self.bottleneck_in(bottleneck)))
        bottleneck = self._fuse(4, bottleneck, priors)

        decoded = self.dec_eighth(bottleneck, eighth)
        decoded = self.dec_quarter(decoded, quarter)
        decoded = self.dec_half(decoded, half)
        decoded = self.dec_full(decoded, full)
        return self.head(decoded, rgb)


def hybridmamba_core_3ch() -> HybridMambaUNet:
    return HybridMambaUNet(use_local_branch=False, use_attention=False)


def hybridmamba_local_3ch() -> HybridMambaUNet:
    return HybridMambaUNet(use_local_branch=True, use_attention=False)


def hybridmamba_attn_3ch() -> HybridMambaUNet:
    return HybridMambaUNet(use_local_branch=True, use_attention=True)


def hybridmambafusion_7ch_tv() -> HybridMambaUNet:
    return HybridMambaUNet(
        in_channels=7, use_local_branch=True, use_attention=True, use_physics=True
    )


__all__ = [
    "ECAResidual",
    "GatedDecoderBlock",
    "HybridMambaBlock",
    "HybridMambaUNet",
    "WindowAttentionBlock",
    "window_partition",
    "window_reverse",
]
