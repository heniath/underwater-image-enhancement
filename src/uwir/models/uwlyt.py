"""Lightweight luminance/chrominance network for underwater enhancement.

UW-LYT keeps colour conversion fixed, processes luminance and chrominance in
separate paths, and predicts only an RGB residual.  The residual head is
zero-initialised, making a newly constructed model an exact RGB identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert an ``(..., 3, H, W)`` RGB tensor to full-range YCbCr."""
    if rgb.shape[-3] != 3:
        raise ValueError(f"expected three RGB channels, got shape {tuple(rgb.shape)}")
    red, green, blue = rgb.unbind(dim=-3)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
    return torch.stack((y, cb, cr), dim=-3)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Invert :func:`rgb_to_ycbcr` without clipping the result."""
    if ycbcr.shape[-3] != 3:
        raise ValueError(f"expected three YCbCr channels, got shape {tuple(ycbcr.shape)}")
    y, cb, cr = ycbcr.unbind(dim=-3)
    cb = cb - 0.5
    cr = cr - 0.5
    red = y + 1.402 * cr
    green = y - 0.344136 * cb - 0.714136 * cr
    blue = y + 1.772 * cb
    return torch.stack((red, green, blue), dim=-3)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(inputs, 1)
        scale = torch.sigmoid(self.expand(F.silu(self.reduce(scale))))
        return inputs * scale


class AxialDepthwiseBlock(nn.Module):
    """Residual axial depthwise convolution followed by channel mixing and SE."""

    def __init__(self, channels: int):
        super().__init__()
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 5), padding=(0, 2), groups=channels
        )
        self.vertical = nn.Conv2d(
            channels, channels, (5, 1), padding=(2, 0), groups=channels
        )
        self.mix = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.se = SqueezeExcitation(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.horizontal(inputs)
        features = self.vertical(features)
        features = F.silu(self.norm(self.mix(features)))
        return inputs + self.se(features)


class LowResolutionAttention(nn.Module):
    """Linear-cost channel attention evaluated on a 4x downsampled feature map."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.reduction = reduction
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.project = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(inputs, self.reduction, self.reduction)
        query, key, value = self.qkv(pooled).chunk(3, dim=1)
        batch, channels, height, width = query.shape
        query = F.normalize(query.flatten(2), dim=-1)
        key = F.normalize(key.flatten(2), dim=-1)
        attention = torch.softmax(query @ key.transpose(1, 2) / math.sqrt(channels), dim=-1)
        attended = (attention @ value.flatten(2)).view(batch, channels, height, width)
        attended = self.project(attended)
        attended = F.interpolate(attended, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return inputs + self.gate * attended


class UWLYT(nn.Module):
    """Underwater LYT model supporting RGB and legacy physics-channel inputs."""

    def __init__(self, in_channels: int = 3, width: int = 40):
        super().__init__()
        if in_channels not in (3, 4, 5):
            raise ValueError("UWLYT supports 3, 4, or 5 input channels")
        self.in_channels = in_channels
        self.width = width
        self.luminance_stem = nn.Conv2d(1, width, 3, padding=1)
        self.chrominance_stem = nn.Conv2d(2, width, 3, padding=1)
        self.luminance = nn.Sequential(AxialDepthwiseBlock(width), AxialDepthwiseBlock(width))
        self.chrominance = nn.Sequential(AxialDepthwiseBlock(width), AxialDepthwiseBlock(width))
        self.attention = LowResolutionAttention(width)

        prior_channels = in_channels - 3
        self.prior_projection = (
            nn.Sequential(nn.Conv2d(prior_channels, width, 1), nn.SiLU())
            if prior_channels
            else None
        )
        fusion_channels = width * (3 if prior_channels else 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, width, 1),
            nn.SiLU(),
            AxialDepthwiseBlock(width),
            AxialDepthwiseBlock(width),
        )
        self.residual_head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @staticmethod
    def _pad(inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = inputs.shape[-2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        if not (pad_h or pad_w):
            return inputs, 0, 0
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        return F.pad(inputs, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )
        inputs, pad_h, pad_w = self._pad(inputs)
        rgb = inputs[:, :3]
        ycbcr = rgb_to_ycbcr(rgb)
        luminance = self.luminance(self.luminance_stem(ycbcr[:, :1]))
        chrominance = self.chrominance(self.chrominance_stem(ycbcr[:, 1:]))
        chrominance = self.attention(chrominance)
        branches = [luminance, chrominance]
        if self.prior_projection is not None:
            branches.append(self.prior_projection(inputs[:, 3:]))
        ycbcr_residual = self.residual_head(self.fusion(torch.cat(branches, dim=1)))
        # Convert the learned correction back to RGB while subtracting the
        # same fixed inverse at zero correction. This preserves exact identity
        # initialization despite the rounded YCbCr coefficients.
        rgb_residual = ycbcr_to_rgb(ycbcr + ycbcr_residual) - ycbcr_to_rgb(ycbcr)
        output = torch.clamp(rgb + rgb_residual, 0.0, 1.0)
        if pad_h:
            output = output[..., :-pad_h, :]
        if pad_w:
            output = output[..., :, :-pad_w]
        return output


def build_uwlyt(in_channels: int, tiny: bool = False) -> UWLYT:
    """Construct a standard (40-wide) or tiny (24-wide) UW-LYT."""
    return UWLYT(in_channels=in_channels, width=24 if tiny else 40)


__all__ = ["UWLYT", "build_uwlyt", "rgb_to_ycbcr", "ycbcr_to_rgb"]
