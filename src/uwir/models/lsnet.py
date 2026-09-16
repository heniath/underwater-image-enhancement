"""
lsnet.py
--------
LSNet: Lightweight Selective Attention Network for Underwater Image Enhancement.

References:
    - "Lightweight Selective Attention Network for Underwater Image Enhancement" (2024)

Key components:
    - Selective Attention Block (SAB)
    - Depthwise Separable Convolutions for minimal parameter footprint
    - Gated Selective Channel-Spatial Attention for turbidity filtering
    - Residual multi-stage enhancement (~18k parameters)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 followed by Pointwise 1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            bias=bias,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SelectiveAttentionBlock(nn.Module):
    """
    Selective Attention Block (SAB):
    Applies depthwise feature extraction modulated by a dual selective attention gate.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.feat_conv = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.PReLU(channels),
            DepthwiseSeparableConv(channels, channels),
        )

        # Channel selective gate
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels, 1),
            nn.Sigmoid(),
        )

        # Spatial selective gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feat_conv(x)
        c_att = self.channel_gate(feat)
        s_att = self.spatial_gate(feat)
        # Selective modulation
        modulated = feat * c_att * s_att
        return x + modulated


class LSNet(nn.Module):
    """
    Lightweight Selective Attention Network (LSNet).

    Args:
        in_channels (int): Input channels (3 for RGB, 4/5 for physics priors).
        out_channels (int): Output channels (default 3).
        base_channels (int): Base feature width (default 16).
        num_blocks (int): Number of SAB blocks per stage (default 2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 16,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        c1 = base_channels
        c2 = base_channels * 2

        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1, bias=True)

        self.stage1 = nn.Sequential(*[SelectiveAttentionBlock(c1) for _ in range(num_blocks)])
        self.trans1 = DepthwiseSeparableConv(c1, c2)
        self.stage2 = nn.Sequential(*[SelectiveAttentionBlock(c2) for _ in range(num_blocks)])
        self.trans2 = DepthwiseSeparableConv(c2, c1)
        self.stage3 = nn.Sequential(*[SelectiveAttentionBlock(c1) for _ in range(num_blocks)])

        self.head = nn.Conv2d(c1, out_channels, 3, padding=1, bias=True)

        # Zero-initialize head for residual identity at start
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        x = self.stem(inputs)
        x = self.stage1(x)
        x2 = self.trans1(x)
        x2 = self.stage2(x2)
        x3 = self.trans2(x2) + x
        x3 = self.stage3(x3)

        residual = self.head(x3)
        return torch.clamp(inputs[:, :3] + residual, 0.0, 1.0)


def build_lsnet(in_channels: int = 3) -> LSNet:
    """Build LSNet (~18k parameters)."""
    return LSNet(in_channels=in_channels, out_channels=3, base_channels=16, num_blocks=2)


__all__ = [
    "DepthwiseSeparableConv",
    "LSNet",
    "SelectiveAttentionBlock",
    "build_lsnet",
]
