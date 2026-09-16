"""
fanet.py
--------
Feature Attention Network (FA-Net / FA+Net) for underwater image enhancement.

References:
    - "FA-Net: A Deep-Learning Based Approach for Underwater Single Image Enhancement" (ICDIP)
    - "FA+Net: A Fast and Lightweight Network for Underwater Image Enhancement" (Applied Sciences)

Key components:
    - Residual Feature Attention Block (RFAB)
    - Channel Attention (CA) for wavelength/channel dependency
    - Pixel Attention (PA) for spatially varying haze and turbidity
    - Residual global & local learning
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention module using global average pooling and a 2-layer MLP."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.avg_pool(x))


class PixelAttention(nn.Module):
    """Pixel Attention module capturing spatially varying degradation."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, max(channels // 2, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 2, 8), 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualFeatureAttentionBlock(nn.Module):
    """
    Residual Feature Attention Block (RFAB):
    Combines spatial convolution with Channel Attention and Pixel Attention.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.ca = ChannelAttention(channels)
        self.pa = PixelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        ca_weight = self.ca(res)
        pa_weight = self.pa(res)
        res = res * ca_weight * pa_weight
        return x + res


class FANet(nn.Module):
    """
    Feature Attention Network (FA-Net) for underwater image enhancement.

    Args:
        in_channels (int): Number of input channels (3 for RGB, 4/5 for physics priors).
        out_channels (int): Number of output channels (default 3).
        channels (int): Number of internal feature channels (default 32).
        num_blocks (int): Number of stacked RFAB blocks (default 6).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 32,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1, bias=True)
        self.blocks = nn.ModuleList(
            [ResidualFeatureAttentionBlock(channels) for _ in range(num_blocks)]
        )
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.head = nn.Conv2d(channels, out_channels, 3, padding=1, bias=True)

        # Zero-initialize head for clean residual start
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        feat = self.stem(inputs)
        x = feat
        for block in self.blocks:
            x = block(x)
        x = self.fuse(x) + feat
        residual = self.head(x)

        return torch.clamp(inputs[:, :3] + residual, 0.0, 1.0)


def build_fanet(in_channels: int = 3) -> FANet:
    """Build FA-Net with 32 channels and 4 RFAB blocks (~85k parameters)."""
    return FANet(in_channels=in_channels, out_channels=3, channels=32, num_blocks=4)



__all__ = [
    "ChannelAttention",
    "FANet",
    "PixelAttention",
    "ResidualFeatureAttentionBlock",
    "build_fanet",
]
