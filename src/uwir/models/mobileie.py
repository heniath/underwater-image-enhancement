"""
mobileie.py
-----------
MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time
Image Enhancement on Mobile Devices (ICCV 2025).

Adapted from official repository:
    https://github.com/AVC2-UESTC/MobileIE

Key features:
    - Multi-Branch Re-parameterized Convolutions (MBRConv)
    - Feature Self-Transform (FST)
    - Hierarchical Dual-Path Attention (channel + spatial max pooling)
    - Extremely low parameter count (~8k - 15k parameters)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBRConv3(nn.Module):
    """3x3 Multi-Branch Re-parameterizable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        mid = out_channels * rep_scale
        self.conv = nn.Conv2d(in_channels, mid, 3, padding=1, bias=True)
        self.conv_bn = nn.BatchNorm2d(mid)
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(mid)
        self.conv_crossh = nn.Conv2d(in_channels, mid, (3, 1), padding=(1, 0), bias=True)
        self.conv_crossh_bn = nn.BatchNorm2d(mid)
        self.conv_crossv = nn.Conv2d(in_channels, mid, (1, 3), padding=(0, 1), bias=True)
        self.conv_crossv_bn = nn.BatchNorm2d(mid)
        self.conv_out = nn.Conv2d(mid * 8, out_channels, 1, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x = torch.cat(
            [
                x0,
                x1,
                x2,
                x3,
                self.conv_bn(x0),
                self.conv1_bn(x1),
                self.conv_crossh_bn(x2),
                self.conv_crossv_bn(x3),
            ],
            dim=1,
        )
        return self.conv_out(x)


class MBRConv5(nn.Module):
    """5x5 Multi-Branch Re-parameterizable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        mid = out_channels * rep_scale
        self.conv = nn.Conv2d(in_channels, mid, 5, padding=2, bias=True)
        self.conv_bn = nn.BatchNorm2d(mid)
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=True)
        self.conv1_bn = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(in_channels, mid, 3, padding=1, bias=True)
        self.conv2_bn = nn.BatchNorm2d(mid)
        self.conv_crossh = nn.Conv2d(in_channels, mid, (3, 1), padding=(1, 0), bias=True)
        self.conv_crossh_bn = nn.BatchNorm2d(mid)
        self.conv_crossv = nn.Conv2d(in_channels, mid, (1, 3), padding=(0, 1), bias=True)
        self.conv_crossv_bn = nn.BatchNorm2d(mid)
        self.conv_out = nn.Conv2d(mid * 10, out_channels, 1, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [
                x1,
                x2,
                x3,
                x4,
                x5,
                self.conv_bn(x1),
                self.conv1_bn(x2),
                self.conv2_bn(x3),
                self.conv_crossh_bn(x4),
                self.conv_crossv_bn(x5),
            ],
            dim=1,
        )
        return self.conv_out(x)


class MBRConv1(nn.Module):
    """1x1 Multi-Branch Re-parameterizable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid = out_channels * rep_scale
        self.conv = nn.Conv2d(in_channels, mid, 1, bias=True)
        self.conv_bn = nn.BatchNorm2d(mid)
        self.conv_out = nn.Conv2d(mid * 2, out_channels, 1, bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(inp)
        x = torch.cat([x0, self.conv_bn(x0)], dim=1)
        return self.conv_out(x)


class FeatureSelfTransform(nn.Module):
    """Feature Self-Transform (FST) with learnable weights and channel bias."""

    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block = block
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block(x)
        w1 = self.weight1 * x1
        w2 = self.weight2 * x1
        return w1 * w2 + self.bias


class MobileIENet(nn.Module):
    """
    MobileIE architecture adapted for underwater image enhancement.

    Args:
        in_channels (int): Input channels (3 for RGB, 4/5 for physics priors).
        out_channels (int): Output channels (default 3).
        channels (int): Internal feature channels (default 16).
        rep_scale (int): Re-parameterization width expansion (default 2 for lightweight).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 16,
        rep_scale: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        self.head = FeatureSelfTransform(
            nn.Sequential(
                MBRConv5(in_channels, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale),
            ),
            channels,
        )
        self.body = FeatureSelfTransform(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels,
        )

        # Hierarchical Dual-Path Attention
        self.att_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid(),
        )
        self.att_spatial = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid(),
        )

        self.tail = MBRConv3(channels, out_channels, rep_scale=rep_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        x0 = self.head(inputs)
        x1 = self.body(x0)

        # Dual-path attention
        x2 = self.att_channel(x1)
        max_out, _ = torch.max(x2 * x1, dim=1, keepdim=True)
        x3 = self.att_spatial(max_out)
        x4 = torch.mul(x2, x3) * x1

        residual = self.tail(x4)
        return torch.clamp(inputs[:, :3] + residual, 0.0, 1.0)


def build_mobileie(in_channels: int = 3) -> MobileIENet:
    """Build MobileIE architecture (~12k parameters)."""
    return MobileIENet(in_channels=in_channels, out_channels=3, channels=16, rep_scale=2)


__all__ = [
    "FeatureSelfTransform",
    "MBRConv1",
    "MBRConv3",
    "MBRConv5",
    "MobileIENet",
    "build_mobileie",
]
