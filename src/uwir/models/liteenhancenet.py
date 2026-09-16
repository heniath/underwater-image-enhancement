"""
liteenhancenet.py
-----------------
LiteEnhanceNet: A Lightweight Network for Real-Time Single Underwater
Image Enhancement (Expert Systems with Applications 2024).

Adapted from official repository:
    https://github.com/zhangsong1213/LiteEnhanceNet

Key features:
    - Depthwise Separable Convolutions (DSConv)
    - One-Shot Aggregation (OSA) across multi-stage features
    - Squeeze-and-Excitation (SELayer) with HardSwish / HardSigmoid
    - Ultra-compact parameter footprint (~15k parameters)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
    """Hard-Sigmoid activation: ReLU6(x + 3) / 6."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu6(x + 3.0, inplace=False) / 6.0


class HardSwish(nn.Module):
    """Hard-Swish activation: x * HardSigmoid(x)."""

    def __init__(self):
        super().__init__()
        self.hsigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hsigmoid(x)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer with Hard-Sigmoid gating."""

    def __init__(self, inp: int, oup: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(inp // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, oup, 1, bias=True),
            HardSigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ConvBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = HardSwish()
        self.pw = nn.Conv2d(16, 32, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = HardSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class ConvBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = HardSwish()
        self.pw = nn.Conv2d(32, 64, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = HardSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class ConvBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = HardSwish()
        self.pw = nn.Conv2d(64, 32, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = HardSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class ConvBlock4(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(80, 80, 3, padding=1, groups=80, bias=False)
        self.bn1 = nn.BatchNorm2d(80)
        self.act1 = HardSwish()
        self.se = SELayer(80, 80)
        self.pw = nn.Conv2d(80, 32, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = HardSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw(x)))
        x = self.se(x)
        x = self.act2(self.bn2(self.pw(x)))
        return x


class LiteEnhanceNet(nn.Module):
    """
    LiteEnhanceNet architecture.

    Args:
        in_channels (int): Number of input channels (3 for RGB, 4/5 for physics priors).
        out_channels (int): Number of output channels (default 3).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_layer = nn.Conv2d(in_channels, 16, 1, bias=False)
        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()
        self.output_layer = nn.Conv2d(32, out_channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        x0 = self.input_layer(inputs)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # One-shot aggregation (OSA): concatenate input feature, block1, and block3
        # 16 + 32 + 32 = 80 channels
        fused = torch.cat([x0, x1, x3], dim=1)
        x4 = self.block4(fused)
        residual = self.output_layer(x4)

        return torch.clamp(inputs[:, :3] + residual, 0.0, 1.0)


def build_liteenhancenet(in_channels: int = 3) -> LiteEnhanceNet:
    """Build LiteEnhanceNet (~15k parameters)."""
    return LiteEnhanceNet(in_channels=in_channels, out_channels=3)


__all__ = [
    "ConvBlock1",
    "ConvBlock2",
    "ConvBlock3",
    "ConvBlock4",
    "HardSigmoid",
    "HardSwish",
    "LiteEnhanceNet",
    "SELayer",
    "build_liteenhancenet",
]
