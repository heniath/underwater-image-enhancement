"""
LiteEnhanceNet: A lightweight network for real-time single underwater image enhancement
Source: https://github.com/zhangsong1213/LiteEnhanceNet
Official implementation (~13.69k parameters).
"""

import torch
import torch.nn as nn


def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6.0


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
            HardSigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock1(nn.Module):
    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs1 = HardSwish()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.hs2 = HardSwish()

    def forward(self, x):
        a = self.hs1(self.bn1(self.dw(x)))
        return self.hs2(self.bn2(self.pw(a)))


class ConvBlock2(nn.Module):
    def __init__(self, in_channels=32, out_channels=64):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs1 = HardSwish()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.hs2 = HardSwish()

    def forward(self, x):
        a = self.hs1(self.bn1(self.dw(x)))
        return self.hs2(self.bn2(self.pw(a)))


class ConvBlock3(nn.Module):
    def __init__(self, in_channels=64, out_channels=32):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs1 = HardSwish()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.hs2 = HardSwish()

    def forward(self, x):
        a = self.hs1(self.bn1(self.dw(x)))
        return self.hs2(self.bn2(self.pw(a)))


class ConvBlock4(nn.Module):
    def __init__(self, in_channels=80, out_channels=32):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs1 = HardSwish()
        self.se = SELayer(in_channels, in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.hs2 = HardSwish()

    def forward(self, x):
        a = self.hs1(self.bn1(self.dw(x)))
        a = self.se(a)
        return self.hs2(self.bn2(self.pw(a)))


class LiteEnhanceNet(nn.Module):
    """
    Official LiteEnhanceNet model (~13.69k parameters).
    Residual connection: output = clamp(x + delta, 0, 1).
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.input = nn.Conv2d(in_channels, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1 = ConvBlock1(16, 32)
        self.block2 = ConvBlock2(32, 64)
        self.block3 = ConvBlock3(64, 32)
        self.block4 = ConvBlock4(80, 32)
        self.output = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        inp_rgb = x[:, :3, :, :] if x.shape[1] > 3 else x
        f0 = self.input(x)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        # One-Shot Aggregation: concat f0 (16), f1 (32), f3 (32) -> 80
        f_cat = torch.cat([f0, f1, f3], dim=1)
        f4 = self.block4(f_cat)
        delta = self.output(f4)
        out = torch.clamp(inp_rgb + delta, 0.0, 1.0)
        return out
