"""
LiteEnhanceNet: A Lightweight Network for Real-time Single Underwater Image Enhancement
Official Architecture from zhangsong1213/LiteEnhanceNet
Optimized with PyTorch native CUDA fused Hardswish / Hardsigmoid
Total Parameters: 13,688
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()
        self.DW = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, groups=16, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(16)
        self.HS = nn.Hardswish(inplace=True)
        self.PW = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a


class ConvBlock2(nn.Module):
    def __init__(self):
        super(ConvBlock2, self).__init__()
        self.DW = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(32)
        self.HS = nn.Hardswish(inplace=True)
        self.PW = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a


class ConvBlock3(nn.Module):
    def __init__(self):
        super(ConvBlock3, self).__init__()
        self.DW = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(64)
        self.HS = nn.Hardswish(inplace=True)
        self.PW = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a


class ConvBlock4(nn.Module):
    def __init__(self):
        super(ConvBlock4, self).__init__()
        self.DW = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, groups=80, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(80)
        self.HS = nn.Hardswish(inplace=True)
        self.PW = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)
        self.SE = SELayer(80, 80)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.SE(a)
        a = self.HS(self.BNN(self.PW(a)))
        return a


class LiteEnhanceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(LiteEnhanceNet, self).__init__()
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        self.output = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()

    def forward(self, x):
        inp = self.input(x)
        x1 = self.block1(inp)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x_cat = torch.cat((inp, x1, x3), 1)
        x4 = self.block4(x_cat)
        out = self.output(x4)
        return torch.clamp(out, 0.0, 1.0)


Mynet = LiteEnhanceNet
