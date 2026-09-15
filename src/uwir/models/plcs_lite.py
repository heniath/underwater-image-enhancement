"""
PLCS-Lite: Physics-Guided Learnable Color & Multi-Scale Dilated Network
Synergizes:
  - LCCM: Learnable Color Correction Module (from LCS-Net, Sensors 2026)
  - SMSDB: Selective Multi-Scale Dilated Block (dilation=1,2,3 for underwater haze)
  - Depthwise Separable Convolutions with HardSwish (from LiteEnhanceNet)
  - Forward Optical Degradation Re-synthesis (from m20566 & LMF-Net)

Target parameters: ~95k - 120k params. Real-time inference > 120 FPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCCM(nn.Module):
    """
    Learnable Color Correction Module (LCS-Net, Sensors 2026).
    Predicts affine transformation (alpha, beta) from global color statistics.
    Cost: ~150 parameters.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 6),
        )

    def forward(self, x):
        stats = self.mlp(x)
        alpha = torch.sigmoid(stats[:, :3]).view(-1, 3, 1, 1) * 2.0
        beta = torch.tanh(stats[:, 3:]).view(-1, 3, 1, 1) * 0.25
        return torch.clamp(alpha * x + beta, 0.0, 1.0)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.Hardswish(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        return self.act2(self.bn2(self.pw(x)))


class SMSDB(nn.Module):
    """
    Selective Multi-Scale Dilated Block (SMSDB, from LCS-Net).
    Aggregates multi-receptive-field context with dilations 1, 2, 3.
    """

    def __init__(self, channels=96):
        super().__init__()
        self.d1 = DepthwiseSeparableConv(channels, channels, dilation=1)
        self.d2 = DepthwiseSeparableConv(channels, channels, dilation=2)
        self.d3 = DepthwiseSeparableConv(channels, channels, dilation=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )
        # Channel attention for selective fusion
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels // 4), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f1 = self.d1(x)
        f2 = self.d2(x)
        f3 = self.d3(x)
        f_cat = torch.cat([f1, f2, f3], dim=1)
        fused = self.fuse(f_cat)
        w = self.ca(fused)
        return x + fused * w


class PLCSLite(nn.Module):
    """
    Proposed Hybrid Architecture: PLCS-Lite (~105k parameters).
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.lccm = LCCM()

        # Channels: [32, 64, 96]
        c1, c2, c3 = 32, 64, 96

        self.stem = DepthwiseSeparableConv(in_channels, c1)
        self.enc1 = DepthwiseSeparableConv(c1, c1)
        self.down1 = nn.MaxPool2d(2)  # H/2 (128)

        self.enc2 = DepthwiseSeparableConv(c1, c2)
        self.down2 = nn.MaxPool2d(2)  # H/4 (64)

        self.enc3 = DepthwiseSeparableConv(c2, c3)
        self.down3 = nn.MaxPool2d(2)  # H/8 (32)

        # Bottleneck: Multi-Scale Selective Dilated Block (at H/8)
        self.bottleneck = SMSDB(c3)

        # Decoder stages
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # -> H/4 (64)
        self.dec3 = DepthwiseSeparableConv(c3 + c3, c2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # -> H/2 (128)
        self.dec2 = DepthwiseSeparableConv(c2 + c2, c1)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # -> H (256)
        self.dec1 = DepthwiseSeparableConv(c1 + c1, c1)

        # Head 1: Scene Radiance J
        self.head_j = nn.Sequential(
            DepthwiseSeparableConv(c1, c1),
            nn.Conv2d(c1, out_channels, kernel_size=1),
        )

        # Head 2: Transmission Map t(x)
        self.head_t = nn.Sequential(
            DepthwiseSeparableConv(c1, 16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Head 3: Global Background Light B
        self.head_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, return_physics=False):
        inp_rgb = x[:, :3, :, :] if x.shape[1] > 3 else x
        
        # 1. Learnable color correction
        x_norm = self.lccm(x)

        # 2. Encoder
        e0 = self.stem(x_norm)    # 256, c1
        e1 = self.enc1(e0)        # 256, c1
        e2 = self.enc2(self.down1(e1))  # 128, c2
        e3 = self.enc3(self.down2(e2))  # 64, c3

        # 3. Bottleneck
        b = self.bottleneck(self.down3(e3))  # 32, c3

        # 4. Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # 64, c2
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 128, c1
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 256, c1

        # 5. Output heads
        delta = self.head_j(d1)
        j = torch.clamp(inp_rgb + delta, 0.0, 1.0)

        if return_physics or self.training:
            t = torch.clamp(self.head_t(d1), 0.05, 1.0)
            bg = self.head_b(d1).view(-1, 3, 1, 1)
            i_redeg = j * t + bg * (1.0 - t)
            if return_physics:
                return j, i_redeg, t, bg

        return j
