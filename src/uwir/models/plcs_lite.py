"""
PLCS-Lite: Proposed Lightweight Hybrid Underwater Image Enhancement Model
Integrating:
    - LCCM (Learnable Color Correction Module)
    - SMSDB (Selective Multi-Scale Dilated Block)
    - Depthwise OSA (Depthwise One-Shot Aggregation Backbone)
    - Physical Re-degradation Branch: I_redeg = J * t + B * (1 - t)
Total Parameters: 101,270
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCCM(nn.Module):
    """
    Learnable Color Correction Module:
    Estimates global 3x3 affine color transformation matrix and 3-channel bias
    from global color statistics with minimal computational overhead.
    """
    def __init__(self, mid_c=16):
        super(LCCM, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(3, mid_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, 12, 1, bias=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        params = self.fc(self.pool(x)).view(b, 12)
        m = params[:, :9].view(b, 3, 3)
        bias = params[:, 9:].view(b, 3, 1, 1)
        
        x_flat = x.view(b, 3, -1)
        x_corr = torch.bmm(m, x_flat).view(b, 3, h, w) + bias
        return torch.clamp(x_corr, 0.0, 1.0)


class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super(SELayer, self).__init__()
        mid = channels // reduction
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class DepthwiseBlock(nn.Module):
    """Depthwise-Separable Convolution Block with optional SE module."""
    def __init__(self, in_c, out_c, with_se=False):
        super(DepthwiseBlock, self).__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.act1 = nn.Hardswish(inplace=True)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.Hardswish(inplace=True)
        self.se = SELayer(in_c) if with_se else None

    def forward(self, x):
        out = self.act1(self.bn1(self.dw(x)))
        if self.se is not None:
            out = self.se(out)
        out = self.act2(self.bn2(self.pw(out)))
        return out


class SMSDB(nn.Module):
    """
    Selective Multi-Scale Dilated Block:
    Multi-branch bottleneck aggregating receptive fields (rates 1, 2, 4, global)
    with adaptive channel-wise selective reweighting.
    """
    def __init__(self, in_c=128, sub_c=32):
        super(SMSDB, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, sub_c, 1, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(sub_c, sub_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, sub_c, 1, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(sub_c, sub_c, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, sub_c, 1, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(sub_c, sub_c, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, sub_c, 1, bias=False),
            nn.BatchNorm2d(sub_c),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * sub_c, in_c, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c // 4, in_c, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = F.interpolate(self.b4(x), size=(h, w), mode='bilinear', align_corners=False)
        cat = torch.cat([out1, out2, out3, out4], dim=1)
        fused = self.fuse(cat)
        w_att = self.gate(fused)
        return x + fused * w_att


class PLCSLite(nn.Module):
    """
    PLCS-Lite: Proposed Hybrid Lightweight Architecture
    LCCM + SMSDB + Depthwise OSA + Physical Re-degradation
    Total parameters: 101,270
    """
    def __init__(self, in_channels=3):
        super(PLCSLite, self).__init__()
        # 1. Color Correction Stage
        self.lccm = LCCM(mid_c=16)
        
        # 2. Depthwise OSA Backbone
        self.input = nn.Conv2d(in_channels, 24, 1, bias=True)
        self.block1 = DepthwiseBlock(24, 64)
        self.block2 = DepthwiseBlock(64, 128)
        
        # 3. Multi-Scale Context Bottleneck
        self.smsdb = SMSDB(in_c=128, sub_c=32)
        
        self.block3 = DepthwiseBlock(128, 32)
        
        # One-Shot Aggregation (OSA): concatenate inp(24), block1(64), block3(32) -> 120 channels
        self.block4 = DepthwiseBlock(120, 32, with_se=True)
        
        # 4. Residual prediction head
        self.output = nn.Conv2d(32, 3, 1, bias=False)
        
        # 5. Forward Physical Re-degradation Branch: estimates t (1 ch) and B (3 ch)
        self.phy_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 3, padding=1, bias=True)
        )

    def forward(self, x, return_physics=False):
        """
        Args:
            x: Input degraded image [B, 3, H, W] in [0, 1]
            return_physics: If True, always returns (J, redeg, t, B)
        """
        # Step 1: Global color rectification
        x_corr = self.lccm(x)
        
        # Step 2: OSA feature extraction & multi-scale dilation
        inp = self.input(x_corr)
        x1 = self.block1(inp)
        x2 = self.block2(x1)
        x2_smsdb = self.smsdb(x2)
        x3 = self.block3(x2_smsdb)
        x_cat = torch.cat([inp, x1, x3], dim=1)
        x4 = self.block4(x_cat)
        
        # Step 3: Enhanced image J with residual connection
        j = torch.clamp(self.output(x4) + x_corr, 0.0, 1.0)
        
        # Step 4: Physical re-degradation
        phy = torch.sigmoid(self.phy_branch(x))
        t = phy[:, 0:1, :, :]  # transmission map [0, 1]
        b = phy[:, 1:4, :, :]  # background light [0, 1]
        redeg = j * t + b * (1.0 - t)
        
        if self.training or return_physics:
            return j, redeg, t, b
        return j
