"""
pcf_tiny_unet.py
----------------
Original PCF-Net Lightweight Architecture (~0.17M params) from:
  "Underwater Image Enhancement via HSV-CS Representation and Perception-Driven Adaptive Fusion"
  (MDPI Remote Sensing 2026).

Key characteristics:
  - 2 downsampling stages: 32 -> 64 -> 128 (Bottleneck).
  - 2 upsampling stages: PixelShuffle(scale=2) sub-pixel reconstruction.
  - Re-parameterizable Depthwise Separable Convolutions (RepDSC).
  - Dual-Branch (RGB/Physical stream + Stabilized HSV-CS stream) with CSAGF fusion.
  - Ultra-lightweight footprint (~0.17M - 0.20M parameters).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcf_modules import (
    CSAGF,
    ColorBiasAwareModule,
    Rep3C,
    RepDepthwiseConv2d,
    RepDSC,
    ValueConfidenceModule,
    rgb_to_hsv_cs,
)

Rep3CBlock = Rep3C


class PCFTinyUNet(nn.Module):
    """
    Original 0.17M parameter PCF-Net architecture.

    Args:
        in_channels: Input channels (3 for RGB, or 5 for RGB + UDCP). Default: 3.
        out_channels: Output channels (3 for RGB). Default: 3.
        base_channels: Base channel dimension C (default: 32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c = base_channels  # 32

        # ------------------------------------------------------------------
        # 1. Dual-Branch Feature Extraction
        # ------------------------------------------------------------------
        # RGB / Physical stream
        self.branch_phys = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            RepDSC(c, c),
        )

        # HSV-CS stream
        self.branch_hsv = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            RepDSC(c, c),
        )

        # ------------------------------------------------------------------
        # 2. Color-Bias-Aware & Value-Confidence
        # ------------------------------------------------------------------
        self.color_bias = ColorBiasAwareModule(in_channels=3, mid_channels=16)
        self.value_conf = ValueConfidenceModule(mid_channels=16)
        self.hsv_mod_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # 3. CSAGF Fusion
        # ------------------------------------------------------------------
        self.csagf = CSAGF(channels=c)

        # ------------------------------------------------------------------
        # 4. 2-Stage Lightweight Encoder
        # ------------------------------------------------------------------
        # Down 1: 32 -> 64
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = RepDSC(c, c * 2)

        # Down 2 (Bottleneck): 64 -> 128
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = RepDSC(c * 2, c * 4)

        # ------------------------------------------------------------------
        # 5. 2-Stage Lightweight Decoder with PixelShuffle
        # ------------------------------------------------------------------
        # Up 1: 128 -> expands to 256 -> PixelShuffle(2) -> 64
        self.up_exp1 = RepDSC(c * 4, (c * 2) * 4)
        self.ps1 = nn.PixelShuffle(2)
        # Skip concat: 64 + 64 = 128 -> refine to 64
        self.dec1 = Rep3CBlock(c * 2 * 2, c * 2)

        # Up 2: 64 -> expands to 128 -> PixelShuffle(2) -> 32
        self.up_exp2 = RepDSC(c * 2, c * 4)
        self.ps2 = nn.PixelShuffle(2)
        # Skip concat: 32 + 32 = 64 -> refine to 32
        self.dec2 = Rep3CBlock(c * 2, c)

        # ------------------------------------------------------------------
        # 6. Output Head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback if 3-channel input passed to 5-channel model
        if x.shape[1] < self.in_channels:
            pad_ch = self.in_channels - x.shape[1]
            x = torch.cat([x, x[:, :pad_ch, :, :]], dim=1)

        rgb = x[:, :3, :, :]

        # 1. Dual-branch shallow extraction
        f_phys = self.branch_phys(x)

        hsv_cs = rgb_to_hsv_cs(rgb)
        f_hsv = self.branch_hsv(hsv_cs)

        # Color-Bias & Value-Confidence Modulation
        w_cb = self.color_bias(rgb)
        c_conf = self.value_conf(hsv_cs[:, 3:4, :, :])
        w_final = w_cb * c_conf
        f_hsv_mod = self.hsv_mod_conv(f_hsv * w_final)

        # 2. CSAGF Fusion
        f_fused = self.csagf(f_phys, f_hsv_mod)  # (B, 32, H, W)

        # 3. Encoder
        f_enc1 = self.enc1(self.pool1(f_fused))  # (B, 64, H/2, W/2)
        f_bn = self.bottleneck(self.pool2(f_enc1))  # (B, 128, H/4, W/4)

        # 4. Decoder
        up1 = self.ps1(self.up_exp1(f_bn))  # (B, 64, H/2, W/2)
        d1 = self.dec1(torch.cat([up1, f_enc1], dim=1))  # (B, 64, H/2, W/2)

        up2 = self.ps2(self.up_exp2(d1))  # (B, 32, H, W)
        d2 = self.dec2(torch.cat([up2, f_fused], dim=1))  # (B, 32, H, W)

        return self.head(d2)

    def switch_to_deploy(self):
        """Recursively fold all re-parameterization modules for inference."""
        for module in self.modules():
            if hasattr(module, "switch_to_deploy") and module is not self:
                module.switch_to_deploy()
