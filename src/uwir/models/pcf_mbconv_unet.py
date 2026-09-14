"""
pcf_mbconv_unet.py
------------------
PCF-MBConv U-Net: Dual-Branch Underwater Image Restoration Network.

Architecture:
  - Branch 1: Spatial/Physical Stream (RGB + UDCP t(x), B(x) -> 5 channels)
  - Branch 2: Color/Contrast Stream (Stabilized HSV-CS -> 4 channels)
  - Pre-filtering:
      - Color-Bias-Aware Module (W_cb)
      - Value-Confidence Module (C_conf)
      - Modulation: F_hsv_w = ReLU(Conv(F_hsv * (W_cb * C_conf)))
  - Fusion: Channel-Spatial Adaptive Gated Fusion (CSAGF) -> 64 channels
  - Backbone: Rep-MBConv U-Net (64, 128, 192, 256 -> 512 bottleneck)
  - Re-parameterization: switch_to_deploy() folds training branches into single Conv layers.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcf_modules import (
    CSAGF,
    ColorBiasAwareModule,
    RepDepthwiseConv2d,
    ValueConfidenceModule,
    rgb_to_hsv_cs,
)


class RepMBConvBlock(nn.Module):
    """
    Inverted Residual Block using Structural Re-parameterization in the depthwise stage.
    """

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 2):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = in_ch == out_ch

        self.expand = (
            nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
            )
            if expand_ratio != 1
            else nn.Identity()
        )

        self.depthwise = nn.Sequential(
            RepDepthwiseConv2d(mid_ch),
            nn.ReLU6(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_res:
            return x + out
        return out


class RepDoubleConv(nn.Module):
    """Two consecutive RepMBConvBlocks."""

    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            RepMBConvBlock(in_ch, out_ch, expand_ratio),
            RepMBConvBlock(out_ch, out_ch, expand_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepDown(nn.Module):
    """Downsampling block: MaxPool2d + RepDoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            RepDoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepUp(nn.Module):
    """Upsampling block: Bilinear/ConvTranspose2d + 1x1 compression + RepDoubleConv."""

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(prev_ch, prev_ch // 2, kernel_size=1, bias=False),
            )
            combined_ch = (prev_ch // 2) + skip_ch
        else:
            self.up = nn.ConvTranspose2d(prev_ch, prev_ch // 2, kernel_size=2, stride=2)
            combined_ch = (prev_ch // 2) + skip_ch

        self.compress = nn.Sequential(
            nn.Conv2d(combined_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

        self.conv = RepDoubleConv(out_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        d_y = x2.size(2) - x1.size(2)
        d_x = x2.size(3) - x1.size(3)
        if d_x != 0 or d_y != 0:
            x1 = F.pad(x1, [d_x // 2, d_x - d_x // 2, d_y // 2, d_y - d_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.compress(x)
        return self.conv(x)


class PCFMBConvUNet(nn.Module):
    """
    Dual-Branch PCF-MBConv U-Net for Underwater Image Restoration.

    Args:
        in_channels: Total input channels (typically 5 for RGB + t(x) + B(x), or 3 for RGB).
        out_channels: Output channels (default 3 for enhanced RGB).
        features: Channel progression across 4 encoder stages. Default: (64, 128, 192, 256).
        bilinear: If True, uses bilinear interpolation for upsampling; otherwise ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: Tuple[int, int, int, int] = (64, 128, 192, 256),
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        f = features

        # ------------------------------------------------------------------
        # 1. Dual-Branch Shallow Feature Extractors
        # ------------------------------------------------------------------
        # Branch 1: Spatial/Physical Stream (RGB + UDCP: 5 ch -> 64 ch)
        self.shallow_phys = nn.Sequential(
            nn.Conv2d(in_channels, f[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f[0]),
            nn.ReLU6(inplace=True),
        )

        # Branch 2: Color/Contrast Stream (HSV-CS: 4 ch -> 64 ch)
        self.shallow_hsv = nn.Sequential(
            nn.Conv2d(4, f[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f[0]),
            nn.ReLU6(inplace=True),
        )

        # ------------------------------------------------------------------
        # 2. Color-Bias & Value-Confidence Modules
        # ------------------------------------------------------------------
        self.color_bias = ColorBiasAwareModule(in_channels=3, mid_channels=16)
        self.value_conf = ValueConfidenceModule(mid_channels=16)

        # Modulation projection for HSV-CS feature stream
        self.hsv_mod_conv = nn.Sequential(
            nn.Conv2d(f[0], f[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f[0]),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # 3. Adaptive Fusion: CSAGF
        # ------------------------------------------------------------------
        self.csagf = CSAGF(channels=f[0])

        # ------------------------------------------------------------------
        # 4. Backbone Encoder (Rep-MBConv)
        # ------------------------------------------------------------------
        self.enc1 = RepDoubleConv(f[0], f[0])
        self.enc2 = RepDown(f[0], f[1])
        self.enc3 = RepDown(f[1], f[2])
        self.enc4 = RepDown(f[2], f[3])

        self.bottleneck = RepDown(f[3], f[3] * 2)  # 256 -> 512

        # ------------------------------------------------------------------
        # 5. Backbone Decoder (Rep-MBConv)
        # ------------------------------------------------------------------
        self.dec4 = RepUp(f[3] * 2, f[3], f[3], bilinear)
        self.dec3 = RepUp(f[3], f[2], f[2], bilinear)
        self.dec2 = RepUp(f[2], f[1], f[1], bilinear)
        self.dec1 = RepUp(f[1], f[0], f[0], bilinear)

        # ------------------------------------------------------------------
        # 6. Output Reconstruction Head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(f[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 5, H, W) or (B, 3, H, W).
        Returns:
            Enhanced RGB image of shape (B, 3, H, W) in [0, 1].
        """
        # Automatic fallback if 3-channel RGB is passed into a 5-channel model
        if x.shape[1] < self.in_channels:
            pad_ch = self.in_channels - x.shape[1]
            x = torch.cat([x, x[:, :pad_ch, :, :]], dim=1)

        # Split RGB from physical channels
        rgb = x[:, :3, :, :]

        # 1. Branch 1: Spatial/Physical Shallow Features
        f_rgb = self.shallow_phys(x)

        # 2. Branch 2: HSV-CS Color Features & Pre-filtering
        hsv_cs = rgb_to_hsv_cs(rgb)
        f_hsv = self.shallow_hsv(hsv_cs)

        # Color-Bias-Aware weight & Value-Confidence weight
        w_cb = self.color_bias(rgb)
        c_conf = self.value_conf(hsv_cs[:, 3:4, :, :])
        w_final = w_cb * c_conf

        # Modulate HSV-CS features
        f_hsv_mod = self.hsv_mod_conv(f_hsv * w_final)

        # 3. Channel-Spatial Adaptive Gated Fusion
        f_fused = self.csagf(f_rgb, f_hsv_mod)

        # 4. Backbone Encoder
        e1 = self.enc1(f_fused)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)

        # 5. Backbone Decoder
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # 6. Output Head
        return self.head(d1)

    def switch_to_deploy(self):
        """
        Recursively fold all multi-branch re-parameterization modules into single convolutions.
        Call this before model profiling or exporting checkpoints for inference.
        """
        for module in self.modules():
            if hasattr(module, "switch_to_deploy") and module is not self:
                module.switch_to_deploy()
