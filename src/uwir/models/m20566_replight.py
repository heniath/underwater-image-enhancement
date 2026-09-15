"""
m20566_replight.py
------------------
Ultra-lightweight (<200k params) redesigned architectures for underwater image enhancement:
  1. M20566RepLight:
     - Dual-branch: RGB/Physical stream + Stabilized HSV-CS (Sine-Cosine) stream.
     - Perception-driven modulation: Color-Bias-Aware + Value-Confidence.
     - Channel-Spatial Adaptive Gated Fusion (CSAGF).
     - RepDSC (Depthwise Separable + Multi-Branch Structural Re-parameterization).
     - PixelShuffle sub-pixel reconstruction (scale=2) for crisp high frequencies.
     - Residual Learning Framework: J_hat = clamp(I + Delta, 0, 1) to eliminate dark contrast collapse.
     - Footprint: ~89.3k params (3ch) / ~89.9k params (5ch), ~1.97 GFLOPs.

  2. M20566LMF (Variant 1 - LMF Physical Re-synthesis & Cycle Style Learning):
     - Predicts transmission map t(x) and background light B along with scene radiance J.
     - Forward optical degradation re-synthesis: I_redeg = J * t + B * (1 - t).
     - Enables physical cycle-consistency loss and forward style transfer / synthetic data generation.
     - Footprint: ~90.9k params (3ch) / ~91.5k params (5ch), ~1.98 GFLOPs.

  3. M20566LCS (Variant 2 - LCS-Net Hybrid):
     - LCCM (Learnable Color Correction Module) for zero-overhead global color cast normalization.
     - SMSDB (Selective Multi-Scale Dilated Block) at bottleneck with dilations 1, 2, 3 + channel attention.
     - Footprint: ~145.6k params (3ch) / ~146.2k params (5ch), ~2.18 GFLOPs.
"""

from typing import Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Auxiliary Modules from LCS-Net & Depthwise Blocks
# ---------------------------------------------------------------------------


class LCCM(nn.Module):
    """
    Learnable Color Correction Module (from LCS-Net, Sensors 2026).
    Predicts affine transformation factors (alpha, beta) from global channel statistics.
    Parameter cost: ~150 parameters.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.mlp(x)
        alpha = torch.sigmoid(stats[:, :3]).view(-1, 3, 1, 1) * 2.0
        beta = torch.tanh(stats[:, 3:]).view(-1, 3, 1, 1) * 0.25
        return torch.clamp(alpha * x + beta, 0.0, 1.0)


class DepthwiseSeparableConv(nn.Module):
    """Lightweight Depthwise Separable Conv with HardSwish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw(x)))
        return self.act2(self.bn2(self.pw(x)))


class LightweightSMSDB(nn.Module):
    """
    Selective Multi-Scale Dilated Block (SMSDB from LCS-Net) with channel bottleneck.
    Aggregates multi-receptive-field features (dilation rates 1, 2, 3) followed by
    channel-wise attention recalibration.
    """

    def __init__(self, channels: int = 128, mid_channels: int = 64):
        super().__init__()
        self.reduce = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_channels)

        self.d1 = DepthwiseSeparableConv(mid_channels, mid_channels, dilation=1)
        self.d2 = DepthwiseSeparableConv(mid_channels, mid_channels, dilation=2)
        self.d3 = DepthwiseSeparableConv(mid_channels, mid_channels, dilation=3)

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, channels // 4), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_red = F.relu(self.bn0(self.reduce(x)), inplace=True)
        f1 = self.d1(x_red)
        f2 = self.d2(x_red)
        f3 = self.d3(x_red)
        fused = self.fuse(torch.cat([f1, f2, f3], dim=1))
        w = self.ca(fused)
        return x + fused * w


# ---------------------------------------------------------------------------
# Base Architecture: M20566RepLight
# ---------------------------------------------------------------------------


class M20566RepLight(nn.Module):
    """
    Redesigned M20566 architecture under 200k parameters (~89.3k params).
    Employs RepDSC structural re-parameterization, stabilized HSV-CS, CSAGF adaptive fusion,
    PixelShuffle upsampling, and residual learning.

    Args:
        in_channels: Input channels (3 for RGB, 5 for RGB + physics [t, B]). Default: 3.
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
        c = base_channels

        # 1. Dual-Branch Feature Extraction
        self.branch_phys = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            RepDSC(c, c),
        )

        self.branch_hsv = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            RepDSC(c, c),
        )

        # 2. Color-Bias-Aware & Value-Confidence Modulation
        self.color_bias = ColorBiasAwareModule(in_channels=3, mid_channels=16)
        self.value_conf = ValueConfidenceModule(mid_channels=16)
        self.hsv_mod_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # 3. CSAGF Fusion
        self.csagf = CSAGF(channels=c)

        # 4. Lightweight 2-Stage Encoder
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = RepDSC(c, c * 2)  # 32 -> 64

        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = RepDSC(c * 2, c * 4)  # 64 -> 128

        # 5. Lightweight 2-Stage Decoder with PixelShuffle
        self.up_exp1 = RepDSC(c * 4, (c * 2) * 4)  # 128 -> 256
        self.ps1 = nn.PixelShuffle(2)  # 256 -> 64
        self.dec1 = Rep3C(c * 2 * 2, c * 2)  # (64 + 64) -> 64

        self.up_exp2 = RepDSC(c * 2, c * 4)  # 64 -> 128
        self.ps2 = nn.PixelShuffle(2)  # 128 -> 32
        self.dec2 = Rep3C(c * 2, c)  # (32 + 32) -> 32

        # 6. Residual Output Head
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)
        # Zero-initialize so initial output is the exact identity J = I (Delta = 0)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.in_channels:
            pad_ch = self.in_channels - x.shape[1]
            x = torch.cat([x, x[:, :pad_ch, :, :]], dim=1)

        rgb = x[:, :3, :, :]

        # Dual-branch extraction
        f_phys = self.branch_phys(x)
        hsv_cs = rgb_to_hsv_cs(rgb)
        f_hsv = self.branch_hsv(hsv_cs)

        # Modulation
        w_cb = self.color_bias(rgb)
        c_conf = self.value_conf(hsv_cs[:, 3:4, :, :])
        f_hsv_mod = self.hsv_mod_conv(f_hsv * (w_cb * c_conf))

        # CSAGF Adaptive Gated Fusion
        f_fused = self.csagf(f_phys, f_hsv_mod)

        # Encoder
        f_enc1 = self.enc1(self.pool1(f_fused))
        f_bn = self.bottleneck(self.pool2(f_enc1))

        # Decoder with PixelShuffle
        up1 = self.ps1(self.up_exp1(f_bn))
        d1 = self.dec1(torch.cat([up1, f_enc1], dim=1))

        up2 = self.ps2(self.up_exp2(d1))
        d2 = self.dec2(torch.cat([up2, f_fused], dim=1))

        # Residual Learning Output: J_hat = clamp(I + Delta, 0, 1)
        delta = self.head(d2)
        return torch.clamp(rgb + delta, 0.0, 1.0)

    def switch_to_deploy(self):
        """Fold all structural re-parameterization branches for inference."""
        for module in self.modules():
            if hasattr(module, "switch_to_deploy") and module is not self:
                module.switch_to_deploy()


# ---------------------------------------------------------------------------
# Variant 1: M20566LMF (Physical Re-synthesis & Cycle Style Learning)
# ---------------------------------------------------------------------------


class M20566LMF(M20566RepLight):
    """
    M20566 Variant 1 with Forward Optical Degradation Re-synthesis (from LMF-Net).
    Predicts transmission map t(x) and background light B, re-synthesizing:
        I_redeg = J * t + B * (1 - t)
    This provides self-supervised optical constraints and enables forward style transfer.
    Footprint: ~90.9k params (3ch) / ~91.5k params (5ch).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        c = base_channels

        # Transmission map head: t(x) in [0.05, 1.0]
        self.head_t = nn.Sequential(
            RepDSC(c, 16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Background light head: B in [0, 1]^3
        self.head_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, return_physics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if x.shape[1] < self.in_channels:
            pad_ch = self.in_channels - x.shape[1]
            x = torch.cat([x, x[:, :pad_ch, :, :]], dim=1)

        rgb = x[:, :3, :, :]

        f_phys = self.branch_phys(x)
        hsv_cs = rgb_to_hsv_cs(rgb)
        f_hsv = self.branch_hsv(hsv_cs)

        w_cb = self.color_bias(rgb)
        c_conf = self.value_conf(hsv_cs[:, 3:4, :, :])
        f_hsv_mod = self.hsv_mod_conv(f_hsv * (w_cb * c_conf))

        f_fused = self.csagf(f_phys, f_hsv_mod)

        f_enc1 = self.enc1(self.pool1(f_fused))
        f_bn = self.bottleneck(self.pool2(f_enc1))

        up1 = self.ps1(self.up_exp1(f_bn))
        d1 = self.dec1(torch.cat([up1, f_enc1], dim=1))

        up2 = self.ps2(self.up_exp2(d1))
        d2 = self.dec2(torch.cat([up2, f_fused], dim=1))

        delta = self.head(d2)
        j = torch.clamp(rgb + delta, 0.0, 1.0)

        if return_physics or self.training:
            t = torch.clamp(self.head_t(d2), 0.05, 1.0)
            bg = self.head_b(d2).view(-1, 3, 1, 1)
            i_redeg = j * t + bg * (1.0 - t)
            if return_physics:
                return j, i_redeg, t, bg
            return j, i_redeg

        return j


# ---------------------------------------------------------------------------
# Variant 2: M20566LCS (LCS-Net Hybrid with LCCM + SMSDB)
# ---------------------------------------------------------------------------


class M20566LCS(M20566RepLight):
    """
    M20566 Variant 2 integrating key components from LCS-Net (Sensors 2026):
      1. LCCM: Learnable Color Correction Module at input for global cast normalization.
      2. SMSDB: Selective Multi-Scale Dilated Block at bottleneck (dilations 1, 2, 3 + CA).
    Footprint: ~145.6k params (3ch) / ~146.2k params (5ch).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        c = base_channels
        self.lccm = LCCM()
        self.bottleneck_smsdb = LightweightSMSDB(channels=c * 4, mid_channels=c * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.in_channels:
            pad_ch = self.in_channels - x.shape[1]
            x = torch.cat([x, x[:, :pad_ch, :, :]], dim=1)

        rgb = x[:, :3, :, :]

        # LCCM color pre-normalization
        rgb_norm = self.lccm(rgb)
        if x.shape[1] > 3:
            x = torch.cat([rgb_norm, x[:, 3:, :, :]], dim=1)
        else:
            x = rgb_norm

        f_phys = self.branch_phys(x)
        hsv_cs = rgb_to_hsv_cs(rgb_norm)
        f_hsv = self.branch_hsv(hsv_cs)

        w_cb = self.color_bias(rgb_norm)
        c_conf = self.value_conf(hsv_cs[:, 3:4, :, :])
        f_hsv_mod = self.hsv_mod_conv(f_hsv * (w_cb * c_conf))

        f_fused = self.csagf(f_phys, f_hsv_mod)

        f_enc1 = self.enc1(self.pool1(f_fused))
        f_bn = self.bottleneck(self.pool2(f_enc1))

        # SMSDB Selective Multi-Scale Context Aggregation
        f_bn = self.bottleneck_smsdb(f_bn)

        up1 = self.ps1(self.up_exp1(f_bn))
        d1 = self.dec1(torch.cat([up1, f_enc1], dim=1))

        up2 = self.ps2(self.up_exp2(d1))
        d2 = self.dec2(torch.cat([up2, f_fused], dim=1))

        delta = self.head(d2)
        return torch.clamp(rgb + delta, 0.0, 1.0)
