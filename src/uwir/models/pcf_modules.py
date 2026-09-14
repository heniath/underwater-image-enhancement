"""
pcf_modules.py
--------------
Modules adapted from PCF-Net (Remote Sensing 2026) for underwater image restoration:
  1. Stabilized HSV-CS Color Space Conversion (rgb_to_hsv_cs)
  2. Color-Bias-Aware Module (ColorBiasAwareModule)
  3. Value-Confidence Module (ValueConfidenceModule)
  4. Channel-Spatial Adaptive Gated Fusion (CSAGF)
  5. Structural Re-parameterization Depthwise Conv (RepDepthwiseConv2d)
  6. Re-parameterizable Depthwise Separable Conv (RepDSC)
  7. Re-parameterizable 3x3 Conv Block (Rep3C)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Stabilized HSV-CS Conversion
# ---------------------------------------------------------------------------
def rgb_to_hsv_cs(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert an RGB tensor in [0, 1] to the stabilized 4-channel HSV-CS representation:
    (H_C, H_S, S, V), all scaled to [0, 1].

    Args:
        x: Input tensor of shape (B, 3, H, W) in range [0, 1].
        eps: Small constant to avoid zero division.

    Returns:
        Tensor of shape (B, 4, H, W) where channels are (H_C, H_S, S, V).
    """
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]

    max_c, _ = torch.max(x, dim=1, keepdim=True)
    min_c, _ = torch.min(x, dim=1, keepdim=True)
    delta = max_c - min_c

    # Value (V) in [0, 1]
    v = max_c

    # For achromatic pixels (gray/black/white where delta is ~0), hue and saturation are undefined.
    # We use a threshold to prevent gradient explosion from dividing by tiny numbers.
    is_chromatic = (delta >= 1e-4)
    safe_delta = torch.where(is_chromatic, delta, torch.ones_like(delta))
    s = torch.where(max_c > 1e-4, delta / torch.clamp(max_c, min=1e-4), torch.zeros_like(delta))

    # Masks for which channel is max
    is_r = (max_c == r)
    is_g = (max_c == g) & (~is_r)

    # Standard HSV hue in [0, 1)
    h_r = (g - b) / (6.0 * safe_delta)
    h_g = (1.0 / 3.0) + (b - r) / (6.0 * safe_delta)
    h_b = (2.0 / 3.0) + (r - g) / (6.0 * safe_delta)

    h = torch.where(is_r, h_r, torch.where(is_g, h_g, h_b))
    h = torch.where(is_chromatic, torch.remainder(h, 1.0), torch.zeros_like(h))

    # Map to radians [0, 2*pi)
    h_rad = 2.0 * math.pi * h

    # Stabilized Sine-Cosine Hue representation scaled to [0, 1]
    h_c = (torch.cos(h_rad) + 1.0) * 0.5
    h_s = (torch.sin(h_rad) + 1.0) * 0.5

    return torch.cat([h_c, h_s, s, v], dim=1)


# ---------------------------------------------------------------------------
# 2. Color-Bias-Aware Module
# ---------------------------------------------------------------------------
class ColorBiasAwareModule(nn.Module):
    """
    Color-Bias-Aware module from PCF-Net.
    Estimates pixel-wise red attenuation prior and predicts a refinement weight map W_cb.
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Args:
            rgb: RGB image tensor (B, 3, H, W) in [0, 1].
        Returns:
            W_cb: Color-bias weight map of shape (B, 1, H, W) in [0, 1].
        """
        r = rgb[:, 0:1, :, :]
        g = rgb[:, 1:2, :, :]
        b = rgb[:, 2:3, :, :]

        w_prior = 1.0 - (r / (r + g + b + eps))
        w_tilde = self.net(rgb)
        return w_tilde * w_prior


# ---------------------------------------------------------------------------
# 3. Value-Confidence Module
# ---------------------------------------------------------------------------
class ValueConfidenceModule(nn.Module):
    """
    Value-Confidence module from PCF-Net.
    Suppresses unreliable hue/saturation and noisy physical estimates in dark regions.
    """

    def __init__(self, mid_channels: int = 16, v_low: float = 0.1, gamma: float = 20.0):
        super().__init__()
        self.v_low = v_low
        self.gamma = gamma
        self.net = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, v_channel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_channel: V (Value/Brightness) channel of shape (B, 1, H, W).
        Returns:
            C_conf: Confidence weight map of shape (B, 1, H, W) in [0, 1].
        """
        c_enh = v_channel * torch.sigmoid(self.gamma * (v_channel - self.v_low))
        c_tilde = self.net(c_enh)
        return c_tilde


# ---------------------------------------------------------------------------
# 4. Channel-Spatial Adaptive Gated Fusion (CSAGF)
# ---------------------------------------------------------------------------
class CSAGF(nn.Module):
    """
    Channel-Spatial Adaptive Gated Fusion from PCF-Net.
    Dynamically integrates RGB/Physical and modulated HSV-CS streams using
    cross-modal channel attention and spatial gating.
    """

    def __init__(self, channels: int = 64, reduction: int = 4):
        super().__init__()
        self.channels = channels
        mid_ch = max(channels * 2 // reduction, 8)

        # Cross-modal Channel Attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, mid_ch, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_ch, channels * 2, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Adaptive Gating
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb: torch.Tensor, f_hsv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_rgb: Spatial/Physical feature map of shape (B, C, H, W).
            f_hsv: Modulated HSV-CS feature map of shape (B, C, H, W).
        Returns:
            f_fused: Fused feature map of shape (B, C, H, W).
        """
        f_cat = torch.cat([f_rgb, f_hsv], dim=1)  # (B, 2C, H, W)

        # 1. Channel Attention
        a_ch = self.channel_gate(f_cat)  # (B, 2C, 1, 1)
        a_ch_rgb, a_ch_hsv = torch.split(a_ch, self.channels, dim=1)

        f_rgb_prime = f_rgb * a_ch_rgb
        f_hsv_prime = f_hsv * a_ch_hsv

        # 2. Spatial Gating
        f_prime_cat = torch.cat([f_rgb_prime, f_hsv_prime], dim=1)  # (B, 2C, H, W)
        a_sp = self.spatial_gate(f_prime_cat)  # (B, 2, H, W)
        a_sp_rgb, a_sp_hsv = torch.split(a_sp, 1, dim=1)

        # 3. Fused Output
        f_fused = a_sp_rgb * f_rgb_prime + a_sp_hsv * f_hsv_prime
        return f_fused


# ---------------------------------------------------------------------------
# 5. Structural Re-parameterization Depthwise Conv
# ---------------------------------------------------------------------------
class RepDepthwiseConv2d(nn.Module):
    """
    Structural Re-parameterizable Depthwise 3x3 Convolution.

    During Training:
      Branch 1: 3x3 Depthwise Conv + BN
      Branch 2: 1x1 Depthwise Conv + BN
      Branch 3: Identity BN
      Output: branch1(x) + branch2(x) + branch3(x)

    During Deployment (Inference):
      All 3 branches are mathematically folded into a single standard 3x3 Depthwise Conv.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.is_deployed = False

        # Multi-branch during training
        self.conv3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.bn3x3 = nn.BatchNorm2d(channels)

        self.conv1x1 = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0, groups=channels, bias=False
        )
        self.bn1x1 = nn.BatchNorm2d(channels)

        self.bn_identity = nn.BatchNorm2d(channels)

        # Container for deployed conv
        self.deployed_conv: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.deployed_conv(x)

        return self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x)) + self.bn_identity(x)

    def _fuse_conv_bn(self, conv: Optional[nn.Conv2d], bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse Conv2d (or identity if conv is None) + BatchNorm2d into weight and bias."""
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        scale = gamma / std  # (C,)

        if conv is None:
            # Identity kernel of shape (C, 1, 3, 3)
            weight = torch.zeros(self.channels, 1, 3, 3, device=scale.device, dtype=scale.dtype)
            weight[:, 0, 1, 1] = 1.0
            bias = beta - running_mean * scale
            weight = weight * scale.view(-1, 1, 1, 1)
            return weight, bias

        # conv has weight (C, 1, K, K)
        weight = conv.weight * scale.view(-1, 1, 1, 1)
        if conv.bias is not None:
            bias = (conv.bias - running_mean) * scale + beta
        else:
            bias = beta - running_mean * scale

        return weight, bias

    def switch_to_deploy(self):
        """Fold all multi-branch components into a single Conv2d layer."""
        if self.is_deployed:
            return

        # 1. Fuse 3x3 Conv + BN
        w3, b3 = self._fuse_conv_bn(self.conv3x3, self.bn3x3)

        # 2. Fuse 1x1 Conv + BN and pad to 3x3
        w1, b1 = self._fuse_conv_bn(self.conv1x1, self.bn1x1)
        w1_padded = F.pad(w1, [1, 1, 1, 1])

        # 3. Fuse Identity BN
        wi, bi = self._fuse_conv_bn(None, self.bn_identity)

        # 4. Total equivalent weight and bias
        total_w = w3 + w1_padded + wi
        total_b = b3 + b1 + bi

        # 5. Create standard Conv2d
        self.deployed_conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
            bias=True,
            device=total_w.device,
            dtype=total_w.dtype,
        )
        self.deployed_conv.weight.data.copy_(total_w)
        self.deployed_conv.bias.data.copy_(total_b)

        # Remove training branch parameters to free memory
        del self.conv3x3
        del self.bn3x3
        del self.conv1x1
        del self.bn1x1
        del self.bn_identity

        self.is_deployed = True


# ---------------------------------------------------------------------------
# 6. Re-parameterizable Depthwise Separable Conv (RepDSC)
# ---------------------------------------------------------------------------
class RepDSC(nn.Module):
    """
    Re-parameterizable Depthwise Separable Convolution block (RepDSC) from PCF-Net.
    Comprises:
      1. Depthwise stage: RepDepthwiseConv2d (3x3 + 1x1 + Identity BN during train; single 3x3 during deploy)
      2. Pointwise stage: 1x1 Conv + BN + ReLU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.depthwise = RepDepthwiseConv2d(in_ch)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        if self.stride > 1:
            out = F.max_pool2d(out, kernel_size=self.stride, stride=self.stride)
        return self.pointwise(out)

    def switch_to_deploy(self):
        self.depthwise.switch_to_deploy()


# ---------------------------------------------------------------------------
# 7. Re-parameterizable 3x3 Conv Block (Rep3C)
# ---------------------------------------------------------------------------
class Rep3C(nn.Module):
    """
    Re-parameterizable 3x3 Convolution Block (Rep3C) from PCF-Net.
    Used for feature refinement and skip-connection fusion in the decoder.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = RepDepthwiseConv2d(in_ch)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        return self.pointwise(out)

    def switch_to_deploy(self):
        self.depthwise.switch_to_deploy()

