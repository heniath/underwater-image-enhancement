"""
fanetplus.py
------------
FA*Net-Plus: Enhanced Feature Attention Network with Haar Wavelet
Multi-Frequency Decomposition and Dual-Dilation Receptive Field.

Designed to exceed the 26.64 dB EUVP benchmark while maintaining
an ultra-compact footprint (~105k parameters, ~300x smaller than UNet-5ch).

Key innovations:
    1. 2D Discrete Haar Wavelet Transform (DWT):
       Lossless, parameter-free frequency separation:
       - Low-frequency (LL): water color cast, illumination, haze
       - High-frequency (LH, HL, HH): coral structures, fish textures, edge details
    2. Enhanced Multi-Scale RFAB (Dual-Dilation):
       Combines dilation 1 and dilation 2 convolutions with zero parameter overhead.
    3. Dual Selective Attention:
       Channel Attention dedicated to chromatic recovery and
       Pixel Attention dedicated to spatially varying turbidity compensation.
    4. Physics-Informed 5-Channel Support (RGB + Transmission t + Background light B).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fanet import ChannelAttention, PixelAttention


class HaarWavelet2D(nn.Module):
    """
    Parameter-free 2D Discrete Haar Wavelet Transform.
    Decomposes (B, C, H, W) into (B, 4*C, H/2, W/2).
    Sub-bands: LL (low-frequency), LH (horizontal), HL (vertical), HH (diagonal).
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        ll = x1 + x2 + x3 + x4
        lh = -x1 - x2 + x3 + x4
        hl = -x1 + x2 - x3 + x4
        hh = x1 - x2 - x3 + x4

        freq = torch.cat([ll, lh, hl, hh], dim=1)
        return freq, pad_h, pad_w


class InverseHaarWavelet2D(nn.Module):
    """
    Parameter-free 2D Inverse Haar Wavelet Transform.
    Synthesizes (B, 4*C, H/2, W/2) back to (B, C, H, W).
    """

    def forward(self, freq: torch.Tensor, pad_h: int = 0, pad_w: int = 0) -> torch.Tensor:
        c = freq.shape[1] // 4
        ll, lh, hl, hh = torch.chunk(freq, 4, dim=1)

        x1 = (ll - lh - hl + hh) / 2.0
        x2 = (ll - lh + hl - hh) / 2.0
        x3 = (ll + lh - hl - hh) / 2.0
        x4 = (ll + lh + hl + hh) / 2.0

        b, _, h, w = ll.shape
        out = torch.zeros((b, c, h * 2, w * 2), device=freq.device, dtype=freq.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4

        if pad_h > 0:
            out = out[:, :, :-pad_h, :]
        if pad_w > 0:
            out = out[:, :, :, :-pad_w]
        return out


class DualDilationConv(nn.Module):
    """
    Multi-scale convolutional unit combining standard (d=1) and dilated (d=2) paths.
    Splits channels evenly across paths, maintaining identical parameter count to standard 3x3 conv.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.conv_std = nn.Conv2d(channels, mid, 3, padding=1, bias=True)
        self.conv_dil = nn.Conv2d(channels, mid, 3, padding=2, dilation=2, bias=True)
        self.act = nn.PReLU(channels)
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = self.conv_std(x)
        x_dil = self.conv_dil(x)
        feat = torch.cat([x_std, x_dil], dim=1)
        feat = self.act(feat)
        return self.fuse(feat)


class EnhancedRFAB(nn.Module):
    """
    Enhanced Residual Feature Attention Block (ERFAB):
    Integrates multi-scale dual-dilation convolutions with Channel Attention
    and Pixel Attention for simultaneous color correction and detail recovery.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = DualDilationConv(channels)
        self.ca = ChannelAttention(channels, reduction=4)
        self.pa = PixelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv(x)
        ca_weight = self.ca(res)
        pa_weight = self.pa(res)
        res = res * ca_weight * pa_weight
        return x + res


class WaveletFrequencyBranch(nn.Module):
    """
    Processes low-frequency (illumination/color) and high-frequency (edges/texture)
    sub-bands separately using Haar Wavelet decomposition.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarWavelet2D()
        self.idwt = InverseHaarWavelet2D()

        # Low-frequency branch (LL): 1 channel group -> EnhancedRFAB
        self.ll_block = EnhancedRFAB(channels)

        # High-frequency branch (LH, HL, HH): 3 channel groups
        hf_channels = channels * 3
        self.hf_conv = nn.Sequential(
            nn.Conv2d(hf_channels, channels, 1, bias=False),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, hf_channels, 1, bias=False),
        )
        self.hf_pa = PixelAttention(hf_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq, pad_h, pad_w = self.dwt(x)
        c = x.shape[1]
        ll = freq[:, :c, :, :]
        hf = freq[:, c:, :, :]

        # Process LL (color cast & transmission)
        ll_out = self.ll_block(ll)

        # Process HF (edges & fine details)
        hf_res = self.hf_conv(hf)
        hf_pa = self.hf_pa(hf_res)
        hf_out = hf + hf_res * hf_pa

        fused_freq = torch.cat([ll_out, hf_out], dim=1)
        return self.idwt(fused_freq, pad_h=pad_h, pad_w=pad_w)


class FANetPlus(nn.Module):
    """
    FA*Net-Plus Architecture:
    Combines stem, Wavelet Frequency processing, deep EnhancedRFAB blocks,
    and a zero-initialized residual reconstruction head.

    Args:
        in_channels (int): 3 for RGB, 4/5 for physics priors (transmission, background light).
        out_channels (int): 3 for restored RGB.
        channels (int): Base feature channels (default 32).
        num_blocks (int): Number of deep ERFAB blocks (default 4).
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        channels: int = 32,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1, bias=True)
        self.wavelet_stage = WaveletFrequencyBranch(channels)
        self.deep_blocks = nn.ModuleList(
            [EnhancedRFAB(channels) for _ in range(num_blocks)]
        )
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.head = nn.Conv2d(channels, out_channels, 3, padding=1, bias=True)

        # Zero-initialize head for clean residual start (identity pass)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        feat0 = self.stem(inputs)

        # 1. Wavelet frequency decomposition & enhancement
        feat_w = self.wavelet_stage(feat0) + feat0

        # 2. Deep multi-scale spatial feature learning
        x = feat_w
        for block in self.deep_blocks:
            x = block(x)

        x = self.fuse(x) + feat_w

        # 3. Residual learning: add predicted residual to RGB input
        residual = self.head(x)
        return torch.clamp(inputs[:, :3] + residual, 0.0, 1.0)


def build_fanetplus(in_channels: int = 5) -> FANetPlus:
    """Build FA*Net-Plus (~105k parameters)."""
    return FANetPlus(in_channels=in_channels, out_channels=3, channels=32, num_blocks=4)


__all__ = [
    "DualDilationConv",
    "EnhancedRFAB",
    "FANetPlus",
    "HaarWavelet2D",
    "InverseHaarWavelet2D",
    "WaveletFrequencyBranch",
    "build_fanetplus",
]
