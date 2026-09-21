"""
fgdpa.py
--------
Frequency-Guided Dual-Path Attention Network (FGDPA) for Real-Time Underwater
Image Enhancement (ICME 2026).

Paper: "Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention"
arXiv: https://arxiv.org/abs/2606.30314
Official Repo: https://github.com/LethyZhang/FGDPA

Key Features:
- MBRConv-DCT: Multi-branch re-parameterizable convolution with fixed DCT priors
  during training, collapsible to standard Conv2d at inference (zero extra cost).
- FGDPA: Frequency-Guided Dual-Path Attention combining spatial and spectral
  (low-resolution FFT magnitude) representations.
- Ultra-lightweight: ~4.234K parameters in slim/inference mode.
- High throughput: 600+ FPS reported (800+ FPS on RTX 5070).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. DCT Basis and Fixed Frequency Prior Helpers
# ===========================================================================

def _make_1d_dct_basis(n: int) -> torch.Tensor:
    x = torch.arange(n).float()
    k = torch.arange(n).float().unsqueeze(1)
    basis = torch.cos(math.pi * (x + 0.5) * k / n)
    return basis


def _make_2d_dct_bank(ksize: int, K: int) -> torch.Tensor:
    assert ksize in (3, 5)
    B = _make_1d_dct_basis(ksize)
    pairs = []
    for u in range(ksize):
        for v in range(ksize):
            if not (u == 0 and v == 0):  # exclude DC
                pairs.append((u, v))
    pairs = pairs[:K]
    bank = []
    for u, v in pairs:
        k2d = torch.ger(B[u], B[v])
        bank.append(k2d)
    if len(bank) < K:
        for _ in range(K - len(bank)):
            bank.append(torch.ger(B[0], B[1]))
    return torch.stack(bank, dim=0)


def _zero_mean_unit_norm(k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    k = k - k.mean(dim=(-2, -1), keepdim=True)
    n = torch.linalg.vector_norm(k, ord=2, dim=(-2, -1), keepdim=True)
    return k / (n + eps)


def _center_to_3x3(kernel_ksize: int, k: torch.Tensor) -> torch.Tensor:
    if kernel_ksize == 3:
        return k
    if kernel_ksize == 5:
        return k[..., 1:4, 1:4]
    raise ValueError(f"Unsupported kernel size {kernel_ksize}")


class DCTLinearBranch(nn.Module):
    """Linear depthwise DCT branch with learnable pointwise projection and gate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 4,
        ksize: int = 3,
        use_gate: bool = True,
    ):
        super().__init__()
        assert ksize == 3
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.K = int(K)
        self.ksize = 3

        self.dw = nn.Conv2d(
            self.in_channels,
            self.in_channels * self.K,
            kernel_size=self.ksize,
            padding=1,
            groups=self.in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(
            self.in_channels * self.K,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

        self.use_gate = use_gate
        if self.use_gate:
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("gamma_dummy", torch.tensor(1.0))

        with torch.no_grad():
            k_bank = _make_2d_dct_bank(self.ksize, self.K)
            k_bank = _zero_mean_unit_norm(k_bank)
            k_bank = k_bank.unsqueeze(0).repeat(self.in_channels, 1, 1, 1)
            k_bank = k_bank.reshape(self.in_channels * self.K, 1, 3, 3)
            self.dw.weight.copy_(k_bank)

        for p in self.dw.parameters():
            p.requires_grad = False

        nn.init.zeros_(self.pw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        if self.use_gate:
            y = y * torch.tanh(self.gamma)
        return y

    def get_equivalent_3x3_weight(self) -> torch.Tensor:
        device = self.pw.weight.device
        dtype = self.pw.weight.dtype
        DW = self.dw.weight.detach()
        CK = DW.shape[0]
        K = self.K
        C = CK // K
        O = self.pw.weight.shape[0]

        DW_3x3 = DW.view(C, K, 3, 3)
        PW = self.pw.weight.view(O, C, K)
        W = torch.einsum("ock,ckhw->ochw", PW, DW_3x3)
        if self.use_gate:
            W = W * torch.tanh(self.gamma).to(dtype=dtype, device=device)
        return W

    def get_equivalent_bias(self) -> torch.Tensor:
        device = self.pw.weight.device
        dtype = self.pw.weight.dtype
        return torch.zeros(self.pw.weight.shape[0], device=device, dtype=dtype)


# ===========================================================================
# 2. Multi-Branch Re-parameterizable Convolutions (MBRConv)
# ===========================================================================

class MBRConv5(nn.Module):
    """5x5 Multi-Branch Re-parameterizable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid = out_channels * rep_scale

        self.conv = nn.Conv2d(in_channels, mid, 5, 1, 2)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv2 = nn.Conv2d(in_channels, mid, 3, 1, 1)
        self.conv2_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv_crossh = nn.Conv2d(in_channels, mid, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv_crossv = nn.Conv2d(in_channels, mid, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv_out = nn.Conv2d(mid * 10, out_channels, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat(
            [
                x1,
                x2,
                x3,
                x4,
                x5,
                self.conv_bn(x1),
                self.conv1_bn(x2),
                self.conv2_bn(x3),
                self.conv_crossh_bn(x4),
                self.conv_crossv_bn(x5),
            ],
            dim=1,
        )
        return self.conv_out(x)

    def slim(self) -> tuple[torch.Tensor, torch.Tensor]:
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        conv1_weight = F.pad(self.conv1.weight, (2, 2, 2, 2))
        conv1_bias = self.conv1.bias
        conv2_weight = F.pad(self.conv2.weight, (1, 1, 1, 1))
        conv2_bias = self.conv2.bias
        conv_crossv_weight = F.pad(self.conv_crossv.weight, (1, 1, 2, 2))
        conv_crossv_bias = self.conv_crossv.bias
        conv_crossh_weight = F.pad(self.conv_crossh.weight, (2, 2, 1, 1))
        conv_crossh_bias = self.conv_crossh.bias

        def fuse_bn(conv_w, conv_b, bn_seq, pad_shape=None):
            bn = bn_seq[0]
            k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv_w * k.view(-1, 1, 1, 1)
            b = (conv_b - bn.running_mean) * k + bn.bias
            if pad_shape is not None:
                w = F.pad(w, pad_shape)
            return w, b

        c_bn_w, c_bn_b = fuse_bn(self.conv.weight, self.conv.bias, self.conv_bn)
        c1_bn_w, c1_bn_b = fuse_bn(self.conv1.weight, self.conv1.bias, self.conv1_bn, (2, 2, 2, 2))
        c2_bn_w, c2_bn_b = fuse_bn(self.conv2.weight, self.conv2.bias, self.conv2_bn, (1, 1, 1, 1))
        cv_bn_w, cv_bn_b = fuse_bn(self.conv_crossv.weight, self.conv_crossv.bias, self.conv_crossv_bn, (1, 1, 2, 2))
        ch_bn_w, ch_bn_b = fuse_bn(self.conv_crossh.weight, self.conv_crossh.bias, self.conv_crossh_bn, (2, 2, 1, 1))

        weight_all = torch.cat(
            [conv_weight, conv1_weight, conv2_weight, conv_crossh_weight, conv_crossv_weight,
             c_bn_w, c1_bn_w, c2_bn_w, ch_bn_w, cv_bn_w],
            dim=0,
        )
        bias_all = torch.cat(
            [conv_bias, conv1_bias, conv2_bias, conv_crossh_bias, conv_crossv_bias,
             c_bn_b, c1_bn_b, c2_bn_b, ch_bn_b, cv_bn_b],
            dim=0,
        )

        weight_compress = self.conv_out.weight.view(self.conv_out.out_channels, -1)
        W_flat = weight_all.view(weight_all.size(0), -1)
        weight = torch.matmul(weight_compress, W_flat).view(self.conv_out.out_channels, self.in_channels, 5, 5)
        bias = torch.matmul(weight_compress, bias_all)
        if self.conv_out.bias is not None:
            bias = bias + self.conv_out.bias
        return weight, bias


class MBRConv3(nn.Module):
    """3x3 Multi-Branch Re-parameterizable Convolution with DCT frequency branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rep_scale: int = 4,
        use_dct: bool = True,
        dct_K: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        self.use_dct = use_dct
        mid = out_channels * rep_scale

        self.conv = nn.Conv2d(in_channels, mid, 3, 1, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv_crossh = nn.Conv2d(in_channels, mid, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv_crossv = nn.Conv2d(in_channels, mid, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(mid))

        if self.use_dct:
            self.dct_branch = DCTLinearBranch(
                in_channels=in_channels,
                out_channels=mid,
                K=dct_K,
                ksize=3,
                use_gate=True,
            )
            self.conv_out = nn.Conv2d(mid * 9, out_channels, 1)
        else:
            self.dct_branch = None
            self.conv_out = nn.Conv2d(mid * 8, out_channels, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        feats = [
            x0, x1, x2, x3,
            self.conv_bn(x0),
            self.conv1_bn(x1),
            self.conv_crossh_bn(x2),
            self.conv_crossv_bn(x3),
        ]
        if self.dct_branch is not None:
            feats.append(self.dct_branch(inp))
        x = torch.cat(feats, dim=1)
        return self.conv_out(x)

    def slim(self) -> tuple[torch.Tensor, torch.Tensor]:
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        conv1_weight = F.pad(self.conv1.weight, (1, 1, 1, 1))
        conv1_bias = self.conv1.bias
        conv_crossh_weight = F.pad(self.conv_crossh.weight, (1, 1, 0, 0))
        conv_crossh_bias = self.conv_crossh.bias
        conv_crossv_weight = F.pad(self.conv_crossv.weight, (0, 0, 1, 1))
        conv_crossv_bias = self.conv_crossv.bias

        def fuse_bn(conv_w, conv_b, bn_seq, pad_shape=None):
            bn = bn_seq[0]
            k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv_w * k.view(-1, 1, 1, 1)
            b = (conv_b - bn.running_mean) * k + bn.bias
            if pad_shape is not None:
                w = F.pad(w, pad_shape)
            return w, b

        c_bn_w, c_bn_b = fuse_bn(self.conv.weight, self.conv.bias, self.conv_bn)
        c1_bn_w, c1_bn_b = fuse_bn(self.conv1.weight, self.conv1.bias, self.conv1_bn, (1, 1, 1, 1))
        ch_bn_w, ch_bn_b = fuse_bn(self.conv_crossh.weight, self.conv_crossh.bias, self.conv_crossh_bn, (1, 1, 0, 0))
        cv_bn_w, cv_bn_b = fuse_bn(self.conv_crossv.weight, self.conv_crossv.bias, self.conv_crossv_bn, (0, 0, 1, 1))

        weight_list = [
            conv_weight, conv1_weight, conv_crossh_weight, conv_crossv_weight,
            c_bn_w, c1_bn_w, ch_bn_w, cv_bn_w,
        ]
        bias_list = [
            conv_bias, conv1_bias, conv_crossh_bias, conv_crossv_bias,
            c_bn_b, c1_bn_b, ch_bn_b, cv_bn_b,
        ]

        if self.dct_branch is not None:
            weight_list.append(self.dct_branch.get_equivalent_3x3_weight())
            bias_list.append(self.dct_branch.get_equivalent_bias())

        weight_all = torch.cat(weight_list, dim=0)
        bias_all = torch.cat(bias_list, dim=0)

        weight_compress = self.conv_out.weight.view(self.conv_out.out_channels, -1)
        W_flat = weight_all.view(weight_all.size(0), -1)
        weight = torch.matmul(weight_compress, W_flat).view(self.conv_out.out_channels, self.in_channels, 3, 3)
        bias = torch.matmul(weight_compress, bias_all)
        if self.conv_out.bias is not None:
            bias = bias + self.conv_out.bias
        return weight, bias


class MBRConv1(nn.Module):
    """1x1 Multi-Branch Re-parameterizable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, rep_scale: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale
        mid = out_channels * rep_scale

        self.conv = nn.Conv2d(in_channels, mid, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(mid))
        self.conv_out = nn.Conv2d(mid * 2, out_channels, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x0 = self.conv(inp)
        x = torch.cat([x0, self.conv_bn(x0)], dim=1)
        return self.conv_out(x)

    def slim(self) -> tuple[torch.Tensor, torch.Tensor]:
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        bn = self.conv_bn[0]
        k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        c_bn_w = conv_weight * k.view(-1, 1, 1, 1)
        c_bn_b = (conv_bias - bn.running_mean) * k + bn.bias

        weight_all = torch.cat([conv_weight, c_bn_w], dim=0)
        bias_all = torch.cat([conv_bias, c_bn_b], dim=0)

        weight_compress = self.conv_out.weight.view(self.conv_out.out_channels, -1)
        W_flat = weight_all.view(weight_all.size(0), -1)
        weight = torch.matmul(weight_compress, W_flat).view(self.conv_out.out_channels, self.in_channels, 1, 1)
        bias = torch.matmul(weight_compress, bias_all)
        if self.conv_out.bias is not None:
            bias = bias + self.conv_out.bias
        return weight, bias


# ===========================================================================
# 3. Feature Self-Transform (FST)
# ===========================================================================

class FeatureSelfTransform(nn.Module):
    """Feature Self-Transform (FST) with learnable weights and channel bias."""

    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block1 = block
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        return (self.weight1 * x1) * (self.weight2 * x1) + self.bias


class FeatureSelfTransformSlim(nn.Module):
    """Inference counterpart of Feature Self-Transform with standard Conv2d."""

    def __init__(self, block: nn.Module, channels: int):
        super().__init__()
        self.block1 = block
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        return (self.weight1 * x1) * (self.weight2 * x1) + self.bias


# ===========================================================================
# 4. Downsampling and Frequency-Guided Attention
# ===========================================================================

def downsample_to_target_avgpool(F_map: torch.Tensor, target: int = 32) -> torch.Tensor:
    """Deterministic downsampling to (target, target) for FFT magnitude computation."""
    _, _, H, W = F_map.shape
    kh = H // target
    kw = W // target
    if kh == 0 or kw == 0:
        return F_map
    if (H % target) != 0 or (W % target) != 0:
        H2 = max(kh * target, target)
        W2 = max(kw * target, target)
        F_map = F.interpolate(F_map, size=(H2, W2), mode="bilinear", align_corners=False)
        kh = H2 // target
        kw = W2 // target
    return F.avg_pool2d(F_map, kernel_size=(kh, kw), stride=(kh, kw))


# ===========================================================================
# 5. Full Architecture (Training: FGDPANet, Inference: FGDPASlimNet)
# ===========================================================================

class FGDPANet(nn.Module):
    """
    FGDPA Training Architecture with MBRConv-DCT and Frequency-Guided Dual-Path Attention.

    Args:
        in_channels (int): Input image channels (3 for RGB, 4/5 for physics priors).
        out_channels (int): Output channels (default 3).
        channels (int): Internal feature channels (default 12 as in paper).
        rep_scale (int): Expansion factor for MBRConv branches (default 4).
        fft_size (int): Low-resolution FFT target dimension (default 32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 12,
        rep_scale: int = 4,
        fft_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.rep_scale = rep_scale
        self.fft_size = fft_size

        self.head = FeatureSelfTransform(
            nn.Sequential(
                MBRConv5(in_channels, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale),
            ),
            channels,
        )

        self.body = FeatureSelfTransform(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels,
        )

        # FGDPA dual-path attention layers
        self.fgdra_fca = MBRConv1(2 * channels, channels, rep_scale=rep_scale)
        self.fgdra_fgsa = MBRConv1(2, channels, rep_scale=rep_scale)

        # Fusion gate parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.lam = nn.Parameter(torch.tensor(0.5))

        self.tail = MBRConv3(channels, out_channels, rep_scale=rep_scale)

    def _fgdpa_attention(self, F_map: torch.Tensor) -> torch.Tensor:
        F_ds = downsample_to_target_avgpool(F_map, target=self.fft_size)
        freq = torch.fft.fft2(F_ds)
        M = torch.log1p(torch.abs(freq))

        max_map, _ = torch.max(F_map, dim=1, keepdim=True)
        avg_map = torch.mean(F_map, dim=1, keepdim=True)
        spatial_in = torch.cat([max_map, avg_map], dim=1)
        A_s = torch.sigmoid(self.fgdra_fgsa(spatial_in))

        gap_F = torch.mean(F_map, dim=(2, 3), keepdim=True)
        gap_M = torch.mean(M, dim=(2, 3), keepdim=True)
        gap_M = gap_M / (gap_M.mean(dim=1, keepdim=True) + 1e-6)
        channel_in = torch.cat([gap_F, gap_M], dim=1)
        A_c = torch.sigmoid(self.fgdra_fca(channel_in))

        lam = torch.clamp(self.lam, 0.0, 1.0)
        A_lin = self.alpha * A_c + self.beta * A_s
        A_int = A_c * A_s
        return (1.0 - lam) * A_lin + lam * A_int

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        x0 = self.head(inputs)
        F_map = self.body(x0)
        A = self._fgdpa_attention(F_map)
        F_hat = A * F_map
        out = self.tail(F_hat)
        return torch.clamp(out, 0.0, 1.0)

    def slim(self) -> FGDPASlimNet:
        """Structural re-parameterization: compress all multi-branch convolutions into single Conv2d layers."""
        net_slim = FGDPASlimNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            fft_size=self.fft_size,
        )
        weight_slim = net_slim.state_dict()

        for name, mod in self.named_modules():
            if isinstance(mod, (MBRConv1, MBRConv3, MBRConv5)):
                w, b = mod.slim()
                if f"{name}.weight" in weight_slim:
                    weight_slim[f"{name}.weight"] = w
                    weight_slim[f"{name}.bias"] = b
            elif isinstance(mod, FeatureSelfTransform):
                if f"{name}.bias" in weight_slim:
                    weight_slim[f"{name}.bias"] = mod.bias
                if f"{name}.weight1" in weight_slim:
                    weight_slim[f"{name}.weight1"] = mod.weight1
                if f"{name}.weight2" in weight_slim:
                    weight_slim[f"{name}.weight2"] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                if f"{name}.weight" in weight_slim:
                    weight_slim[f"{name}.weight"] = mod.weight

        if "alpha" in weight_slim:
            weight_slim["alpha"] = self.alpha.detach()
        if "beta" in weight_slim:
            weight_slim["beta"] = self.beta.detach()
        if "lam" in weight_slim:
            weight_slim["lam"] = self.lam.detach()

        net_slim.load_state_dict(weight_slim)
        return net_slim


class FGDPASlimNet(nn.Module):
    """
    FGDPA Re-parameterized Inference Architecture.
    Exactly 4,234 parameters for in_channels=3, channels=12.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 12,
        fft_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.fft_size = fft_size

        self.head = FeatureSelfTransformSlim(
            nn.Sequential(
                nn.Conv2d(in_channels, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1),
            ),
            channels,
        )

        self.body = FeatureSelfTransformSlim(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels,
        )

        self.fgdra_fca = nn.Conv2d(2 * channels, channels, 1, 1)
        self.fgdra_fgsa = nn.Conv2d(2, channels, 1, 1)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.lam = nn.Parameter(torch.tensor(0.5))

        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def _fgdpa_attention(self, F_map: torch.Tensor) -> torch.Tensor:
        F_ds = downsample_to_target_avgpool(F_map, target=self.fft_size)
        freq = torch.fft.fft2(F_ds)
        M = torch.log1p(torch.abs(freq))

        max_map, _ = torch.max(F_map, dim=1, keepdim=True)
        avg_map = torch.mean(F_map, dim=1, keepdim=True)
        spatial_in = torch.cat([max_map, avg_map], dim=1)
        A_s = torch.sigmoid(self.fgdra_fgsa(spatial_in))

        gap_F = torch.mean(F_map, dim=(2, 3), keepdim=True)
        gap_M = torch.mean(M, dim=(2, 3), keepdim=True)
        gap_M = gap_M / (gap_M.mean(dim=1, keepdim=True) + 1e-6)
        channel_in = torch.cat([gap_F, gap_M], dim=1)
        A_c = torch.sigmoid(self.fgdra_fca(channel_in))

        lam = torch.clamp(self.lam, 0.0, 1.0)
        A_lin = self.alpha * A_c + self.beta * A_s
        A_int = A_c * A_s
        return (1.0 - lam) * A_lin + lam * A_int

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        x0 = self.head(inputs)
        F_map = self.body(x0)
        A = self._fgdpa_attention(F_map)
        F_hat = A * F_map
        out = self.tail(F_hat)
        return torch.clamp(out, 0.0, 1.0)


# ===========================================================================
# 6. Builder Functions
# ===========================================================================

def build_fgdpa(
    in_channels: int = 3,
    channels: int = 12,
    rep_scale: int = 4,
) -> FGDPANet:
    """Build the multi-branch training FGDPA network."""
    return FGDPANet(
        in_channels=in_channels,
        out_channels=3,
        channels=channels,
        rep_scale=rep_scale,
    )


def build_fgdpaslim(
    in_channels: int = 3,
    channels: int = 12,
    pretrained: bool = False,
) -> FGDPASlimNet:
    """
    Build the re-parameterized inference FGDPA network (4.234K params for 3ch).
    If pretrained=True and in_channels=3, automatically loads weights from checkpoints/fgdpa/.
    """
    model = FGDPASlimNet(
        in_channels=in_channels,
        out_channels=3,
        channels=channels,
    )
    if pretrained and in_channels == 3:
        ckpt_path = Path("checkpoints/fgdpa/model_best_slim.pkl")
        if not ckpt_path.exists():
            # Try absolute path based on repository location
            ckpt_path = Path(__file__).resolve().parents[3] / "checkpoints" / "fgdpa" / "model_best_slim.pkl"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "DCTLinearBranch",
    "MBRConv1",
    "MBRConv3",
    "MBRConv5",
    "FeatureSelfTransform",
    "FeatureSelfTransformSlim",
    "FGDPANet",
    "FGDPASlimNet",
    "build_fgdpa",
    "build_fgdpaslim",
]
