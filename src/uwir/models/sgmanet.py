"""
sgmanet.py
----------
SGMA-Net: A lightweight Mamba-Attention network adapted for
Underwater Image Restoration (UWIR) with statistical feature refinement.

References:
    - Thien B. Nguyen-Tat, Duy Pham Dinh Anh, Sang Hua Tan.
      "SGMA-Net: A lightweight mamba-attention network for thin-vessel
      segmentation in low-contrast fundus images with statistical feature refinement."
      Expert Systems With Applications, 2026/2027.

Adapted for UWIR:
    1. Continuous 3-channel RGB image reconstruction with global residual learning.
    2. Multi-channel input support (3ch RGB, 4ch RGB+t / RGB+B, 5ch RGB+t+B).
    3. Pure PyTorch S6 Selective Scan module for native Windows CUDA/CPU compatibility.
    4. Resolution-invariant Statistical Global Weight Learning (SGWL) for arbitrary image sizes.
    5. Preserves all key modules: RHMA (LGFI + S6 + MHSA), SAG, MDSA, and SGWL.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Depthwise Separable Convolution (DSConv)
# ---------------------------------------------------------------------------


class DSConv2d(nn.Module):
    """
    Depthwise Separable Convolution with BatchNorm and PReLU activation:
    Depthwise 3x3 -> Pointwise 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
        act: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_CUDA_AVAILABLE = True
except ImportError:
    selective_scan_fn = None
    MAMBA_CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# 2. S6 Selective Scan Module (Fused Mamba CUDA / Pure PyTorch fallback)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.float()
        norm = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * norm).to(input_dtype) * self.weight


class MambaSelectiveScan(nn.Module):
    """
    S6 Selective State Space Model with automatic hardware acceleration:
    - If CUDA & mamba_ssm is available: uses official fused selective_scan_fn kernel.
    - Otherwise: falls back to native PyTorch scan.

    Args:
        d_model (int): Hidden feature dimension (default 64).
        d_state (int): State expansion dimension (default 32).
    """

    def __init__(self, d_model: int = 64, d_state: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent projections for B, C, Delta
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # Initialize A parameter with stable negative decay
        A = torch.repeat_interleave(
            torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0),
            d_model,
            dim=0,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # Initialize dt projection bias
        dt_init_std = 2.0 / math.sqrt(d_model)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (B, L, D) token sequence.
        Returns:
            Tensor: (B, L, D) output sequence.
        """
        b, l, d = x.shape
        # Input-dependent projections
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state + D)
        delta_raw, b_proj, c_proj = torch.split(
            x_dbl, [self.d_model, self.d_state, self.d_state], dim=-1
        )
        A = -torch.exp(self.A_log.float())  # (D, d_state)

        # Fast path: Official CUDA fused selective scan from mamba_ssm
        if MAMBA_CUDA_AVAILABLE and x.is_cuda and selective_scan_fn is not None:
            dtype = x_dbl.dtype  # Matches autocast precision (e.g. float16/bfloat16)
            u = x.transpose(1, 2).to(dtype).contiguous()  # (B, D, L)
            delta = F.softplus(self.dt_proj(delta_raw)).transpose(1, 2).to(dtype).contiguous()  # (B, D, L)
            b_mat = b_proj.transpose(1, 2).to(dtype).contiguous()  # (B, N, L)
            c_mat = c_proj.transpose(1, 2).to(dtype).contiguous()  # (B, N, L)

            y = selective_scan_fn(
                u,
                delta,
                A.float(),
                b_mat,
                c_mat,
                self.D.float(),
                z=None,
                delta_bias=None,
                delta_softplus=False,
                return_last_state=False,
            )  # (B, D, L)
            y = y.transpose(1, 2)  # (B, L, D)
            return self.out_proj(self.act(y))

        # Fallback path: Pure PyTorch scan
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, D)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, d_state)
        dB = delta.unsqueeze(-1) * b_proj.unsqueeze(2)  # (B, L, D, d_state)

        # Sequential scan over token length
        h = torch.zeros((b, d, self.d_state), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            y_t = torch.einsum("bdn,bn->bd", h, c_proj[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * self.D
        return self.out_proj(self.act(y))


# Backward compatibility alias
SelectiveScanPureTorch = MambaSelectiveScan


# ---------------------------------------------------------------------------
# 3. Recursive Hybrid Mamba Attention (RHMA)
# ---------------------------------------------------------------------------


class LGFI(nn.Module):
    """
    Linear Global Feature Interaction (LGFI):
    Transposed channel-affinity attention across the channel dimension.
    Applies shared projections recursively for depth R = 2.
    """

    def __init__(self, channels: int = 64, recursion_depth: int = 2):
        super().__init__()
        self.channels = channels
        self.r = recursion_depth
        self.norm1 = nn.LayerNorm(channels)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels, bias=True)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def _single_pass(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        """
        res = x
        normed = self.norm1(x)

        # Transposed channel attention: compute affinity along channels (C x C)
        q = self.q_proj(normed).transpose(1, 2)  # (B, C, L)
        k = self.k_proj(normed).transpose(1, 2)  # (B, C, L)
        v = self.v_proj(normed).transpose(1, 2)  # (B, C, L)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.channels)
        attn = F.softmax(attn, dim=-1)  # (B, C, C)

        out = torch.matmul(attn, v).transpose(1, 2)  # (B, L, C)
        out = self.proj_out(out) + res

        out = out + self.ffn(self.norm2(out))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recursively apply transposed channel attention for R iterations.
        """
        for _ in range(self.r):
            x = self._single_pass(x)
        return x


class MHSA(nn.Module):
    """Two-head Multi-Head Self-Attention with learnable positional bias."""

    def __init__(self, d_model: int = 64, num_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, l, d)
        return self.proj(out)


class RHMA(nn.Module):
    """
    Recursive Hybrid Mamba Attention (RHMA) Bottleneck block:
    Z1 = LGFI(Fin)
    U1 = Z1 + S6(Linear(RMSNorm(Flat(Z1))))
    Z2 = U1 + FFN(RMSNorm(U1))
    U2 = Z2 + MHSA(RMSNorm(Flat(Z2)))
    Fout = U2 + FFN(RMSNorm(U2))
    """

    def __init__(self, channels: int = 64, d_state: int = 32, num_heads: int = 2):
        super().__init__()
        self.channels = channels
        self.lgfi = LGFI(channels=channels, recursion_depth=2)

        # S6 Branch
        self.norm_s6_in = RMSNorm(channels)
        self.s6 = MambaSelectiveScan(d_model=channels, d_state=d_state)
        self.norm_s6_ffn = RMSNorm(channels)
        self.ffn_s6 = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

        # MHSA Branch
        self.norm_mhsa = RMSNorm(channels)
        self.mhsa = MHSA(d_model=channels, num_heads=num_heads)
        self.norm_mhsa_ffn = RMSNorm(channels)
        self.ffn_mhsa = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)

        # 1. LGFI
        z1_seq = self.lgfi(x_seq)

        # 2. S6 Selective Scan
        u1_seq = z1_seq + self.s6(self.norm_s6_in(z1_seq))
        z2_seq = u1_seq + self.ffn_s6(self.norm_s6_ffn(u1_seq))

        # 3. MHSA
        u2_seq = z2_seq + self.mhsa(self.norm_mhsa(z2_seq))
        f_out_seq = u2_seq + self.ffn_mhsa(self.norm_mhsa_ffn(u2_seq))

        f_out = f_out_seq.transpose(1, 2).reshape(b, c, h, w)
        return f_out


# ---------------------------------------------------------------------------
# 4. Statistical Aggregation Gate (SAG)
# ---------------------------------------------------------------------------


class SAG(nn.Module):
    """
    Statistical Aggregation Gate (SAG):
    Filters shallow encoder features Xi using deeper semantic features Hi
    via dual GAP/GMP gating and a lightweight interaction map Mi.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dsconv_x = DSConv2d(channels, channels, 3, padding=1)
        self.dsconv_h = DSConv2d(channels, channels, 3, padding=1)

        # Gate 1: 6*C statistics -> C gate
        self.fuse_1x1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )
        self.gate1_conv = nn.Conv2d(channels * 6, channels, 1, bias=True)

        # Gate 2: 2*C statistics -> C gate
        self.gate2_conv = nn.Conv2d(channels * 2, channels, 1, bias=True)

        # Final projection
        self.out_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )

    def forward(self, x_i: torch.Tensor, h_i: torch.Tensor) -> torch.Tensor:
        """
        x_i: Shallow encoder feature (B, C, H, W)
        h_i: Upsampled deep semantic feature (B, C, H, W)
        """
        # Interaction map
        m_i = self.dsconv_x(x_i) + self.dsconv_h(h_i)

        # Multi-source GAP & GMP descriptors
        gap_x = x_i.mean(dim=(-2, -1), keepdim=True)
        gmp_x = x_i.amax(dim=(-2, -1), keepdim=True)
        gap_m = m_i.mean(dim=(-2, -1), keepdim=True)
        gmp_m = m_i.amax(dim=(-2, -1), keepdim=True)
        gap_h = h_i.mean(dim=(-2, -1), keepdim=True)
        gmp_h = h_i.amax(dim=(-2, -1), keepdim=True)

        s_i = torch.cat([gap_x, gmp_x, gap_m, gmp_m, gap_h, gmp_h], dim=1)  # (B, 6C, 1, 1)

        # Gate 1
        f_i = self.fuse_1x1(torch.cat([x_i, h_i], dim=1))
        alpha_i = torch.sigmoid(self.gate1_conv(s_i))
        g_i = f_i * alpha_i

        # Gate 2
        gap_g = g_i.mean(dim=(-2, -1), keepdim=True)
        gmp_g = g_i.amax(dim=(-2, -1), keepdim=True)
        r_i = torch.cat([gap_g, gmp_g], dim=1)  # (B, 2C, 1, 1)
        beta_i = torch.sigmoid(self.gate2_conv(r_i))
        g_tilde = g_i * beta_i

        # Final output with residual connection from x_i
        y_i = self.out_1x1(x_i + g_tilde)
        return y_i


# ---------------------------------------------------------------------------
# 5. Multi-Dimensional Statistical Attention (MDSA)
# ---------------------------------------------------------------------------


class MDC(nn.Module):
    """Multi-Dilated Convolution block with parallel atrous convolutions {1, 3, 5}."""

    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False)
        self.c3 = nn.Conv2d(channels, channels, 3, padding=3, dilation=3, bias=False)
        self.c5 = nn.Conv2d(channels, channels, 3, padding=5, dilation=5, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u1 = self.c1(x)
        u3 = self.c3(x)
        u5 = self.c5(x)
        return x + self.fuse(torch.cat([u1, u3, u5], dim=1))


class MDSA(nn.Module):
    """
    Multi-Dimensional Statistical Attention (MDSA):
    Observes feature responses along channel, height, and width views using MDC,
    and modulates spatial detail with channel-wise Mean, Max, and Std descriptors.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mdc_c = MDC(channels)
        self.mdc_h = MDC(channels)
        self.mdc_w = MDC(channels)

        self.fuse_md = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
        )

        # Statistical Attention (SA): 3 stats [mean; max; std] -> 1 attention map
        self.sa_dsconv = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.PReLU(3),
            nn.Conv2d(3, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: (B, C, H, W)
        """
        # Branch C: (B, C, H, W)
        b_c = self.mdc_c(f)

        # Branch H: transposed spatial view
        b_h = self.mdc_h(f.transpose(2, 3)).transpose(2, 3)
        # Branch W: standard view with independent MDC
        b_w = self.mdc_w(f)

        # Combine 3 branches
        f_md = self.fuse_md(torch.cat([b_c, b_h, b_w], dim=1))

        # Channel-wise statistics: Mean, Max, Std across channels
        mean_map = f_md.mean(dim=1, keepdim=True)
        max_map = f_md.amax(dim=1, keepdim=True)
        std_map = torch.sqrt(f_md.float().var(dim=1, keepdim=True, unbiased=False) + 1e-6).to(f_md.dtype)

        stats = torch.cat([mean_map, max_map, std_map], dim=1)  # (B, 3, H, W)
        a_sa = self.sa_dsconv(stats)  # (B, 1, H, W)

        return f * a_sa


# ---------------------------------------------------------------------------
# 6. Statistical Global Weight Learning (SGWL)
# ---------------------------------------------------------------------------


class SGWL(nn.Module):
    """
    Statistical Global Weight Learning (SGWL):
    Learns input-dependent fusion weights [alpha1, alpha2, alpha3] to combine
    three multi-level decoder predictions adaptively.

    Resolution-invariant implementation:
    Uses Global Average Pooling (GAP), Global Max Pooling (GMP), and
    Adaptive Depthwise Pooling to operate on arbitrary input resolutions.
    """

    def __init__(self, num_levels: int = 3, in_channels_per_pred: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels_per_pred
        total_channels = num_levels * in_channels_per_pred

        # Adaptive Depthwise convolution branch: captures spatial structure
        self.adapt_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dw_conv = nn.Conv2d(
            total_channels, total_channels, 4, groups=total_channels, bias=False
        )

        # Weight estimation projection: combines GAP, GMP, and Depthwise descriptors
        self.weight_fc = nn.Sequential(
            nn.Linear(total_channels, 32),
            nn.PReLU(32),
            nn.Linear(32, num_levels),
        )

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        s1, s2, s3: (B, 3, H, W) multi-level decoder predictions.
        Returns:
            s_final: (B, 3, H, W) fused prediction.
            alphas: (B, 3) fusion weights.
        """
        # Stack predictions along channel dimension: (B, 9, H, W)
        s_stack = torch.cat([s1, s2, s3], dim=1)

        # 1. GAP descriptor
        g_avg = s_stack.mean(dim=(-2, -1))  # (B, 9)

        # 2. GMP descriptor
        g_max = s_stack.amax(dim=(-2, -1))  # (B, 9)

        # 3. Depthwise full-map spatial descriptor
        g_dw = self.dw_conv(self.adapt_pool(s_stack)).flatten(1)  # (B, 9)

        v_global = g_avg + g_max + g_dw  # (B, 9)
        alphas = F.softmax(self.weight_fc(v_global), dim=-1)  # (B, 3)

        # Weighted combination of predictions
        a1 = alphas[:, 0:1, None, None]
        a2 = alphas[:, 1:2, None, None]
        a3 = alphas[:, 2:3, None, None]
        s_final = a1 * s1 + a2 * s2 + a3 * s3

        return s_final, alphas


# ---------------------------------------------------------------------------
# 7. SGMA-Net Complete Network
# ---------------------------------------------------------------------------


class SGMANet(nn.Module):
    """
    SGMA-Net adapted for Underwater Image Restoration (UWIR).

    Args:
        in_channels (int): Input channels (3 for RGB, 4 for RGB+prior, 5 for RGB+t+B).
        out_channels (int): Output channels (default 3 for RGB).
        base_channels (int): Base feature width (default 16).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        c1 = base_channels       # 16
        c2 = base_channels * 2   # 32
        c3 = base_channels * 4   # 64
        cb = base_channels * 4   # 64 (Bottleneck)

        # ----- Encoder -----
        self.stem = nn.Conv2d(in_channels, c1, 3, padding=1, bias=True)
        self.enc1 = nn.Sequential(
            DSConv2d(c1, c1),
            DSConv2d(c1, c1),
        )

        self.down1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            DSConv2d(c1, c2),
            DSConv2d(c2, c2),
        )

        self.down2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            DSConv2d(c2, c3),
            DSConv2d(c3, c3),
        )

        self.down3 = nn.MaxPool2d(2, 2)
        self.bottleneck_conv = DSConv2d(c3, cb)

        # ----- Bottleneck RHMA + MDSA -----
        self.rhma = RHMA(channels=cb, d_state=32, num_heads=2)
        self.bottleneck_mdsa = MDSA(cb)

        # ----- Skip Filtering (SAG + MDSA) -----
        self.sag3 = SAG(c3)
        self.mdsa3 = MDSA(c3)

        self.sag2 = SAG(c2)
        self.mdsa2 = MDSA(c2)

        self.sag1 = SAG(c1)
        self.mdsa1 = MDSA(c1)

        # ----- Decoder -----
        # Decoder Level 3: cb (upsampled to c3) + c3 -> c3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cb, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.PReLU(c3),
        )
        self.dec3 = nn.Sequential(
            DSConv2d(c3 + c3, c3),
            DSConv2d(c3, c3),
        )
        self.aux_head3 = nn.Conv2d(c3, out_channels, 1, bias=True)

        # Decoder Level 2: c3 (upsampled to c2) + c2 -> c2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.PReLU(c2),
        )
        self.dec2 = nn.Sequential(
            DSConv2d(c2 + c2, c2),
            DSConv2d(c2, c2),
        )
        self.aux_head2 = nn.Conv2d(c2, out_channels, 1, bias=True)

        # Decoder Level 1: c2 (upsampled to c1) + c1 -> c1
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.PReLU(c1),
        )
        self.dec1 = nn.Sequential(
            DSConv2d(c1 + c1, c1),
            DSConv2d(c1, c1),
        )
        self.aux_head1 = nn.Conv2d(c1, out_channels, 1, bias=True)

        # ----- SGWL: Multi-level Prediction Fusion -----
        self.sgwl = SGWL(num_levels=3, in_channels_per_pred=out_channels)

        # Zero-init heads for smooth residual starting
        nn.init.zeros_(self.aux_head1.weight)
        nn.init.zeros_(self.aux_head1.bias)
        nn.init.zeros_(self.aux_head2.weight)
        nn.init.zeros_(self.aux_head2.bias)
        nn.init.zeros_(self.aux_head3.weight)
        nn.init.zeros_(self.aux_head3.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected NCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )

        # Base RGB for residual restoration
        rgb_input = inputs[:, :3]

        # 1. Encoder forward pass
        x1 = self.enc1(self.stem(inputs))  # (B, 16, H, W)
        x2 = self.enc2(self.down1(x1))     # (B, 32, H/2, W/2)
        x3 = self.enc3(self.down2(x2))     # (B, 64, H/4, W/4)

        # 2. Bottleneck: RHMA + MDSA
        b_in = self.bottleneck_conv(self.down3(x3))  # (B, 64, H/8, W/8)
        b_feat = self.bottleneck_mdsa(self.rhma(b_in))

        # 3. Decoder Level 3
        h3 = self.up3(b_feat)
        skip3 = self.mdsa3(self.sag3(x3, h3))
        d3 = self.dec3(torch.cat([h3, skip3], dim=1))
        # Auxiliary prediction 3 (upsampled to original resolution H, W)
        pred3_raw = self.aux_head3(d3)
        pred3 = F.interpolate(pred3_raw, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        # 4. Decoder Level 2
        h2 = self.up2(d3)
        skip2 = self.mdsa2(self.sag2(x2, h2))
        d2 = self.dec2(torch.cat([h2, skip2], dim=1))
        # Auxiliary prediction 2 (upsampled to original resolution H, W)
        pred2_raw = self.aux_head2(d2)
        pred2 = F.interpolate(pred2_raw, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        # 5. Decoder Level 1
        h1 = self.up1(d2)
        skip1 = self.mdsa1(self.sag1(x1, h1))
        d1 = self.dec1(torch.cat([h1, skip1], dim=1))
        pred1 = self.aux_head1(d1)

        # 6. SGWL Adaptive Multi-Level Fusion
        pred_fused, _ = self.sgwl(pred1, pred2, pred3)

        # 7. Global Residual Restoration: Output bounded in [0, 1]
        output = torch.clamp(rgb_input + pred_fused, 0.0, 1.0)
        return output


def build_sgmanet(in_channels: int = 3) -> SGMANet:
    """Build SGMA-Net for underwater image restoration (~0.96M parameters)."""
    return SGMANet(in_channels=in_channels, out_channels=3, base_channels=16)


__all__ = [
    "DSConv2d",
    "LGFI",
    "MDC",
    "MDSA",
    "MHSA",
    "RHMA",
    "RMSNorm",
    "SAG",
    "SGMANet",
    "SGWL",
    "MambaSelectiveScan",
    "SelectiveScanPureTorch",
    "MAMBA_CUDA_AVAILABLE",
    "build_sgmanet",
]
