"""
losses.py
---------
Loss functions for the physics-guided 5-channel U-Net
(underwater image restoration, EUVP dataset).

Classes
-------
VGGPerceptualLoss  – VGG-16 feature-space L1 at relu1_2 + relu2_2
SSIMLoss           – 1 − SSIM (via kornia)
CompositeLoss      – λ_l1·L1 + λ_perc·Perceptual + λ_ssim·SSIM
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.jit.script is deprecated.*")

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

# ImageNet statistics for VGG normalisation
_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
_VGG_STD = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# VGG Perceptual Loss
# ---------------------------------------------------------------------------


class VGGPerceptualLoss(nn.Module):
    """
    VGG-16 feature-space loss using relu1_2 and relu2_2 activations.

    Args:
        device (str | torch.device): Target device for the frozen VGG backbone.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # relu1_2 → first 4 children; relu2_2 → children 4-9
        self.stage1 = nn.Sequential(*list(vgg.children())[:4]).to(device).eval()
        self.stage2 = nn.Sequential(*list(vgg.children())[4:9]).to(device).eval()
        for p in self.parameters():
            p.requires_grad = False

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean/std normalisation expected by VGG."""
        mean = _VGG_MEAN.to(x.device, x.dtype).view(1, 3, 1, 1)
        std = _VGG_STD.to(x.device, x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, 3, H, W) predicted RGB in [0, 1].
            target (Tensor): (N, 3, H, W) ground-truth RGB in [0, 1].

        Returns:
            Tensor: scalar perceptual loss.
        """
        # VGG expects ImageNet-normalised inputs; raw [0,1] causes large
        # intermediate activations that can overflow gradients to NaN.
        pred = self._normalise(pred.clamp(0, 1))
        target = self._normalise(target.clamp(0, 1))

        p1 = self.stage1(pred)
        t1 = self.stage1(target)
        p2 = self.stage2(p1)
        t2 = self.stage2(t1)
        return F.l1_loss(p1, t1) + F.l1_loss(p2, t2)


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------


class SSIMLoss(nn.Module):
    """
    SSIM-based loss: ``1 − mean(SSIM map)``.
    Uses ``kornia.metrics.ssim`` which returns the per-pixel SSIM map.

    Args:
        window_size (int): Gaussian window size. Default: 11.
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, C, H, W) in [0, 1].
            target (Tensor): (N, C, H, W) in [0, 1].

        Returns:
            Tensor: scalar loss ∈ [0, 2].
        """
        ssim_map = kornia.metrics.ssim(pred, target, self.window_size)
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Charbonnier Loss
# ---------------------------------------------------------------------------


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: sqrt((pred - target)^2 + eps^2).
    A smooth, continuously differentiable approximation to L1 loss near 0.

    Args:
        eps (float): Small constant to avoid zero gradients. Default: 1e-3.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


# ---------------------------------------------------------------------------
# Color Angle Loss (Cosine Distance on RGB Vectors)
# ---------------------------------------------------------------------------


class ColorAngleLoss(nn.Module):
    """
    Color angle (cosine distance) loss between RGB pixel vectors:
        loss = 1 - mean(cos(pred, target))
    Penalizes chromatic / color cast deviations invariant to illumination intensity.

    Args:
        eps (float): Epsilon for numerical stability in L2 norm. Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, 3, H, W) in [0, 1].
            target (Tensor): (N, 3, H, W) in [0, 1].
        """
        dot = (pred * target).sum(dim=1)
        norm_p = torch.norm(pred, p=2, dim=1).clamp(min=self.eps)
        norm_t = torch.norm(target, p=2, dim=1).clamp(min=self.eps)
        cos_sim = dot / (norm_p * norm_t)
        return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# Wavelet Frequency Domain Loss
# ---------------------------------------------------------------------------


class WaveletLoss(nn.Module):
    """
    2D Haar Wavelet domain loss for multi-frequency fidelity:
    Decomposes pred and target into LL (illumination/color) and LH, HL, HH (edge/texture).
    Applies L1 loss on LL and weighted L1 loss on high-frequency bands.

    Args:
        hf_weight (float): Relative weight for high-frequency sub-bands (LH, HL, HH). Default: 0.5.
    """

    def __init__(self, hf_weight: float = 0.5):
        super().__init__()
        self.hf_weight = hf_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h, w = pred.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            pred = F.pad(pred, (0, pad_w, 0, pad_h), mode="replicate")
            target = F.pad(target, (0, pad_w, 0, pad_h), mode="replicate")

        def _haar(x: torch.Tensor):
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
            return ll, lh, hl, hh

        ll_p, lh_p, hl_p, hh_p = _haar(pred)
        ll_t, lh_t, hl_t, hh_t = _haar(target)

        l_ll = F.l1_loss(ll_p, ll_t)
        l_hf = (F.l1_loss(lh_p, lh_t) + F.l1_loss(hl_p, hl_t) + F.l1_loss(hh_p, hh_t)) / 3.0
# ---------------------------------------------------------------------------
# Total Variation (TV) Loss
# ---------------------------------------------------------------------------


class TVLoss(nn.Module):
    """
    Total Variation (TV) Loss for spatial smoothness and noise suppression.

    Penalizes high-frequency variance between adjacent spatial pixels.
    Formula:
        L_tv = mean(|x[:, :, 1:, :] - x[:, :, :-1, :]|) + mean(|x[:, :, :, 1:] - x[:, :, :, :-1]|)
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        del target
        diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        tv = diff_h.mean() + diff_w.mean()
        return tv * self.weight


# ---------------------------------------------------------------------------
# Edge Loss (Gaussian-Laplacian Pyramid, Burt & Adelson 1983 / Root losses.py)
# ---------------------------------------------------------------------------


class EdgeLoss(nn.Module):
    """
    Edge preservation loss via Gaussian-Laplacian pyramid decomposition (Burt & Adelson 1983).
    Calculates distance between Laplacian edge feature maps:
        Lap(I) = I - Blur(Expand(Down(Blur(I))))

    Supports:
        - Arbitrary input channels (1, 3, etc.) with automatic kernel channel adaptation.
        - Distance formulations: "mse" (L2, default), "l1" (MAE), or "charbonnier" (smoothed L1).
    """

    def __init__(self, loss_weight: float = 1.0, loss_type: str = "mse"):
        super().__init__()
        k = torch.tensor([0.05, 0.25, 0.40, 0.25, 0.05], dtype=torch.float32)
        kernel = torch.matmul(k.unsqueeze(1), k.unsqueeze(0)).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.weight = loss_weight
        self.loss_type = loss_type.lower()

    def _conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        n_channels = img.shape[1]
        kernel = self.kernel
        if kernel.shape[0] != n_channels:
            kernel = kernel[:1].repeat(n_channels, 1, 1, 1)
        kw, kh = kernel.shape[2], kernel.shape[3]
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, kernel.to(img.device, img.dtype), groups=n_channels)

    def _laplacian(self, current: torch.Tensor) -> torch.Tensor:
        filtered = self._conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        # Burt & Adelson zero-insertion upsample scaled by 4
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4.0
        filtered = self._conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = self._laplacian(pred) - self._laplacian(target)
        if self.loss_type == "l1":
            loss = torch.mean(torch.abs(diff))
        elif self.loss_type == "charbonnier":
            loss = torch.mean(torch.sqrt(diff ** 2 + 1e-6))
        else:  # "mse"
            loss = torch.mean(diff ** 2)
        return loss * self.weight


# ---------------------------------------------------------------------------
# Gradient Difference Loss (GDL, Mathieu et al., ICLR 2016)
# ---------------------------------------------------------------------------


class GradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss (GDL / GD) for sharp edge and micro-gradient preservation.
    (Mathieu et al., "Deep multi-scale video prediction beyond mean square error", ICLR 2016).

    Penalizes the absolute difference between predicted and ground-truth spatial gradients
    along both vertical (height) and horizontal (width) axes:
        diff_h = |(pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :])|
        diff_w = |(pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])|
        L_gd = mean(diff_h) + mean(diff_w)
    """

    def __init__(self, loss_weight: float = 1.0, alpha: int = 1):
        super().__init__()
        self.weight = loss_weight
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs((pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :]))
        diff_w = torch.abs((pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1]))
        if self.alpha == 2:
            loss = torch.mean(diff_h ** 2) + torch.mean(diff_w ** 2)
        else:
            loss = torch.mean(diff_h) + torch.mean(diff_w)
        return loss * self.weight


# ---------------------------------------------------------------------------
# Local Variance-Weighted (LVW) / Outlier-Aware Loss (MobileIE, Yan et al., 2025)
# Official repository: https://github.com/AVC2-UESTC/MobileIE/blob/main/loss.py
# ---------------------------------------------------------------------------



class LocalVarianceLoss(nn.Module):
    """
    Local Variance-Weighted (LVW) / Outlier-Aware Loss from MobileIE (Yan et al., 2025).
    Verbatim implementation from official code: https://github.com/AVC2-UESTC/MobileIE/blob/main/loss.py

    Official Author Formulation:
        delta = out - lab
        var = delta.std((2, 3), keepdims=True) / (2 ** 0.5)
        avg = delta.mean((2, 3), True)
        weight = torch.tanh((delta - avg).abs() / (var + 1e-6)).detach()
        loss = (delta.abs() * weight).mean()

    Note on `.detach()`:
        Detaching the weight ensures gradients backpropagate strictly through `delta.abs()`,
        preventing adversarial oscillation / drift from the weighting function.

    Args:
        mode (str):
            - "spatial" (default, official MobileIE code): computes std / sqrt(2) and mean over (H, W).
            - "window": computes local statistics across a sliding KxK window.
        kernel_size (int): Window size if mode="window". Default: 7.
        loss_weight (float): Loss weight lambda_lvw. Default: 1.0 (or 0.1 in composite).
        eps (float): Small epsilon to prevent division by zero. Default: 1e-6.
    """

    def __init__(
        self,
        mode: str = "spatial",
        kernel_size: int = 7,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mode = mode
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.weight = loss_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = pred - target
        if self.mode == "spatial":
            # Official MobileIE repository (OutlierAwareLoss)
            var = delta.std(dim=(-2, -1), keepdim=True) / (2.0 ** 0.5)
            avg = delta.mean(dim=(-2, -1), keepdim=True)
            w = torch.tanh((delta - avg).abs() / (var + self.eps)).detach()
            loss = (delta.abs() * w).mean()
        else:
            # Sliding window over KxK neighborhood
            abs_d = delta.abs()
            mu = F.avg_pool2d(abs_d, kernel_size=self.k, stride=1, padding=self.pad)
            d_sq = F.avg_pool2d(abs_d ** 2, kernel_size=self.k, stride=1, padding=self.pad)
            sigma = torch.sqrt(torch.clamp(d_sq - mu ** 2, min=1e-8)) / (2.0 ** 0.5)
            w = torch.tanh(torch.abs(abs_d - mu) / (sigma + self.eps)).detach()
            loss = (abs_d * w).mean()

        return loss * self.weight


# Alias for compatibility with MobileIE official naming
OutlierAwareLoss = LocalVarianceLoss


# ---------------------------------------------------------------------------
# Differentiable UIQM Loss (Panetta et al., 2016 & Mamba-Frequency UWIR, 2025)
# ---------------------------------------------------------------------------


class UIQMLoss(nn.Module):
    """
    Differentiable Underwater Image Quality Measure (UIQM) Loss.

    Components:
        1. UICM  : Colorfulness via RG and YB opponent channels
        2. UISM  : Sharpness via Sobel gradients on luminance
        3. UIConM: Contrast via luminance standard deviation / mean
        UIQM = 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM
        L_uiqm = mean(1.0 / (clamp(UIQM, min=0.1) + eps))
    """

    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.weight = loss_weight
        self.eps = eps
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 4.0
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        del target
        pred_c = pred.clamp(0.0, 1.0)
        R = pred_c[:, 0:1, :, :]
        G = pred_c[:, 1:2, :, :]
        B = pred_c[:, 2:3, :, :]

        # 1. UICM (Colorfulness)
        RG = R - G
        YB = 0.5 * (R + G) - B
        mu_rg = RG.mean(dim=(-2, -1))
        mu_yb = YB.mean(dim=(-2, -1))
        std_rg = RG.std(dim=(-2, -1), unbiased=False)
        std_yb = YB.std(dim=(-2, -1), unbiased=False)
        uicm = -0.0268 * torch.sqrt(mu_rg ** 2 + mu_yb ** 2 + self.eps) + 0.1586 * torch.sqrt(std_rg ** 2 + std_yb ** 2 + self.eps)

        # 2. UISM (Sharpness on luminance)
        gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        sx = F.conv2d(gray_pad, self.sobel_x.to(pred.device, pred.dtype))
        sy = F.conv2d(gray_pad, self.sobel_y.to(pred.device, pred.dtype))
        uism = torch.sqrt(sx ** 2 + sy ** 2 + self.eps).mean(dim=(-2, -1))

        # 3. UIConM (Contrast on luminance)
        mu_g = gray.mean(dim=(-2, -1))
        std_g = gray.std(dim=(-2, -1), unbiased=False)
        uiconm = std_g / (mu_g + 1e-4)

        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
        loss = torch.mean(1.0 / (torch.clamp(uiqm, min=0.1) + self.eps))
        return loss * self.weight


# ---------------------------------------------------------------------------
# HVI (Hue-Value-Intensity) Color Space Loss (WWE-UIE, WACV 2026)
# ---------------------------------------------------------------------------


class HVILoss(nn.Module):
    """
    HVI Loss from WWE-UIE (Cheng et al., WACV 2026):
    "WWE-UIE: A Wavelet & White Balance Efficient Network for Underwater Image Enhancement".

    Transforms RGB images into HVI (Hue, Value, Intensity) space, where color
    sensitivity C = (sin(I * pi / 2) + eps)^k modulates hue and saturation into
    Cartesian coordinates (H, V) alongside intensity I.
    Computes L1 loss between predicted and target HVI representations:
        L_hvi(Y, Y') = || HVI(Y) - HVI(Y') ||_1

    Args:
        density_k (float): Color sensitivity modulation factor (WWE-UIE paper default: 0.2).
        loss_weight (float): Multiplier for the loss (default: 1.0).
    """

    def __init__(self, density_k: float = 0.2, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.register_buffer("density_k", torch.tensor(float(density_k)))

    def rgb_to_hvi(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB image tensor in [0, 1] of shape [B, 3, H, W] to HVI tensor of shape [B, 3, H, W].
        Matches exact formulation of WWE-UIE without in-place tensor mutations.
        """
        eps = 1e-8
        r = img[:, 0:1, :, :]
        g = img[:, 1:2, :, :]
        b = img[:, 2:3, :, :]
        value = torch.max(img, dim=1, keepdim=True)[0]
        img_min = torch.min(img, dim=1, keepdim=True)[0]
        diff = value - img_min + eps

        h_r = ((g - b) / diff) % 6.0
        h_g = 2.0 + (b - r) / diff
        h_b = 4.0 + (r - g) / diff

        hue = torch.where(b == value, h_b, torch.zeros_like(r))
        hue = torch.where(g == value, h_g, hue)
        hue = torch.where(r == value, h_r, hue)
        hue = torch.where(value == img_min, torch.zeros_like(hue), hue)
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation)

        pi = 3.141592653589793
        color_sensitive = (torch.sin(value * 0.5 * pi) + eps).pow(self.density_k)
        ch = torch.cos(2.0 * pi * hue)
        cv = torch.sin(2.0 * pi * hue)

        h_coord = color_sensitive * saturation * ch
        v_coord = color_sensitive * saturation * cv
        i_coord = value
        return torch.cat([h_coord, v_coord, i_coord], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp pred to [0, 1] as in author's official implementation
        pred_clamped = pred.clamp(0.0, 1.0)
        pred_hvi = self.rgb_to_hvi(pred_clamped)
        target_hvi = self.rgb_to_hvi(target)
        return self.loss_weight * F.l1_loss(pred_hvi, target_hvi)


# ---------------------------------------------------------------------------
# Laplacian Pyramid Loss (Adapt-PEFT, Malik & Martinel, ICPR 2026)
# Section 3.5, Eq. (9):
#     L_Lap = \sum_{j=0}^K 2^{2j} || L_j(I*) - L_j(\hat{I}) ||_1
# ---------------------------------------------------------------------------


class LaplacianPyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss from Adapt-PEFT (Malik & Martinel, ICPR 2026).
    Paper Section 3.5, Equation (9):
        L_Lap = \\sum_{j=0}^K 2^{2j} || L_j(I^*) - L_j(\\hat{I}) ||_1

    Decomposes both images into frequency bands using Laplacian operators and
    minimizes differences at each pyramid level with 2^(2j) exponential weighting
    to emphasize structural information across multiple spatial scales.

    Args:
        num_levels (int): Number of pyramid decomposition levels (default: 3).
        loss_weight (float): Multiplier for the total loss (default: 1.0).
    """

    def __init__(self, num_levels: int = 3, loss_weight: float = 1.0):
        super().__init__()
        self.num_levels = num_levels
        self.loss_weight = loss_weight
        # 5x5 Gaussian kernel (Burt & Adelson 1983)
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        kernel = torch.outer(k, k).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def _conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        n_channels = img.shape[1]
        kernel = self.kernel
        if kernel.shape[0] != n_channels:
            kernel = kernel[:1].repeat(n_channels, 1, 1, 1)
        img = F.pad(img, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(img, kernel.to(img.device, img.dtype), groups=n_channels)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, ::2, ::2]

    def _upsample(self, x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
        b, c, h, w = x.shape
        up = torch.zeros(b, c, h * 2, w * 2, device=x.device, dtype=x.dtype)
        up[:, :, ::2, ::2] = x * 4.0
        up = self._conv_gauss(up)
        if up.shape[2:] != target_shape:
            up = F.interpolate(up, size=target_shape, mode="bilinear", align_corners=False)
        return up

    def _build_pyramid(self, img: torch.Tensor) -> list[torch.Tensor]:
        current = img
        pyr = []
        for _ in range(self.num_levels):
            filtered = self._conv_gauss(current)
            down = self._downsample(filtered)
            up = self._upsample(down, current.shape[2:])
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)  # lowest-frequency residual base band
        return pyr

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pyr_pred = self._build_pyramid(pred)
        pyr_target = self._build_pyramid(target)
        loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        for j, (p, t) in enumerate(zip(pyr_pred, pyr_target)):
            w = float(2 ** (2 * j))
            loss = loss + w * F.l1_loss(p, t)
        return self.loss_weight * loss


# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """
    Weighted combination of pixel fidelity (L1 or Charbonnier), VGG perceptual,
    SSIM, Total Variation, Edge, Local Variance (MobileIE), UIQM, HVI (WWE-UIE),
    and Laplacian Pyramid (Adapt-PEFT) losses.

    Supports simple 0/1 toggles for easy ablation on Kaggle:
        use_l1, use_perc, use_ssim, use_tv, use_edge, use_lvw, use_uiqm, use_hvi, use_lap_pyr
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_perc: float = 1.0,
        lambda_ssim: float = 0.1,
        lambda_color: float = 0.0,
        lambda_wavelet: float = 0.0,
        lambda_tv: float = 0.001,
        lambda_edge: float = 0.1,
        lambda_gd: float = 1.0,
        lambda_lvw: float = 0.1,
        lambda_uiqm: float = 0.05,
        lambda_hvi: float = 0.5,
        lambda_lap_pyr: float = 1.0,
        lap_pyr_levels: int = 3,
        lvw_mode: str = "spatial",
        density_k: float = 0.2,
        use_l1: int = 1,
        use_perc: int = 1,
        use_ssim: int = 0,
        use_color: int = 0,
        use_wavelet: int = 0,
        use_tv: int = 0,
        use_edge: int = 0,
        use_gd: int = 0,
        use_lvw: int = 0,
        use_uiqm: int = 0,
        use_hvi: int = 0,
        use_lap_pyr: int = 0,
        use_charbonnier: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        # Calculate effective weights based on 0/1 toggles
        self.eff_l1 = float(lambda_l1) if int(use_l1) else 0.0
        self.eff_perc = float(lambda_perc) if int(use_perc) else 0.0
        self.eff_ssim = float(lambda_ssim) if int(use_ssim) else 0.0
        self.eff_color = float(lambda_color) if int(use_color) else 0.0
        self.eff_wavelet = float(lambda_wavelet) if int(use_wavelet) else 0.0
        self.eff_tv = float(lambda_tv) if int(use_tv) else 0.0
        self.eff_edge = float(lambda_edge) if int(use_edge) else 0.0
        self.eff_gd = float(lambda_gd) if int(use_gd) else 0.0

        self.eff_lvw = float(lambda_lvw) if int(use_lvw) else 0.0
        self.eff_uiqm = float(lambda_uiqm) if int(use_uiqm) else 0.0
        self.eff_hvi = float(lambda_hvi) if int(use_hvi) else 0.0
        self.eff_lap_pyr = float(lambda_lap_pyr) if int(use_lap_pyr) else 0.0

        self.use_charbonnier = use_charbonnier
        self.l1 = (CharbonnierLoss() if use_charbonnier else nn.L1Loss()) if self.eff_l1 else None
        self.perc = VGGPerceptualLoss(device) if self.eff_perc else None
        self.ssim = SSIMLoss() if self.eff_ssim else None
        self.color = ColorAngleLoss() if self.eff_color else None
        self.wavelet = WaveletLoss() if self.eff_wavelet else None
        self.tv = TVLoss(loss_weight=1.0) if self.eff_tv else None
        self.edge = EdgeLoss(loss_weight=1.0) if self.eff_edge else None
        self.gd = GradientDifferenceLoss(loss_weight=1.0) if self.eff_gd else None
        self.lvw = LocalVarianceLoss(mode=lvw_mode, kernel_size=7, loss_weight=1.0) if self.eff_lvw else None
        self.uiqm = UIQMLoss(loss_weight=1.0) if self.eff_uiqm else None
        self.hvi = HVILoss(density_k=density_k, loss_weight=1.0) if self.eff_hvi else None
        self.lap_pyr = LaplacianPyramidLoss(num_levels=lap_pyr_levels, loss_weight=1.0) if self.eff_lap_pyr else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
        parts = {}

        if self.l1 is not None and self.eff_l1:
            l_l1 = self.l1(pred, target)
            total = total + self.eff_l1 * l_l1
            parts["l1"] = l_l1.item()
        else:
            parts["l1"] = 0.0

        if self.perc is not None and self.eff_perc:
            l_perc = self.perc(pred, target)
            total = total + self.eff_perc * l_perc
            parts["perceptual"] = l_perc.item()
        else:
            parts["perceptual"] = 0.0

        if self.ssim is not None and self.eff_ssim:
            l_ssim = self.ssim(pred, target)
            total = total + self.eff_ssim * l_ssim
            parts["ssim_loss"] = l_ssim.item()
        else:
            parts["ssim_loss"] = 0.0

        if self.tv is not None and self.eff_tv:
            l_tv = self.tv(pred, target)
            total = total + self.eff_tv * l_tv
            parts["tv"] = l_tv.item()
        else:
            parts["tv"] = 0.0

        if self.edge is not None and self.eff_edge:
            l_edge = self.edge(pred, target)
            total = total + self.eff_edge * l_edge
            parts["edge"] = l_edge.item()
        else:
            parts["edge"] = 0.0

        if self.gd is not None and self.eff_gd:
            l_gd = self.gd(pred, target)
            total = total + self.eff_gd * l_gd
            parts["gd"] = l_gd.item()
        else:
            parts["gd"] = 0.0

        if self.lvw is not None and self.eff_lvw:
            l_lvw = self.lvw(pred, target)
            total = total + self.eff_lvw * l_lvw
            parts["lvw"] = l_lvw.item()
        else:
            parts["lvw"] = 0.0


        if self.uiqm is not None and self.eff_uiqm:
            l_uiqm = self.uiqm(pred, target)
            total = total + self.eff_uiqm * l_uiqm
            parts["uiqm"] = l_uiqm.item()
        else:
            parts["uiqm"] = 0.0

        if self.hvi is not None and self.eff_hvi:
            l_hvi = self.hvi(pred, target)
            total = total + self.eff_hvi * l_hvi
            parts["hvi"] = l_hvi.item()
        else:
            parts["hvi"] = 0.0

        if self.lap_pyr is not None and self.eff_lap_pyr:
            l_lap_pyr = self.lap_pyr(pred, target)
            total = total + self.eff_lap_pyr * l_lap_pyr
            parts["lap_pyr"] = l_lap_pyr.item()
        else:
            parts["lap_pyr"] = 0.0

        if self.color is not None and self.eff_color:
            l_color = self.color(pred, target)
            total = total + self.eff_color * l_color
            parts["color"] = l_color.item()
        else:
            parts["color"] = 0.0

        if self.wavelet is not None and self.eff_wavelet:
            l_wav = self.wavelet(pred, target)
            total = total + self.eff_wavelet * l_wav
            parts["wavelet"] = l_wav.item()
        else:
            parts["wavelet"] = 0.0

        parts["total"] = total.item()
        return total, parts


__all__ = [
    "CharbonnierLoss",
    "ColorAngleLoss",
    "CompositeLoss",
    "EdgeLoss",
    "GradientDifferenceLoss",
    "HVILoss",
    "LaplacianPyramidLoss",
    "LocalVarianceLoss",
    "OutlierAwareLoss",
    "SSIMLoss",
    "TVLoss",
    "UIQMLoss",
    "VGGPerceptualLoss",
    "WaveletLoss",
]

