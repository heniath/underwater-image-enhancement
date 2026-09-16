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
        return l_ll + self.hf_weight * l_hf


# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """
    Weighted combination of pixel fidelity (L1 or Charbonnier), VGG perceptual,
    SSIM, Color Angle, and Wavelet frequency losses:

        loss = λ_l1·L1_or_Charb + λ_perc·Perceptual + λ_ssim·SSIM + λ_color·Color + λ_wav·Wavelet

    Args:
        lambda_l1       (float): Weight for L1 or Charbonnier loss. Default: 1.0.
        lambda_perc     (float): Weight for perceptual loss.       Default: 0.1.
        lambda_ssim     (float): Weight for SSIM loss.             Default: 0.5.
        lambda_color    (float): Weight for Color Angle loss.      Default: 0.0.
        lambda_wavelet  (float): Weight for Wavelet domain loss.   Default: 0.0.
        use_charbonnier  (bool): If True, use CharbonnierLoss instead of L1. Default: False.
        device (str | torch.device): Device for VGG backbone.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_ssim: float = 0.5,
        lambda_color: float = 0.0,
        lambda_wavelet: float = 0.0,
        use_charbonnier: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_wavelet = lambda_wavelet
        self.use_charbonnier = use_charbonnier

        self.l1 = CharbonnierLoss() if use_charbonnier else nn.L1Loss()
        self.perc = VGGPerceptualLoss(device) if lambda_perc else None
        self.ssim = SSIMLoss() if lambda_ssim else None
        self.color = ColorAngleLoss() if lambda_color else None
        self.wavelet = WaveletLoss() if lambda_wavelet else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred   (Tensor): (N, 3, H, W) model output in [0, 1].
            target (Tensor): (N, 3, H, W) ground truth in [0, 1].

        Returns:
            total (Tensor): Scalar combined loss.
            parts (dict):   Per-component losses as Python floats
                            with keys ``"l1"``, ``"perceptual"``,
                            ``"ssim_loss"``, ``"color"``, ``"wavelet"``, ``"total"``.
        """
        l_l1 = self.l1(pred, target)
        total = self.lambda_l1 * l_l1
        parts = {"l1": l_l1.item()}

        if self.perc is not None and self.lambda_perc:
            l_perc = self.perc(pred, target)
            total = total + self.lambda_perc * l_perc
            parts["perceptual"] = l_perc.item()
        else:
            parts["perceptual"] = 0.0

        if self.ssim is not None and self.lambda_ssim:
            l_ssim = self.ssim(pred, target)
            total = total + self.lambda_ssim * l_ssim
            parts["ssim_loss"] = l_ssim.item()
        else:
            parts["ssim_loss"] = 0.0

        if self.color is not None and self.lambda_color:
            l_color = self.color(pred, target)
            total = total + self.lambda_color * l_color
            parts["color"] = l_color.item()
        else:
            parts["color"] = 0.0

        if self.wavelet is not None and self.lambda_wavelet:
            l_wav = self.wavelet(pred, target)
            total = total + self.lambda_wavelet * l_wav
            parts["wavelet"] = l_wav.item()
        else:
            parts["wavelet"] = 0.0

        parts["total"] = total.item()
        return total, parts


__all__ = [
    "CharbonnierLoss",
    "ColorAngleLoss",
    "CompositeLoss",
    "SSIMLoss",
    "VGGPerceptualLoss",
    "WaveletLoss",
]
