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
        import kornia

        self._kornia = kornia

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, C, H, W) in [0, 1].
            target (Tensor): (N, C, H, W) in [0, 1].

        Returns:
            Tensor: scalar loss ∈ [0, 2].
        """
        ssim_map = self._kornia.metrics.ssim(pred, target, self.window_size)
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# HSV-CS Loss (PCF-Net)
# ---------------------------------------------------------------------------


class HSVCSLoss(nn.Module):
    """
    HSV-CS Loss from PCF-Net (Remote Sensing 2026):
        L_hsvcs = λ_h · L_hue + λ_sv · L_sv

    where:
        L_hue = 1 − (H_Cx · H_Cy + H_Sx · H_Sy)   (cosine distance on unit circle)
        L_sv  = |V_x · S_x − V_y · S_y|             (saturation-value coupling)

    Args:
        lambda_h  (float): Weight for hue angular distance. Default: 1.0.
        lambda_sv (float): Weight for saturation-value coupling. Default: 1.0.
    """

    def __init__(self, lambda_h: float = 1.0, lambda_sv: float = 1.0):
        super().__init__()
        self.lambda_h = lambda_h
        self.lambda_sv = lambda_sv

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, 3, H, W) model output in [0, 1].
            target (Tensor): (N, 3, H, W) ground truth in [0, 1].

        Returns:
            Tensor: scalar HSV-CS loss.
        """
        from .models.pcf_modules import rgb_to_hsv_cs

        pred_hsv = rgb_to_hsv_cs(pred.clamp(0, 1))
        tgt_hsv = rgb_to_hsv_cs(target.clamp(0, 1))

        # Re-scale H_C, H_S from [0, 1] back to [-1, 1] for unit circle cosine
        hc_x = 2.0 * pred_hsv[:, 0:1, :, :] - 1.0
        hs_x = 2.0 * pred_hsv[:, 1:2, :, :] - 1.0
        hc_y = 2.0 * tgt_hsv[:, 0:1, :, :] - 1.0
        hs_y = 2.0 * tgt_hsv[:, 1:2, :, :] - 1.0

        # Cosine distance on the unit circle
        l_hue = torch.mean(1.0 - (hc_x * hc_y + hs_x * hs_y))

        # Saturation-Value coupling
        s_x, v_x = pred_hsv[:, 2:3, :, :], pred_hsv[:, 3:4, :, :]
        s_y, v_y = tgt_hsv[:, 2:3, :, :], tgt_hsv[:, 3:4, :, :]
        l_sv = F.l1_loss(v_x * s_x, v_y * s_y)

        return self.lambda_h * l_hue + self.lambda_sv * l_sv


# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """
    Weighted combination of L1, VGG perceptual, SSIM, and HSV-CS losses:

        loss = λ_l1 · L1 + λ_perc · Perceptual + λ_ssim · SSIM + λ_hsvcs · HSVCS

    Args:
        lambda_l1    (float): Weight for L1 loss.           Default: 1.0.
        lambda_perc  (float): Weight for perceptual loss.   Default: 0.1.
        lambda_ssim  (float): Weight for SSIM loss.         Default: 0.5.
        lambda_hsvcs (float): Weight for HSV-CS loss.       Default: 0.0.
        device (str | torch.device): Device for VGG backbone.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_ssim: float = 0.5,
        lambda_hsvcs: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_hsvcs = lambda_hsvcs

        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss(device) if lambda_perc else None
        self.ssim = SSIMLoss() if lambda_ssim else None
        self.hsvcs = HSVCSLoss() if lambda_hsvcs else None

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
                            ``"ssim_loss"``, ``"hsvcs"``, ``"total"``.
        """
        l_l1 = self.l1(pred, target)
        l_perc = self.perc(pred, target) if self.perc is not None else pred.new_zeros(())
        l_ssim = self.ssim(pred, target) if self.ssim is not None else pred.new_zeros(())
        l_hsvcs = self.hsvcs(pred, target) if self.hsvcs is not None else pred.new_zeros(())

        total = (
            self.lambda_l1 * l_l1
            + self.lambda_perc * l_perc
            + self.lambda_ssim * l_ssim
            + self.lambda_hsvcs * l_hsvcs
        )

        parts = {
            "l1": l_l1.item(),
            "perceptual": l_perc.item(),
            "ssim_loss": l_ssim.item(),
            "hsvcs": l_hsvcs.item(),
            "total": total.item(),
        }
        return total, parts
