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


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_filter(x: torch.Tensor, win_1d: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    win_2d = torch.outer(win_1d, win_1d).view(1, 1, win_1d.shape[0], win_1d.shape[0]).repeat(channels, 1, 1, 1).to(x.device, x.dtype)
    pad = win_1d.shape[0] // 2
    return F.conv2d(x, win_2d, padding=pad, groups=channels)


class SSIMLoss(nn.Module):
    """
    SSIM-based loss: ``1 − mean(SSIM map)``.
    Uses ``kornia.metrics.ssim`` if available, with pure-PyTorch fallback.

    Args:
        window_size (int): Gaussian window size. Default: 11.
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        try:
            import kornia
            self._kornia = kornia
        except ImportError:
            self._kornia = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, C, H, W) in [0, 1].
            target (Tensor): (N, C, H, W) in [0, 1].

        Returns:
            Tensor: scalar loss ∈ [0, 2].
        """
        pred = pred.float()
        target = target.float()

        if self._kornia is not None:
            ssim_map = self._kornia.metrics.ssim(pred, target, self.window_size)
            return 1.0 - ssim_map.mean()

        # Pure PyTorch fallback
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        win_1d = _fspecial_gauss_1d(self.window_size, 1.5)
        mu1 = _gaussian_filter(pred, win_1d)
        mu2 = _gaussian_filter(target, win_1d)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = _gaussian_filter(pred * pred, win_1d) - mu1_sq
        sigma2_sq = _gaussian_filter(target * target, win_1d) - mu2_sq
        sigma12 = _gaussian_filter(pred * target, win_1d) - mu1_mu2
        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
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

        pred_f = pred.float().clamp(0.0, 1.0)
        tgt_f = target.float().clamp(0.0, 1.0)
        pred_hsv = rgb_to_hsv_cs(pred_f)
        tgt_hsv = rgb_to_hsv_cs(tgt_f)

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
    Weighted combination of L1, MSE, VGG perceptual, SSIM, and HSV-CS losses:

        loss = λ_l1 · L1 + λ_mse · MSE + λ_perc · Perceptual + λ_ssim · SSIM + λ_hsvcs · HSVCS + λ_redeg · ReDeg

    Args:
        lambda_l1    (float): Weight for L1 loss.           Default: 1.0.
        lambda_mse   (float): Weight for MSE loss.          Default: 0.0.
        lambda_perc  (float): Weight for perceptual loss.   Default: 0.1.
        lambda_ssim  (float): Weight for SSIM loss.         Default: 0.5.
        lambda_hsvcs (float): Weight for HSV-CS loss.       Default: 0.0.
        lambda_hue   (float): Weight for Hue loss.          Default: 1.0.
        lambda_sv    (float): Weight for SV loss.           Default: 1.0.
        lambda_redeg (float): Weight for re-degradation.    Default: 0.0.
        device (str | torch.device): Device for VGG backbone.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_mse: float = 0.0,
        lambda_perc: float = 0.1,
        lambda_ssim: float = 0.5,
        lambda_hsvcs: float = 0.0,
        lambda_hue: float = 1.0,
        lambda_sv: float = 1.0,
        lambda_redeg: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_mse = lambda_mse
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_hsvcs = lambda_hsvcs
        self.lambda_redeg = lambda_redeg

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss() if lambda_mse else None
        self.perc = VGGPerceptualLoss(device) if lambda_perc else None
        self.ssim = SSIMLoss() if lambda_ssim else None
        self.hsvcs = HSVCSLoss(lambda_h=lambda_hue, lambda_sv=lambda_sv) if lambda_hsvcs else None

    def forward(
        self,
        pred: torch.Tensor | tuple,
        target: torch.Tensor,
        input_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred   (Tensor | tuple): (N, 3, H, W) model output in [0, 1], or
                                    tuple (J, I_redeg, t, B) from physical models.
            target (Tensor):        (N, 3, H, W) ground truth in [0, 1].
            input_image (Tensor):   Optional (N, 3, H, W) input image for re-degradation loss.

        Returns:
            total (Tensor): Scalar combined loss.
            parts (dict):   Per-component losses as Python floats.
        """
        i_redeg = None
        if isinstance(pred, (tuple, list)):
            j_pred = pred[0]
            if len(pred) > 1:
                i_redeg = pred[1]
        else:
            j_pred = pred

        l_l1 = self.l1(j_pred, target)
        l_mse = self.mse(j_pred, target) if self.mse is not None else j_pred.new_zeros(())
        l_perc = self.perc(j_pred, target) if self.perc is not None else j_pred.new_zeros(())
        l_ssim = self.ssim(j_pred, target) if self.ssim is not None else j_pred.new_zeros(())
        l_hsvcs = self.hsvcs(j_pred, target) if self.hsvcs is not None else j_pred.new_zeros(())

        total = (
            self.lambda_l1 * l_l1
            + self.lambda_mse * l_mse
            + self.lambda_perc * l_perc
            + self.lambda_ssim * l_ssim
            + self.lambda_hsvcs * l_hsvcs
        )

        parts = {
            "l1": l_l1.item(),
            "mse": l_mse.item(),
            "perceptual": l_perc.item(),
            "ssim_loss": l_ssim.item(),
            "hsvcs": l_hsvcs.item(),
        }

        if self.lambda_redeg > 0 and i_redeg is not None and input_image is not None:
            inp_rgb = input_image[:, :3, :, :] if input_image.shape[1] > 3 else input_image
            l_redeg = self.l1(i_redeg, inp_rgb)
            if self.ssim is not None:
                l_redeg = l_redeg + 0.5 * self.ssim(i_redeg, inp_rgb)
            total = total + self.lambda_redeg * l_redeg
            parts["redeg"] = l_redeg.item()

        parts["total"] = total.item()
        return total, parts
