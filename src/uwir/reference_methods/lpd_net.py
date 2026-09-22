"""Paper-guided LPD-Net reimplementation with an input-only MSRCR prior.

The publication does not provide source code or all channel/block details. This
module therefore preserves the documented method components and training loss,
but deliberately identifies itself as a faithful reimplementation rather than
an official or checkpoint-compatible port.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_hsv, rgb_to_lab
from kornia.metrics import ssim
from torch import nn

from .base import ReferenceMethodAdapter, backward_and_step


def _msrcr_prior(rgb: torch.Tensor) -> torch.Tensor:
    """Generate a deterministic MSRCR prior from degraded RGB only."""
    priors = []
    for sample in rgb.detach().float().cpu().permute(0, 2, 3, 1).numpy():
        image = np.clip(sample, 0, 1).astype(np.float32) + 1 / 255
        log_image = np.log(image)
        retinex = np.zeros_like(image)
        # Canonical MSRCR scales; the paper states multiple Gaussian scales but
        # does not publish their numeric values.
        for sigma in (15.0, 80.0, 250.0):
            kernel = min(int(2 * np.ceil(3 * sigma) + 1), 127)
            blurred = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
            retinex += log_image - np.log(np.maximum(blurred, 1e-6))
        retinex /= 3.0
        restoration = 46.0 * (
            np.log(125.0 * image) - np.log(np.maximum(image.sum(axis=2, keepdims=True), 1e-6))
        )
        enhanced = restoration * retinex
        # Robust per-channel normalization prevents isolated MSRCR extrema from
        # dominating the learned gate while remaining input-only.
        for channel in range(3):
            low, high = np.percentile(enhanced[..., channel], (1, 99))
            if high - low > 1e-6:
                enhanced[..., channel] = (enhanced[..., channel] - low) / (high - low)
            else:
                enhanced[..., channel] = image[..., channel]
        prior = np.clip(enhanced, 0, 1).astype(np.float32)
        priors.append(torch.from_numpy(prior).permute(2, 0, 1))
    return torch.stack(priors).to(rgb.device, rgb.dtype)


class _LKA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dilated = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        attention = self.project(self.dilated(self.depthwise(x)))
        return self.norm(x + x * attention)


class _HDA(nn.Module):
    """Hybrid frequency-domain and spatial channel attention."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.frequency = nn.Conv2d(channels * 2, channels * 2, 1)
        self.alpha = nn.Parameter(torch.ones(()))
        self.beta = nn.Parameter(torch.ones(()))
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        packed = torch.cat((spectrum.real, spectrum.imag), dim=1).to(x.dtype)
        real, imaginary = self.frequency(packed).chunk(2, dim=1)
        frequency = torch.fft.irfft2(
            torch.complex(real.float(), imaginary.float()), s=x.shape[-2:], norm="ortho"
        ).to(x.dtype)
        spatial = x * self.channel(x)
        return self.norm(x + self.alpha * spatial + self.beta * frequency)


class _CSC(nn.Module):
    """Composite horizontal, vertical, and square shape convolution."""

    def __init__(self, channels: int):
        super().__init__()
        branch = max(8, channels // 3)
        self.horizontal = nn.Conv2d(channels, branch, (1, 5), padding=(0, 2))
        self.vertical = nn.Conv2d(channels, branch, (5, 1), padding=(2, 0))
        self.square = nn.Conv2d(channels, branch, 3, padding=1)
        self.project = nn.Conv2d(branch * 3, channels, 1)

    def forward(self, x):
        values = (self.horizontal(x), self.vertical(x), self.square(x))
        return x + self.project(torch.cat(values, dim=1))


class _PolyBlock(nn.Module):
    def __init__(self, channels: int, *, csc: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            _HDA(channels),
            _LKA(channels),
            _CSC(channels) if csc else nn.Identity(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.layers(x)


class LPDNet(nn.Module):
    """Dual-stream adaptive gate followed by a poly-scale U-shaped refiner."""

    def __init__(self, width: int = 48):
        super().__init__()
        self.raw_embed = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(True))
        self.prior_embed = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(True))
        self.gate = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.encoder1 = _PolyBlock(width)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.encoder2 = _PolyBlock(width * 2)
        self.down2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(_PolyBlock(width * 4), _PolyBlock(width * 4))
        self.up2 = nn.Conv2d(width * 4, width * 2, 1)
        self.decoder2 = _PolyBlock(width * 2, csc=True)
        self.up1 = nn.Conv2d(width * 2, width, 1)
        self.decoder1 = _PolyBlock(width, csc=True)
        self.output = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, raw, prior):
        raw_features, prior_features = self.raw_embed(raw), self.prior_embed(prior)
        difference = torch.abs(raw_features - prior_features)
        gamma = self.gate(torch.cat((raw_features, prior_features, difference), dim=1))
        fused = gamma * raw_features + (1 - gamma) * prior_features
        one = self.encoder1(fused)
        two = self.encoder2(self.down1(one))
        value = self.bottleneck(self.down2(two))
        value = F.interpolate(self.up2(value), size=two.shape[-2:], mode="bilinear") + two
        value = self.decoder2(value)
        value = F.interpolate(self.up1(value), size=one.shape[-2:], mode="bilinear") + one
        residual = self.output(self.decoder1(value))
        return torch.sigmoid(residual + raw)


def _differentiable_uciqe(image: torch.Tensor) -> torch.Tensor:
    lab = rgb_to_lab(image.clamp(0, 1))
    chroma = torch.sqrt(lab[:, 1].square() + lab[:, 2].square() + 1e-12) / 128.0
    sigma_c = chroma.flatten(1).std(dim=1, unbiased=False)
    luminance = lab[:, 0].flatten(1) / 100.0
    contrast = torch.quantile(luminance, 0.99, dim=1) - torch.quantile(luminance, 0.01, dim=1)
    saturation = rgb_to_hsv(image.clamp(0, 1))[:, 1].flatten(1).mean(dim=1)
    return (0.4680 * sigma_c + 0.2745 * contrast + 0.2576 * saturation).mean()


class LPDNetAdapter(ReferenceMethodAdapter):
    method_name, display_name = "lpd_net", "LPD-Net"

    def build(self) -> None:
        model = LPDNet().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        self.modules, self.optimizers = {"restoration": model}, {"restoration": optimizer}
        self.schedulers = {
            "restoration": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
        }

    def _forward(self, degraded):
        return self.modules["restoration"](degraded, _msrcr_prior(degraded))

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        with self.autocast():
            prediction = self._forward(degraded)
            pixel = F.smooth_l1_loss(prediction, target)
            structural = 1 - ssim(prediction, target, window_size=11, max_val=1.0).mean()
            quality = 1 - _differentiable_uciqe(prediction)
            laplacian = (
                prediction.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                .view(1, 1, 3, 3)
                .repeat(3, 1, 1, 1)
            )
            pred_edges = F.conv2d(prediction, laplacian, padding=1, groups=3)
            target_edges = F.conv2d(target, laplacian, padding=1, groups=3)
            boundary = F.l1_loss(pred_edges, target_edges)
            loss = pixel + 0.2 * structural + 0.01 * quality + 0.1 * boundary
        backward_and_step(
            self,
            loss,
            self.optimizers["restoration"],
            accumulation_steps=accumulation_steps,
            update=update,
        )
        return {
            "total": loss.item(),
            "smooth_l1": pixel.item(),
            "ssim_loss": structural.item(),
            "uciqe_loss": quality.item(),
            "laplacian": boundary.item(),
        }

    def _inference_native(self, degraded):
        return self._forward(degraded)

    def network_benchmark_call(self, degraded):
        return self.modules["restoration"], (degraded, _msrcr_prior(degraded))

    @classmethod
    def provenance(cls):
        return {
            "method": cls.display_name,
            "paper_title": "LPD-Net: lightweight adaptive weight-driven dual-stream network guided by physics-inspired priors for structure-preserving underwater image enhancement",
            "paper_url": "https://doi.org/10.3389/fcomp.2026.1908061",
            "source_repository_url": None,
            "source_commit_sha": "not_available",
            "license": "paper CC BY; no source-code license or repository provided",
            "implementation_source": "paper-guided faithful reimplementation from the supplied publication",
            "important_integration_modifications": [
                "PyTorch implementation of the documented ANGF/HDA/LKA/CSC topology",
                "input-only CPU MSRCR preprocessing",
                "100-epoch external benchmark budget replaces the paper's 500 epochs",
                "batch 2 with two-step accumulation preserves effective batch 4",
            ],
            "reproduction_ambiguity": "No official code was located. The paper does not state exact channel widths, block counts, MSRCR scales/normalization, AdamW weight decay, or differentiable UCIQE normalization. These values are explicitly local compatibility choices, so this must not be described as an exact reproduction.",
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "raw RGB plus deterministic MSRCR prior derived only from degraded RGB",
            "architecture": "paper-guided dual 3x3 embeddings, spatial difference gate, U-shaped HDA/LKA refiner, CSC decoder, global residual sigmoid output; local width=48 configuration",
            "losses": "SmoothL1 + 0.2*(1-SSIM) + 0.01*(1-differentiable UCIQE) + 0.1*L1 Laplacian boundary",
            "optimizers": "AdamW lr=2e-4 (weight decay unspecified by paper; PyTorch default 0.01)",
            "schedulers": "CosineAnnealingLR over standardized 100 epochs, eta_min=1e-6",
            "trainable_initialization": "all neural modules initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "none",
            "native_output": "sigmoid of learned residual plus raw RGB, [0,1]",
            "paper_training_difference": "paper reports batch 2, seed 3407, and 500 epochs; benchmark uses effective batch 4, seeds 0/1/2, and 100 epochs",
        }
