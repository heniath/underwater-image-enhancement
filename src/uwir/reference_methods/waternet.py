"""Water-Net gated fusion and deterministic WB/CLAHE/gamma input construction."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import ReferenceMethodAdapter, VGGFeatureLoss, backward_and_step


def _branch(cin: int, widths: tuple[int, ...], kernels: tuple[int, ...], *, sigmoid=False):
    layers: list[nn.Module] = []
    for index, (cout, kernel) in enumerate(zip(widths, kernels, strict=True)):
        layers.append(nn.Conv2d(cin, cout, kernel, padding=kernel // 2))
        if index < len(widths) - 1:
            layers.append(nn.ReLU(True))
        cin = cout
    if sigmoid:
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence = _branch(
            12, (128, 128, 128, 64, 64, 64, 3), (7, 5, 3, 1, 7, 5, 3), sigmoid=True
        )
        self.refine_wb = _branch(6, (32, 32, 3), (7, 5, 3))
        self.refine_he = _branch(6, (32, 32, 3), (7, 5, 3))
        self.refine_gc = _branch(6, (32, 32, 3), (7, 5, 3))

    def forward(self, rgb, wb, he, gc):
        weights = self.confidence(torch.cat((rgb, wb, he, gc), 1))
        refined = (
            self.refine_wb(torch.cat((rgb, wb), 1)),
            self.refine_he(torch.cat((rgb, he), 1)),
            self.refine_gc(torch.cat((rgb, gc), 1)),
        )
        return sum(item * weights[:, index : index + 1] for index, item in enumerate(refined))


def _waternet_inputs(rgb: torch.Tensor):
    # Exact gamma exponent and percentile-scaled white balance from the authors' MATLAB scripts.
    flat = rgb.flatten(2)
    sums = flat.sum(2).clamp_min(1e-6)
    ratio = sums.max(1, keepdim=True).values / sums
    channels = []
    for channel in range(3):
        values = flat[:, channel]
        low_q = (0.005 * ratio[:, channel]).clamp(0, 0.49)
        high_q = (1.0 - 0.005 * ratio[:, channel]).clamp(0.51, 1)
        balanced = []
        for sample, low, high in zip(values, low_q, high_q, strict=True):
            lo, hi = torch.quantile(sample, torch.stack((low, high)))
            balanced.append((sample.clamp(lo, hi) - lo) / (hi - lo).clamp_min(1e-6))
        channels.append(torch.stack(balanced).view(rgb.shape[0], 1, *rgb.shape[-2:]))
    wb = torch.cat(channels, 1)
    gamma = rgb.clamp_min(0).pow(0.7)
    he_images = []
    for image in rgb.detach().float().cpu().permute(0, 2, 3, 1).numpy():
        u8 = np.rint(image.clip(0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        he_images.append(torch.from_numpy(enhanced).permute(2, 0, 1))
    he = torch.stack(he_images).to(rgb.device, rgb.dtype)
    return wb, he, gamma


class WaterNetAdapter(ReferenceMethodAdapter):
    method_name, display_name = "water_net", "Water-Net"

    def build(self) -> None:
        model = WaterNet().to(self.device)
        self.content = VGGFeatureLoss("vgg19", 36).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.modules, self.optimizers = {"restoration": model}, {"restoration": optimizer}

    def _forward(self, degraded):
        return self.modules["restoration"](degraded, *_waternet_inputs(degraded))

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        with self.autocast():
            prediction = self._forward(degraded)
            pixel = F.l1_loss(prediction, target)
            texture = self.content(prediction, target)
            loss = pixel + 0.05 * texture
        backward_and_step(
            self,
            loss,
            self.optimizers["restoration"],
            accumulation_steps=accumulation_steps,
            update=update,
        )
        return {"total": loss.item(), "l1": pixel.item(), "vgg19_relu5_4": texture.item()}

    def _inference_native(self, degraded):
        return self._forward(degraded)

    def network_benchmark_call(self, degraded):
        return self.modules["restoration"], (degraded, *_waternet_inputs(degraded))

    @classmethod
    def provenance(cls):
        return {
            "method": cls.display_name,
            "paper_title": "An Underwater Image Enhancement Benchmark Dataset and Beyond",
            "paper_url": "https://arxiv.org/abs/1901.05495",
            "source_repository_url": "https://github.com/Li-Chongyi/Water-Net_Code",
            "source_commit_sha": "1dbd59af235de2ed3d034f605f587ab199305bd7",
            "license": "no standalone license file; academic-use attribution in README",
            "implementation_source": "faithful PyTorch compatibility port of official TensorFlow 1.x topology",
            "important_integration_modifications": [
                "NCHW PyTorch port",
                "OpenCV CLAHE replaces MATLAB adapthisteq",
                "derived candidates generated after paired crop",
            ],
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "RGB plus input-only white-balanced, CLAHE, and gamma=0.7 candidates",
            "architecture": "three refinement branches plus 3-map gated confidence fusion",
            "losses": "L1 + 0.05*ImageNet VGG19 relu5_4 feature MSE",
            "optimizers": "Adam lr=1e-3, beta1=0.9 (official executed code)",
            "schedulers": "none",
            "trainable_initialization": "gated fusion network initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "ImageNet VGG19 perceptual network",
            "native_output": "fusion RGB; evaluator clamps once to [0,1]",
        }
