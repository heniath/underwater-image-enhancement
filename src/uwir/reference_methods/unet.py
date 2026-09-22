"""Adapter for the exact RGB U-Net retained by this repository/paper."""

from __future__ import annotations

import torch

from uwir.losses import CompositeLoss
from uwir.models.unet import UNet5ch
from uwir.training.schedulers import CosineAnnealingRestartLR

from .base import ReferenceMethodAdapter, backward_and_step


class UNetAdapter(ReferenceMethodAdapter):
    method_name, display_name = "unet", "U-Net"

    def build(self) -> None:
        model = UNet5ch(in_channels=3, features=(64, 128, 256, 512), bilinear=True).to(self.device)
        self.criterion = CompositeLoss(
            lambda_l1=1.0, lambda_perc=1.0, lambda_ssim=0.0, device=self.device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        self.modules = {"restoration": model}
        self.optimizers = {"restoration": optimizer}
        self.schedulers = {
            "restoration": CosineAnnealingRestartLR(
                optimizer, periods=[100], restart_weights=[1.0], eta_min=1e-6
            )
        }

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        with self.autocast():
            prediction = self.modules["restoration"](degraded)
            loss, parts = self.criterion(prediction, target)
        backward_and_step(
            self,
            loss,
            self.optimizers["restoration"],
            accumulation_steps=accumulation_steps,
            update=update,
        )
        return parts

    def _inference_native(self, degraded):
        return self.modules["restoration"](degraded)

    @classmethod
    def provenance(cls):
        return {
            "method": cls.display_name,
            "paper_title": "Project comparison RGB U-Net baseline",
            "paper_url": None,
            "source_repository_url": "https://github.com/heniath/underwater-image-enhancement",
            "source_commit_sha": "32536e322aec30e3fe5af16548ec92e9601991b3",
            "license": "Apache-2.0",
            "implementation_source": "implementation/config actually used by this repository/paper",
            "important_integration_modifications": [
                "adapter wrapper only; architecture reused directly"
            ],
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "RGB [0,1]",
            "architecture": "4-level U-Net, widths 64/128/256/512, 1024 bottleneck, Conv-BN-ReLU x2, bilinear decoder, sigmoid head",
            "losses": "L1 + ImageNet VGG16 relu1_2/relu2_2 feature L1 (weights 1.0/1.0)",
            "optimizers": "Adam lr=2e-4",
            "schedulers": "single 100-epoch cosine-restart period, eta_min=1e-6",
            "trainable_initialization": "restoration U-Net initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "ImageNet VGG16 for perceptual loss",
            "native_output": "sigmoid RGB [0,1]",
        }
