"""Adapter around the authors' released PyTorch FUnIE-GAN implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import ReferenceMethodAdapter, VGGFeatureLoss


class _UNetDown(nn.Sequential):
    """Released ``UNetDown`` block, kept local for the common adapter."""

    def __init__(self, cin: int, cout: int, *, bn: bool = True):
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(cout, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        super().__init__(*layers)


class _UNetUp(nn.Module):
    """Released ``UNetUp`` block with its skip concatenation."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cout, momentum=0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        return torch.cat((self.model(x), skip), 1)


def _official_weights_normal(module: nn.Module) -> None:
    """Initialization from the released ``PyTorch/nets/commons.py``."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)


class FUnIEGenerator(nn.Module):
    """Author-released five-level PyTorch U-Net generator."""

    def __init__(self):
        super().__init__()
        self.down1 = _UNetDown(3, 32, bn=False)
        self.down2 = _UNetDown(32, 128)
        self.down3 = _UNetDown(128, 256)
        self.down4 = _UNetDown(256, 256)
        self.down5 = _UNetDown(256, 256, bn=False)
        self.up1 = _UNetUp(256, 256)
        self.up2 = _UNetUp(512, 256)
        self.up3 = _UNetUp(512, 128)
        self.up4 = _UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 3, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        return self.final(self.up4(u3, d1))


class FUnIEDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        blocks = ((6, 32, False), (32, 64, True), (64, 128, True), (128, 256, True))
        for cin, cout, bn in blocks:
            layers.append(nn.Conv2d(cin, cout, 4, 2, 1))
            if bn:
                layers.append(nn.BatchNorm2d(cout, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.extend((nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(256, 1, 4, padding=1, bias=False)))
        self.model = nn.Sequential(*layers)

    def forward(self, candidate, degraded):
        return self.model(torch.cat((candidate, degraded), 1))


class FUnIEGANAdapter(ReferenceMethodAdapter):
    method_name, display_name = "funie_gan", "FUnIE-GAN"

    def build(self) -> None:
        generator = FUnIEGenerator().to(self.device)
        discriminator = FUnIEDiscriminator().to(self.device)
        generator.apply(_official_weights_normal)
        discriminator.apply(_official_weights_normal)
        self.content = VGGFeatureLoss("vgg19", 32).to(self.device)
        self.modules = {"generator": generator, "discriminator": discriminator}
        self.optimizers = {
            "generator": torch.optim.Adam(generator.parameters(), 3e-4, betas=(0.5, 0.99)),
            "discriminator": torch.optim.Adam(discriminator.parameters(), 3e-4, betas=(0.5, 0.99)),
        }

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        degraded_native, target_native = degraded.mul(2).sub(1), target.mul(2).sub(1)
        generator, discriminator = self.modules["generator"], self.modules["discriminator"]
        with self.autocast():
            generated = generator(degraded_native)
            real_score = discriminator(target_native, degraded_native)
            fake_score = discriminator(generated.detach(), degraded_native)
            d_loss = 5.0 * (
                F.mse_loss(real_score, torch.ones_like(real_score))
                + F.mse_loss(fake_score, torch.zeros_like(fake_score))
            )
        self.scaler.scale(d_loss / accumulation_steps).backward()
        if update:
            self.scaler.step(self.optimizers["discriminator"])
            self.optimizers["discriminator"].zero_grad(set_to_none=True)

        discriminator.requires_grad_(False)
        with self.autocast():
            generated = generator(degraded_native)
            score = discriminator(generated, degraded_native)
            adversarial = F.mse_loss(score, torch.ones_like(score))
            mae = F.l1_loss(generated, target_native)
            content = self.content(generated, target_native)
            g_loss = adversarial + 7.0 * mae + 3.0 * content
        self.scaler.scale(g_loss / accumulation_steps).backward()
        if update:
            self.scaler.step(self.optimizers["generator"])
            self.scaler.update()
            self.optimizers["generator"].zero_grad(set_to_none=True)
        discriminator.requires_grad_(True)
        return {
            "generator": g_loss.item(),
            "discriminator": d_loss.item(),
            "adversarial": adversarial.item(),
            "mae": mae.item(),
            "content": content.item(),
        }

    def _inference_native(self, degraded):
        height, width = degraded.shape[-2:]
        pad_h, pad_w = (-height) % 32, (-width) % 32
        if pad_h or pad_w:
            mode = "reflect" if height > pad_h and width > pad_w else "replicate"
            degraded = F.pad(degraded, (0, pad_w, 0, pad_h), mode=mode)
        prediction = self.modules["generator"](degraded.mul(2).sub(1)).add(1).mul(0.5)
        return prediction[..., :height, :width]

    def network_benchmark_call(self, degraded):
        return self.modules["generator"], (degraded.mul(2).sub(1),)

    @classmethod
    def provenance(cls):
        return {
            "method": cls.display_name,
            "paper_title": "Fast Underwater Image Enhancement for Improved Visual Perception",
            "paper_url": "https://arxiv.org/abs/1903.09766",
            "source_repository_url": "https://github.com/xahidbuffon/FUnIE-GAN",
            "source_commit_sha": "8f934c834c94e007b00866186b9ee624dc2b7b69",
            "license": "MIT",
            "implementation_source": "adapter around the author-released native PyTorch generator and discriminator",
            "important_integration_modifications": [
                "common RGB dataset adapter",
                "complete resumable GAN state",
            ],
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "RGB scaled to [-1,1]",
            "architecture": "released five-level PyTorch U-Net generator; conditional four-block PatchGAN discriminator",
            "losses": (
                "generator: LSGAN-MSE + 7*MAE + 3*VGG19 conv5_2 feature MSE; "
                "discriminator: 10*0.5*(real MSE + fake MSE)"
            ),
            "optimizers": "Adam(G) and Adam(D), lr=3e-4, betas=(0.5,0.99)",
            "schedulers": "none",
            "trainable_initialization": "generator and discriminator initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "ImageNet VGG19 content features",
            "native_output": "tanh [-1,1], converted once to RGB [0,1]",
        }
