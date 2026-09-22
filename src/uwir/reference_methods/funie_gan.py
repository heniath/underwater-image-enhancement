"""PyTorch compatibility port of the authors' paired FUnIE-GAN v2 path."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import ReferenceMethodAdapter, VGGFeatureLoss


class _Conv(nn.Sequential):
    def __init__(self, cin, cout, kernel, *, pool=False, bn=True):
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, kernel, padding="same"), nn.ReLU(True)]
        if bn:
            layers.append(nn.BatchNorm2d(cout, momentum=0.25))
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class FUnIEGenerator(nn.Module):
    """Official generator2 topology: 32/64/64/128/128/256 with three skips."""

    def __init__(self):
        super().__init__()
        self.d1 = _Conv(3, 32, 5, pool=True, bn=False)
        self.d2 = _Conv(32, 64, 4)
        self.d3 = _Conv(64, 64, 4, pool=True)
        self.d4 = _Conv(64, 128, 3)
        self.d5 = _Conv(128, 128, 3, pool=True)
        self.d6 = _Conv(128, 256, 3)
        self.u1 = _Conv(256, 256, 3)
        self.u2 = _Conv(384, 256, 3)
        self.u3 = _Conv(320, 128, 3)
        self.c4 = _Conv(160, 128, 3)
        self.c5 = _Conv(128, 256, 3)
        self.head = nn.Conv2d(256, 3, 4, padding="same")

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        u1 = torch.cat((self.u1(F.interpolate(d6, scale_factor=2)), d4), 1)
        u2 = torch.cat((self.u2(F.interpolate(u1, scale_factor=2)), d2), 1)
        # The official skip is d1 before its pooling. Reconstruct it from a no-pool first block.
        first = F.relu(F.conv2d(x, self.d1[0].weight, self.d1[0].bias, padding="same"))
        u3 = torch.cat((self.u3(F.interpolate(u2, scale_factor=2)), first), 1)
        return torch.tanh(self.head(self.c5(self.c4(u3))))


class FUnIEDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (6, 32, 64, 128, 256)
        blocks = []
        for index, (cin, cout) in enumerate(zip(channels[:-1], channels[1:], strict=True)):
            blocks.extend([nn.Conv2d(cin, cout, 3, 2, 1), nn.ReLU(True)])
            if index:
                blocks.append(nn.BatchNorm2d(cout, momentum=0.2))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(256, 1, 4, padding="same")

    def forward(self, candidate, degraded):
        return self.head(self.body(torch.cat((candidate, degraded), 1)))


class FUnIEGANAdapter(ReferenceMethodAdapter):
    method_name, display_name = "funie_gan", "FUnIE-GAN"

    def build(self) -> None:
        generator = FUnIEGenerator().to(self.device)
        discriminator = FUnIEDiscriminator().to(self.device)
        self.content = VGGFeatureLoss("vgg19", 32).to(self.device)
        self.modules = {"generator": generator, "discriminator": discriminator}
        self.optimizers = {
            "generator": torch.optim.Adam(generator.parameters(), 3e-4, betas=(0.5, 0.999)),
            "discriminator": torch.optim.Adam(discriminator.parameters(), 3e-4, betas=(0.5, 0.999)),
        }

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        degraded_native, target_native = degraded.mul(2).sub(1), target.mul(2).sub(1)
        generator, discriminator = self.modules["generator"], self.modules["discriminator"]
        with self.autocast():
            generated = generator(degraded_native)
            real_score = discriminator(target_native, degraded_native)
            fake_score = discriminator(generated.detach(), degraded_native)
            d_loss = 0.5 * (
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
            reconstruction = 0.7 * mae + 0.3 * content
            g_loss = 0.2 * adversarial + 0.8 * reconstruction
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
        pad_h, pad_w = (-height) % 8, (-width) % 8
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
            "source_repository_url": "https://github.com/IRVLab/funie-gan",
            "source_commit_sha": "b844d9071dadf308ecfeee11aaf602826a9ca6fe",
            "license": "MIT",
            "implementation_source": "faithful PyTorch compatibility port of official Keras paired generator2/discriminator",
            "important_integration_modifications": [
                "NCHW PyTorch port",
                "common RGB dataset adapter",
                "complete resumable GAN state",
            ],
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "RGB scaled to [-1,1]",
            "architecture": "official generator2; conditional four-block PatchGAN discriminator",
            "losses": "0.2 LSGAN-MSE + 0.8*(0.7 MAE + 0.3 VGG19 block5_conv2 feature MSE)",
            "optimizers": "Adam(G) and Adam(D), lr=3e-4, betas=(0.5,0.999)",
            "schedulers": "none",
            "trainable_initialization": "generator and discriminator initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "ImageNet VGG19 content features",
            "native_output": "tanh [-1,1], converted once to RGB [0,1]",
        }
