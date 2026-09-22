"""UColor multi-colour encoder with input-only transmission guidance."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_hsv, rgb_to_lab
from scipy.ndimage import maximum_filter, median_filter
from torch import nn

from uwir.metrics import tiled_predict

from .base import ReferenceMethodAdapter, VGGFeatureLoss, backward_and_step


def _gdcp_transmission(rgb: torch.Tensor) -> torch.Tensor:
    """Input-only GDCP compatibility path derived from the UColor release scripts."""
    results = []
    for item in rgb.detach().float().cpu().permute(0, 2, 3, 1).numpy():
        luminance = 0.299 * item[..., 0] + 0.587 * item[..., 1] + 0.114 * item[..., 2]
        gy, gx = np.gradient(luminance)
        gradient = np.sqrt(gx * gx + gy * gy)
        gradient = maximum_filter(gradient, size=15, mode="reflect")
        scale = float(np.ptp(gradient))
        rough_depth = 1.0 - ((gradient - gradient.min()) / scale if scale > 1e-12 else 0.0)
        count = max(1, rough_depth.size // 1000)
        indices = np.argpartition(rough_depth.ravel(), -count)[-count:]
        atmosphere = item.reshape(-1, 3)[indices].mean(0)
        distance = np.abs(atmosphere.reshape(1, 1, 3) - item)
        normalizer = np.maximum(np.maximum(atmosphere, 1 - atmosphere), 1e-6)
        transmission = median_filter(np.max(distance / normalizer, axis=2), size=15, mode="reflect")
        low, high = float(transmission.min()), float(transmission.max())
        transmission = (
            np.full_like(transmission, 0.2)
            if high - low <= 1e-12
            else (transmission - low) / (high - low) * (high - 0.2) + 0.2
        )
        results.append(torch.from_numpy(transmission.astype(np.float32)).unsqueeze(0))
    return torch.stack(results).to(rgb.device, rgb.dtype)


class _ResidualStage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        def conv(source=cout):
            return nn.Conv2d(source, cout, 3, padding=1)

        self.layers = nn.ModuleList([conv(cin), *(conv() for _ in range(7))])

    def forward(self, x):
        first = F.relu(self.layers[0](x))
        value = first
        for layer in self.layers[1:3]:
            value = F.relu(layer(value))
        value = first + self.layers[3](value)
        second = F.relu(self.layers[4](value))
        value = second
        for layer in self.layers[5:7]:
            value = F.relu(layer(value))
        return second + self.layers[7](value)


class _ColourEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.one, self.two, self.three = (
            _ResidualStage(3, 128),
            _ResidualStage(128, 256),
            _ResidualStage(256, 512),
        )

    def forward(self, x):
        one = self.one(x)
        two = self.two(F.max_pool2d(one, 2))
        three = self.three(F.max_pool2d(two, 2))
        return one, two, three


class _AttentionFuse(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width * 3, width * 3 // 16, 1),
            nn.ReLU(True),
            nn.Conv2d(width * 3 // 16, width * 3, 1),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(width * 3, width, 3, padding=1)

    def forward(self, items):
        joined = torch.cat(items, 1)
        return self.project(joined * self.attention(joined))


class UColorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb, self.hsv, self.lab = _ColourEncoder(), _ColourEncoder(), _ColourEncoder()
        self.fuse1, self.fuse2, self.fuse3 = (
            _AttentionFuse(128),
            _AttentionFuse(256),
            _AttentionFuse(512),
        )
        self.decode3 = _ResidualStage(512, 512)
        self.up2 = _ResidualStage(512 + 256, 256)
        self.up1 = _ResidualStage(256 + 128, 128)
        self.output = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, rgb, transmission):
        spaces = self.rgb(rgb), self.hsv(rgb_to_hsv(rgb)), self.lab(rgb_to_lab(rgb))
        levels = [
            self.fuse1(tuple(item[0] for item in spaces)),
            self.fuse2(tuple(item[1] for item in spaces)),
            self.fuse3(tuple(item[2] for item in spaces)),
        ]
        inverse = 1 - transmission
        guide2 = F.max_pool2d(inverse, 2)
        guide3 = F.max_pool2d(inverse, 4)
        guided3 = self.decode3(levels[2] * (1 + guide3))
        decoded2 = self.up2(
            torch.cat((F.interpolate(guided3, scale_factor=2, mode="bilinear"), levels[1]), 1)
        )
        decoded2 = decoded2 * (1 + guide2)
        decoded1 = self.up1(
            torch.cat((F.interpolate(decoded2, scale_factor=2, mode="bilinear"), levels[0]), 1)
        )
        return self.output(decoded1 * (1 + inverse))


class UColorAdapter(ReferenceMethodAdapter):
    method_name, display_name = "ucolor", "UColor"

    def build(self) -> None:
        model = UColorNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.modules, self.optimizers = {"restoration": model}, {"restoration": optimizer}
        self.schedulers = {}
        self.perceptual = VGGFeatureLoss("vgg19", stop=36).to(self.device)

    def _forward(self, degraded):
        return self.modules["restoration"](degraded, _gdcp_transmission(degraded))

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        with self.autocast():
            prediction = self._forward(degraded)
            mse = F.mse_loss(prediction, target)
            perceptual = self.perceptual(prediction, target)
            loss = 5.0 * mse + 0.05 * perceptual
        backward_and_step(
            self,
            loss,
            self.optimizers["restoration"],
            accumulation_steps=accumulation_steps,
            update=update,
        )
        return {"total": loss.item(), "mse": mse.item(), "perceptual": perceptual.item()}

    def _inference_native(self, degraded):
        height, width = degraded.shape[-2:]
        pad_h, pad_w = (-height) % 4, (-width) % 4
        if pad_h or pad_w:
            mode = "reflect" if height > pad_h and width > pad_w else "replicate"
            degraded = F.pad(degraded, (0, pad_w, 0, pad_h), mode=mode)
        transmission = _gdcp_transmission(degraded)
        model_input = torch.cat((degraded, transmission), 1)

        def restore(tile):
            return self.modules["restoration"](tile[:, :3], tile[:, 3:])

        prediction = tiled_predict(restore, model_input, tile_size=256, overlap=32, factor=4)
        return prediction[..., :height, :width]

    def network_benchmark_call(self, degraded):
        return self.modules["restoration"], (degraded, _gdcp_transmission(degraded))

    @classmethod
    def provenance(cls):
        return {
            "method": cls.display_name,
            "paper_title": "Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding",
            "paper_url": "https://doi.org/10.1109/TIP.2021.3076367",
            "source_repository_url": "https://github.com/Li-Chongyi/Ucolor",
            "source_commit_sha": "08df481135b400be1083ee705801dd8bb60dc82c",
            "supporting_reimplementation": {
                "url": "https://github.com/CV-Reimplementation/Ucolor-Reimplementation",
                "commit": "2cda66de8c56ebb46b81979361e2aa2f5ebafee4",
            },
            "license": "MIT stated in README; official code distributed separately",
            "implementation_source": "faithful PyTorch reimplementation informed by official release and CV-Reimplementation/Ucolor-Reimplementation",
            "important_integration_modifications": [
                "PyTorch/NCHW",
                "on-the-fly input-only GDCP-compatible transmission estimation",
                "common checkpoint and evaluator wrappers",
            ],
            "reproduction_ambiguity": "The authors' separate TensorFlow bundle was inspected but is not vendored. This attributed PyTorch port preserves its three color spaces, widths, residual stages, channel attention, transmission-guided decoder, and released loss/optimizer, but is not checkpoint-compatible: the original RGB encoder has additional cross-color concatenations, and the local input-only transmission estimator is not bit-identical to the separately generated official depth files.",
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "RGB/HSV/Lab encoders plus one-channel GDCP-compatible transmission/depth guidance derived only from degraded RGB",
            "architecture": "three RGB/HSV/Lab encoders with 128/256/512 widths and two residual blocks per scale; channel-attention fusion; 512/256/128 transmission-guided decoder",
            "losses": "5*MSE + 0.05*ImageNet VGG19 relu5_4 feature MSE",
            "optimizers": "Adam lr=1e-4, betas=(0.5,0.999)",
            "schedulers": "none in the official training code",
            "trainable_initialization": "restoration network initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "ImageNet VGG19 feature extractor (fixed)",
            "native_output": "linear RGB; common inference adapter clips once to [0,1]",
        }
