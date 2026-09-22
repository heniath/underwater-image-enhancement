"""UWFormer wavelet low-frequency transformer / high-frequency Fourier port."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from kornia.metrics import ssim
from torch import nn

from .base import ReferenceMethodAdapter, VGGFeatureLoss, backward_and_step


def _haar_dwt(x):
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) * 0.5, torch.stack(
        ((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5), 2
    )


def _haar_idwt(low, high):
    lh, hl, hh = high.unbind(2)
    a, b = (low + lh + hl + hh) * 0.5, (low - lh + hl - hh) * 0.5
    c, d = (low + lh - hl - hh) * 0.5, (low - lh - hl + hh) * 0.5
    result = low.new_empty(low.shape[0], low.shape[1], low.shape[2] * 2, low.shape[3] * 2)
    result[..., 0::2, 0::2], result[..., 0::2, 1::2] = a, b
    result[..., 1::2, 0::2], result[..., 1::2, 1::2] = c, d
    return result


class _WindowTransformer(nn.Module):
    def __init__(self, width=48, heads=4, window=8):
        super().__init__()
        self.window = window
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        window = self.window
        pad_h, pad_w = (-height) % window, (-width) % window
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        hp, wp = x.shape[-2:]
        tokens = (
            x.permute(0, 2, 3, 1)
            .reshape(batch, hp // window, window, wp // window, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window, channels)
        )
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        x = (
            tokens.reshape(batch, hp // window, wp // window, window, window, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, hp, wp, channels)
            .permute(0, 3, 1, 2)
        )
        return x[..., :height, :width]


class _LowFrequencyPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(3, 48, 3, padding=1)
        self.blocks = nn.Sequential(*[_WindowTransformer() for _ in range(6)])
        self.exit = nn.Conv2d(48, 3, 3, padding=1)

    def forward(self, x):
        return x + self.exit(self.blocks(self.entry(x)))


class _FourierResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 1), nn.BatchNorm2d(width * 2), nn.ReLU(True)
        )
        self.local = nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        # CUDA half-precision FFT only supports power-of-two signal dimensions.
        # Native-resolution evaluation includes arbitrary image sizes, so keep
        # the spectral branch in float32 while allowing the local branch to use AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            packed = torch.cat((spectrum.real, spectrum.imag), 1)
            packed = self.spectral(packed)
            real, imaginary = packed.chunk(2, 1)
            global_features = torch.fft.irfft2(
                torch.complex(real, imaginary), s=x.shape[-2:], norm="ortho"
            )
        return x + self.local(x) + global_features.to(x.dtype)


class _HighFrequencyPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(9, 64, 9, padding=4, padding_mode="reflect")
        self.blocks = nn.Sequential(*[_FourierResidual(64) for _ in range(9)])
        self.exit = nn.Conv2d(64, 9, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        return x + self.exit(self.blocks(F.prelu(self.entry(x), x.new_tensor(0.25))))


class UWFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.low, self.high = _LowFrequencyPath(), _HighFrequencyPath()

    def forward(self, x):
        height, width = x.shape[-2:]
        x = F.pad(x, (0, width % 2, 0, height % 2), mode="reflect")
        low, high = _haar_dwt(x)
        low = self.low(low)
        flat_high = high.reshape(high.shape[0], 9, *high.shape[-2:])
        high = self.high(flat_high).reshape(high.shape)
        return _haar_idwt(low, high)[..., :height, :width]


class UWFormerAdapter(ReferenceMethodAdapter):
    method_name, display_name = "uwformer", "UWFormer"

    def build(self) -> None:
        model = UWFormer().to(self.device)
        self.perceptual = VGGFeatureLoss("vgg16", 16).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
        self.modules, self.optimizers = {"restoration": model}, {"restoration": optimizer}
        # Authors' checked-in LR_MIN equals LR_INITIAL; retain the resulting constant cosine schedule.
        self.schedulers = {
            "restoration": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=2e-4)
        }

    def train_step(self, batch, *, accumulation_steps=1, update=True):
        degraded, target = self.prepare_batch(batch)
        with self.autocast():
            prediction = self.modules["restoration"](degraded)
            mse = F.mse_loss(prediction, target)
            structural = 1 - ssim(prediction, target, window_size=11, max_val=1.0).mean()
            perceptual = self.perceptual(prediction, target)
            loss = perceptual + mse + 0.4 * structural
        backward_and_step(
            self,
            loss,
            self.optimizers["restoration"],
            accumulation_steps=accumulation_steps,
            update=update,
        )
        return {
            "total": loss.item(),
            "perceptual": perceptual.item(),
            "mse": mse.item(),
            "ssim_loss": structural.item(),
        }

    def _inference_native(self, degraded):
        return self.modules["restoration"](degraded)

    @classmethod
    def provenance(cls):
        return {
            "method": cls.display_name,
            "paper_title": "UWFormer: Underwater Image Enhancement via a Semi-Supervised Multi-Scale Transformer",
            "paper_url": "https://arxiv.org/abs/2310.20210",
            "source_repository_url": "https://github.com/leiyingtie/UWFormer",
            "source_commit_sha": "dde9c688903eb076bf47ff5e6eec4a8c05163e4d",
            "license": "no license file in source repository",
            "implementation_source": "faithful dependency-free PyTorch compatibility port of released wavelet/LPViT/FFC formulation",
            "important_integration_modifications": [
                "built-in Haar DWT replaces pytorch_wavelets",
                "built-in window attention replaces timm helper",
                "common checkpoint adapter",
            ],
            "reproduction_ambiguity": "Released training code is supervised despite the paper's semi-supervised title, so this benchmark retains the released paired objective. The dependency-free low-frequency port preserves windowed transformer processing but is not parameter-for-parameter checkpoint compatible with the released LPViT/NAFNet code.",
        }

    @classmethod
    def training_config(cls):
        return {
            "input_formulation": "RGB [0,1], one-level Haar DWT; low-frequency transformer and high-frequency FFC paths",
            "architecture": "LPViT-compatible window transformer low path plus nine Fourier residual high-path blocks",
            "losses": "VGG16 relu1_2/relu2_2/relu3_3 feature MSE + pixel MSE + 0.4*(1-SSIM)",
            "optimizers": "AdamW lr=2e-4, betas=(0.9,0.999), eps=1e-8",
            "schedulers": "CosineAnnealingLR T_max=100 eta_min=2e-4 (constant by released config)",
            "trainable_initialization": "low/high-frequency restoration paths initialized from scratch",
            "fixed_pretrained_auxiliary_modules": "ImageNet VGG16 perceptual network",
            "native_output": "unbounded residual RGB; evaluator clamps once to [0,1]",
        }
