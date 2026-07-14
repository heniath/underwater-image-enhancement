"""Physics-gated U-Net variants for color-preserving underwater restoration."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from .blocks import DecoderBlock
from .context_unet import ResidualASPP, _group_count
from .unet import DoubleConv, Down, Up


class PhysicsPyramid(nn.Module):
    """Small convolutional pyramid for ``[t, V_r, V_g, V_b]``."""

    def __init__(self, widths: tuple[int, ...], in_channels: int = 4):
        super().__init__()
        blocks = []
        previous = in_channels
        for index, width in enumerate(widths):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, 3, stride=1 if index == 0 else 2, padding=1),
                    nn.GroupNorm(_group_count(width), width),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GroupNorm(_group_count(width), width),
                    nn.GELU(),
                )
            )
            previous = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs


class ZeroGatedFusion(nn.Module):
    """Inject physics features through a channel gate initialized to zero."""

    def __init__(self, image_channels: int, physics_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(physics_channels, image_channels, 1, bias=False),
            nn.GroupNorm(_group_count(image_channels), image_channels),
            nn.GELU(),
        )
        self.scale = nn.Parameter(torch.zeros(image_channels))

    def forward(self, image: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        injected = self.project(physics)
        return image + injected * torch.tanh(self.scale).view(1, -1, 1, 1)


class IdentityResidualHead(nn.Module):
    """Predict a residual in logit space and initialize to the RGB identity."""

    def __init__(self, in_channels: int, epsilon: float = 1e-4):
        super().__init__()
        self.delta = nn.Conv2d(in_channels, 3, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.epsilon = epsilon

    def forward(self, features: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        base = rgb.clamp(self.epsilon, 1.0 - self.epsilon)
        logits = torch.logit(base)
        return torch.sigmoid(logits + self.delta(features))


class FusionUNet(nn.Module):
    """Standard U-Net with a separate, rejectable physics feature path."""

    def __init__(self, use_aspp: bool = False):
        super().__init__()
        f = (64, 128, 256, 512)
        self.enc1 = DoubleConv(3, f[0])
        self.enc2 = Down(f[0], f[1])
        self.enc3 = Down(f[1], f[2])
        self.enc4 = Down(f[2], f[3])
        self.bottleneck = Down(f[3], f[3] * 2)
        self.physics = PhysicsPyramid((16, 32, 64, 128, 256))
        self.fusions = nn.ModuleList(
            [
                ZeroGatedFusion(64, 16),
                ZeroGatedFusion(128, 32),
                ZeroGatedFusion(256, 64),
                ZeroGatedFusion(512, 128),
                ZeroGatedFusion(1024, 256),
            ]
        )
        self.context = ResidualASPP(1024) if use_aspp else nn.Identity()
        self.dec4 = Up(1024, 512, 512)
        self.dec3 = Up(512, 256, 256)
        self.dec2 = Up(256, 128, 128)
        self.dec1 = Up(128, 64, 64)
        self.head = IdentityResidualHead(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 7:
            raise ValueError(f"FusionUNet expects 7 channels, got {x.shape[1]}")
        rgb, priors = x[:, :3], x[:, 3:]
        p = self.physics(priors)
        e1 = self.fusions[0](self.enc1(rgb), p[0])
        e2 = self.fusions[1](self.enc2(e1), p[1])
        e3 = self.fusions[2](self.enc3(e2), p[2])
        e4 = self.fusions[3](self.enc4(e3), p[3])
        bn = self.context(self.fusions[4](self.bottleneck(e4), p[4]))
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1, rgb)


class ASPPFusionUNet(FusionUNet):
    def __init__(self):
        super().__init__(use_aspp=True)


class DenseASPPFusionUNet(nn.Module):
    """DenseNet-121 RGB encoder with gated physics and ASPP context."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights, memory_efficient=True).features
        self.full_stem = DoubleConv(3, 32)
        self.enc0 = nn.Sequential(backbone.conv0, backbone.norm0, backbone.relu0)
        self.enc1 = nn.Sequential(backbone.pool0, backbone.denseblock1)
        self.enc2 = nn.Sequential(backbone.transition1, backbone.denseblock2)
        self.enc3 = nn.Sequential(backbone.transition2, backbone.denseblock3)
        self.enc4 = nn.Sequential(
            backbone.transition3,
            backbone.denseblock4,
            backbone.norm5,
            nn.ReLU(inplace=True),
        )
        self.physics = PhysicsPyramid((16, 32, 64, 128, 256, 256))
        self.fusions = nn.ModuleList(
            [
                ZeroGatedFusion(32, 16),
                ZeroGatedFusion(64, 32),
                ZeroGatedFusion(256, 64),
                ZeroGatedFusion(512, 128),
                ZeroGatedFusion(1024, 256),
                ZeroGatedFusion(1024, 256),
            ]
        )
        self.context = ResidualASPP(1024)
        self.dec4 = DecoderBlock(1024, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 32, 32)
        self.head = IdentityResidualHead(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 7:
            raise ValueError(f"DenseASPPFusionUNet expects 7 channels, got {x.shape[1]}")
        rgb, priors = x[:, :3], x[:, 3:]
        p = self.physics(priors)
        full = self.fusions[0](self.full_stem(rgb), p[0])
        e0 = self.fusions[1](self.enc0(rgb), p[1])
        e1 = self.fusions[2](self.enc1(e0), p[2])
        e2 = self.fusions[3](self.enc2(e1), p[3])
        e3 = self.fusions[4](self.enc3(e2), p[4])
        bn = self.context(self.fusions[5](self.enc4(e3), p[5]))
        d4 = self.dec4(bn, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1, full)
        return self.head(d0, rgb)
