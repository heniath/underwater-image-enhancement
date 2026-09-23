"""Latent and physically parameterized wavelength-aware U-Net front-ends."""

from typing import NamedTuple

import torch
import torch.nn as nn

from .unet import UNet5ch


class PhysicsOutput(NamedTuple):
    """Intermediate outputs required by the physics-consistency loss."""

    enhanced: torch.Tensor
    reconstructed: torch.Tensor
    transmission: torch.Tensor
    background: torch.Tensor


class ParameterizedPhysicsOutput(NamedTuple):
    """Outputs of the Beer-Lambert-constrained physics model."""

    enhanced: torch.Tensor
    reconstructed: torch.Tensor
    transmission: torch.Tensor
    background: torch.Tensor
    depth: torch.Tensor
    attenuation: torch.Tensor


class _TransmissionExtractor(nn.Module):
    """Estimate a dense transmission map for every RGB wavelength band."""

    def __init__(self, width: int = 32, minimum: float = 0.1):
        super().__init__()
        self.minimum = minimum
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        unit_transmission = torch.sigmoid(self.net(image))
        return self.minimum + (1.0 - self.minimum) * unit_transmission


class _BackgroundLightExtractor(nn.Module):
    """Estimate one global background-light value for each RGB wavelength band."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width * 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class LearnableLatentUNet(nn.Module):
    """Learn unconstrained wavelength-shaped latent maps before a U-Net.

    This class deliberately makes no claim that its directly predicted maps
    are physical transmission or background-light estimates. It is retained
    as the non-physical ablation and for checkpoint compatibility.

    The underwater image formation equation is used only for an optional
    reconstruction objective:

    ``I_lambda = J_lambda * T_lambda + B_lambda * (1 - T_lambda)``.

    Normal inference returns only ``J``. Training can request all intermediate
    tensors with ``return_physics=True`` for the reconstruction loss.
    """

    supports_physics_loss = True

    def __init__(
        self,
        extractor_width: int = 32,
        transmission_minimum: float = 0.1,
        enhancer: nn.Module | None = None,
    ):
        super().__init__()
        self.transmission_extractor = _TransmissionExtractor(
            width=extractor_width,
            minimum=transmission_minimum,
        )
        self.background_extractor = _BackgroundLightExtractor(width=extractor_width)
        self.enhancer = enhancer if enhancer is not None else UNet5ch(in_channels=9)

    def forward(
        self,
        image: torch.Tensor,
        return_physics: bool = False,
    ) -> torch.Tensor | PhysicsOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"LearnableLatentUNet expects NCHW RGB input; received shape {tuple(image.shape)}"
            )

        transmission = self.transmission_extractor(image)
        global_background = self.background_extractor(image)
        background = global_background.expand_as(image)
        enhanced = self.enhancer(torch.cat((image, transmission, background), dim=1))

        if not return_physics:
            return enhanced

        reconstructed = enhanced * transmission + background * (1.0 - transmission)
        return PhysicsOutput(enhanced, reconstructed, transmission, background)


# Backwards-compatible name for checkpoints produced before the latent maps
# were named more precisely.
LearnablePhysicsUNet = LearnableLatentUNet


class _DepthExtractor(nn.Module):
    """Estimate a shared, normalized optical-depth field in ``[0, 1]``."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class _WaterParameterExtractor(nn.Module):
    """Estimate global ordered attenuation and RGB background light."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width * 2, 6, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_attenuation, raw_background = self.encoder(image).chunk(2, dim=1)

        # Parameterize positive ordered coefficients so red attenuates at least
        # as strongly as green, and green at least as strongly as blue.
        blue = 0.1 + 0.9 * torch.sigmoid(raw_attenuation[:, 0:1])
        green = blue + 0.1 + 0.9 * torch.sigmoid(raw_attenuation[:, 1:2])
        red = green + 0.1 + 0.9 * torch.sigmoid(raw_attenuation[:, 2:3])
        attenuation = torch.cat((red, green, blue), dim=1)
        background = torch.sigmoid(raw_background)
        return attenuation, background


class ParameterizedPhysicsUNet(nn.Module):
    """Derive RGB transmission from depth and wavelength attenuation.

    ``T_lambda(x) = exp(-beta_lambda * d(x))`` couples all colour channels
    through one normalized optical-depth field. The ordering
    ``beta_R >= beta_G >= beta_B`` is guaranteed by construction.
    """

    supports_physics_loss = True

    def __init__(self, extractor_width: int = 32, enhancer: nn.Module | None = None):
        super().__init__()
        self.depth_extractor = _DepthExtractor(width=extractor_width)
        self.water_extractor = _WaterParameterExtractor(width=extractor_width)
        self.enhancer = enhancer if enhancer is not None else UNet5ch(in_channels=9)

    def forward(
        self,
        image: torch.Tensor,
        return_physics: bool = False,
    ) -> torch.Tensor | ParameterizedPhysicsOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "ParameterizedPhysicsUNet expects NCHW RGB input; "
                f"received shape {tuple(image.shape)}"
            )

        depth = self.depth_extractor(image)
        attenuation, global_background = self.water_extractor(image)
        transmission = torch.exp(-attenuation * depth)
        background = global_background.expand_as(image)
        enhanced = self.enhancer(torch.cat((image, transmission, background), dim=1))

        if not return_physics:
            return enhanced

        reconstructed = enhanced * transmission + background * (1.0 - transmission)
        return ParameterizedPhysicsOutput(
            enhanced,
            reconstructed,
            transmission,
            background,
            depth,
            attenuation,
        )
