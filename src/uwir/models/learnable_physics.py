"""End-to-end wavelength-aware physics extraction and image enhancement."""

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


class LearnablePhysicsUNet(nn.Module):
    """Learn wavelength-aware ``T`` and ``B`` before enhancing with U-Net.

    The underwater image formation model is applied channel-wise:

    ``I_lambda = J_lambda * T_lambda + B_lambda * (1 - T_lambda)``.

    Normal inference returns only ``J``. Training can request all intermediate
    tensors with ``return_physics=True`` for the reconstruction loss.
    """

    supports_physics_loss = True

    def __init__(self, extractor_width: int = 32, transmission_minimum: float = 0.1):
        super().__init__()
        self.transmission_extractor = _TransmissionExtractor(
            width=extractor_width,
            minimum=transmission_minimum,
        )
        self.background_extractor = _BackgroundLightExtractor(width=extractor_width)
        self.enhancer = UNet5ch(in_channels=9)

    def forward(
        self,
        image: torch.Tensor,
        return_physics: bool = False,
    ) -> torch.Tensor | PhysicsOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"LearnablePhysicsUNet expects NCHW RGB input; received shape {tuple(image.shape)}"
            )

        transmission = self.transmission_extractor(image)
        global_background = self.background_extractor(image)
        background = global_background.expand_as(image)
        enhanced = self.enhancer(torch.cat((image, transmission, background), dim=1))

        if not return_physics:
            return enhanced

        reconstructed = enhanced * transmission + background * (1.0 - transmission)
        return PhysicsOutput(enhanced, reconstructed, transmission, background)
