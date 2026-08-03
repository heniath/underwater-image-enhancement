"""Model registry for the paper backbone and the retained UW-LYT study."""

from typing import NamedTuple

import torch.nn as nn


class ModelSpec(NamedTuple):
    """Model metadata derived from a registered model name."""

    backbone: str
    in_channels: int
    physics_mode: str


_BACKBONES = ("unet", "uwlyt", "uwlyttiny")
_VARIANTS = {
    "3ch": (3, "none"),
    "4ch_t": (4, "t"),
    "4ch_b": (4, "b"),
    "5ch": (5, "tb"),
}

ALL_MODEL_NAMES = [f"{backbone}_{variant}" for backbone in _BACKBONES for variant in _VARIANTS]


def parse_model_variant(name: str) -> ModelSpec:
    """Parse ``<backbone>_<input variant>`` into its input contract."""
    for backbone in _BACKBONES:
        prefix = f"{backbone}_"
        if name.startswith(prefix):
            variant = name[len(prefix) :]
            if variant not in _VARIANTS:
                break
            in_channels, physics_mode = _VARIANTS[variant]
            return ModelSpec(backbone, in_channels, physics_mode)
    raise ValueError(
        f"Unknown model '{name}'. Expected one of: {', '.join(ALL_MODEL_NAMES)}"
    )


def build_model(name: str, pretrained_backbone: bool = False) -> nn.Module:
    """Build a paper U-Net or retained UW-LYT variant."""
    del pretrained_backbone
    backbone, in_channels, _ = parse_model_variant(name)

    if backbone == "unet":
        from .unet import UNet5ch

        return UNet5ch(in_channels=in_channels)

    from .uwlyt import build_uwlyt

    return build_uwlyt(in_channels, tiny=backbone == "uwlyttiny")
