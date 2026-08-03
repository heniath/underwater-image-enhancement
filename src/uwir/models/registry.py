"""Model registry for the paper backbone and the retained UW-LYT study."""

from typing import NamedTuple

import torch.nn as nn


class ModelSpec(NamedTuple):
    """Model metadata derived from a registered model name."""

    backbone: str
    in_channels: int
    physics_mode: str


_LEGACY_VARIANTS = {
    "3ch": (3, "none"),
    "4ch_t": (4, "t"),
    "4ch_b": (4, "b"),
    "5ch": (5, "tb"),
}
_V2_VARIANTS = {
    "3ch": (3, "none"),
    "4ch_t": (4, "t"),
    "6ch_b": (6, "b_rgb"),
    "7ch": (7, "tb_rgb"),
}
_BACKBONE_VARIANTS = {
    "unet": _LEGACY_VARIANTS,
    "uwlyt": _LEGACY_VARIANTS,
    "uwlyttiny": _LEGACY_VARIANTS,
    "uwlytv2": _V2_VARIANTS,
    "uwlytv2tiny": _V2_VARIANTS,
}

ALL_MODEL_NAMES = [
    f"{backbone}_{variant}"
    for backbone, variants in _BACKBONE_VARIANTS.items()
    for variant in variants
]


def parse_model_variant(name: str) -> ModelSpec:
    """Parse ``<backbone>_<input variant>`` into its input contract."""
    for backbone, variants in _BACKBONE_VARIANTS.items():
        prefix = f"{backbone}_"
        if name.startswith(prefix):
            variant = name[len(prefix) :]
            if variant not in variants:
                break
            in_channels, physics_mode = variants[variant]
            return ModelSpec(backbone, in_channels, physics_mode)
    raise ValueError(f"Unknown model '{name}'. Expected one of: {', '.join(ALL_MODEL_NAMES)}")


def build_model(name: str, pretrained_backbone: bool = False) -> nn.Module:
    """Build a paper U-Net or retained UW-LYT variant."""
    del pretrained_backbone
    backbone, in_channels, physics_schema = parse_model_variant(name)

    if backbone == "unet":
        from .unet import UNet5ch

        return UNet5ch(in_channels=in_channels)

    from .uwlyt import build_uwlyt, build_uwlytv2

    if backbone in ("uwlytv2", "uwlytv2tiny"):
        return build_uwlytv2(physics_schema, tiny=backbone == "uwlytv2tiny")
    return build_uwlyt(in_channels, tiny=backbone == "uwlyttiny")
