"""
registry.py
-----------
Central model factory.

Usage
-----
    from net.registry import build_model, parse_model_variant

    # Instantiate any supported model by name:
    model = build_model("resnet_5ch", pretrained_backbone=True)

    # Inspect what the name implies:
    backbone, in_channels, physics_mode = parse_model_variant("mobilenet_4ch_t")
    # → ("mobilenet", 4, "t")

Naming convention
-----------------
    <backbone>_<variant>

   backbone : unet | asppunet | mambabottleneck | mambaaspp | resnet |
              mobilenet | mbconv | mambavision | mambaunet
  variant  : 3ch | 4ch_t | 4ch_b | 5ch

Physics modes
-------------
  3ch   → physics_mode = "none"  →  input: [R, G, B]
  4ch_t → physics_mode = "t"     →  input: [R, G, B, t(x)]
  4ch_b → physics_mode = "b"     →  input: [R, G, B, B_map]
  5ch   → physics_mode = "tb"    →  input: [R, G, B, t(x), B_map]
"""

from typing import NamedTuple

import torch.nn as nn

# ---------------------------------------------------------------------------
# Supported names
# ---------------------------------------------------------------------------

_BACKBONES = (
    "unet",
    "asppunet",
    "mambabottleneck",
    "mambaaspp",
    "resnet",
    "mobilenet",
    "mbconv",
    "mambavision",
    "mambaunet",
)

_FUSION_BACKBONES = ("fusionunet", "asppfusion", "denseasppfusion")

_VARIANTS = {
    "3ch": (3, "none"),
    "4ch_t": (4, "t"),
    "4ch_b": (4, "b"),
    "5ch": (5, "tb"),
}

_FUSION_VARIANT = "7ch_tv"

_HYBRID_VARIANTS = {
    "hybridmamba_core_3ch": ("hybridmamba_core", 3, "none"),
    "hybridmamba_local_3ch": ("hybridmamba_local", 3, "none"),
    "hybridmamba_attn_3ch": ("hybridmamba_attn", 3, "none"),
    "hybridmambafusion_7ch_tv": ("hybridmambafusion", 7, "tv"),
}


class ModelSpec(NamedTuple):
    """Parsed model metadata; tuple-compatible for legacy callers."""

    backbone: str
    in_channels: int
    physics_mode: str


# All valid model names (used to populate argparse choices)
ALL_MODEL_NAMES = [f"{backbone}_{variant}" for backbone in _BACKBONES for variant in _VARIANTS]
ALL_MODEL_NAMES += [f"{backbone}_{_FUSION_VARIANT}" for backbone in _FUSION_BACKBONES]
ALL_MODEL_NAMES += list(_HYBRID_VARIANTS)


# ---------------------------------------------------------------------------
# Parse helper
# ---------------------------------------------------------------------------


def parse_model_variant(name: str) -> ModelSpec:
    """
    Parse a model name string into its components.

    Args:
        name (str): e.g. ``"resnet_4ch_t"``

    Returns:
        backbone     (str): One of ``"unet"``, ``"resnet"``, ``"mobilenet"``.
        in_channels  (int): Number of input channels (3, 4, or 5).
        physics_mode (str): One of ``"none"``, ``"t"``, ``"b"``, ``"tb"``.

    Raises:
        ValueError: If the name cannot be parsed.
    """
    if name in _HYBRID_VARIANTS:
        return ModelSpec(*_HYBRID_VARIANTS[name])

    for backbone in (*_BACKBONES, *_FUSION_BACKBONES):
        prefix = backbone + "_"
        if name.startswith(prefix):
            variant = name[len(prefix) :]
            if backbone in _FUSION_BACKBONES and variant == _FUSION_VARIANT:
                return ModelSpec(backbone, 7, "tv")
            if backbone in _FUSION_BACKBONES or variant not in _VARIANTS:
                raise ValueError(
                    f"Unknown variant '{variant}' in model name '{name}'. "
                    f"Valid variants: {list(_VARIANTS)}"
                )
            in_channels, physics_mode = _VARIANTS[variant]
            return ModelSpec(backbone, in_channels, physics_mode)

    raise ValueError(
        f"Cannot parse model name '{name}'. "
        f"Expected format: <backbone>_<variant>  "
        f"(backbone ∈ {_BACKBONES}, variant ∈ {list(_VARIANTS)})."
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(name: str, pretrained_backbone: bool = True) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name               (str):  Model name, e.g. ``"resnet_5ch"``.
        pretrained_backbone(bool): Whether to load pretrained ImageNet weights
                                   for ResNet / MobileNet encoders.
                                   Ignored for ``unet`` (no pretrained variant).

    Returns:
        nn.Module: Uninitialised model (not moved to a device).

    Example::

        model = build_model("mobilenet_4ch_t").to("cuda")
    """
    backbone, in_channels, _ = parse_model_variant(name)

    if name in _HYBRID_VARIANTS:
        from .hybrid_mamba_unet import (
            hybridmamba_attn_3ch,
            hybridmamba_core_3ch,
            hybridmamba_local_3ch,
            hybridmambafusion_7ch_tv,
        )

        factories = {
            "hybridmamba_core_3ch": hybridmamba_core_3ch,
            "hybridmamba_local_3ch": hybridmamba_local_3ch,
            "hybridmamba_attn_3ch": hybridmamba_attn_3ch,
            "hybridmambafusion_7ch_tv": hybridmambafusion_7ch_tv,
        }
        return factories[name]()

    if backbone in _FUSION_BACKBONES:
        from .fusion_unet import ASPPFusionUNet, DenseASPPFusionUNet, FusionUNet

        if backbone == "fusionunet":
            return FusionUNet()
        if backbone == "asppfusion":
            return ASPPFusionUNet()
        return DenseASPPFusionUNet(pretrained=pretrained_backbone)

    if backbone == "unet":
        from .unet import UNet5ch

        return UNet5ch(in_channels=in_channels)

    if backbone in {"asppunet", "mambabottleneck", "mambaaspp"}:
        from .context_unet import ASPPUNet, MambaASPPUNet, MambaBottleneckUNet

        model_classes = {
            "asppunet": ASPPUNet,
            "mambabottleneck": MambaBottleneckUNet,
            "mambaaspp": MambaASPPUNet,
        }
        return model_classes[backbone](in_channels=in_channels)

    if backbone == "resnet":
        from .resnet_unet import ResNetUNet

        return ResNetUNet(in_channels=in_channels, pretrained=pretrained_backbone)

    if backbone == "mobilenet":
        from .mobilenet_unet import MobileNetUNet

        return MobileNetUNet(in_channels=in_channels, pretrained=pretrained_backbone)

    if backbone == "mbconv":
        from .mbconv_unet import MBConvUNet

        return MBConvUNet(in_channels=in_channels)

    if backbone == "mambavision":
        from .mambavision_unet import MambaVisionUNet

        return MambaVisionUNet(in_channels=in_channels, pretrained=pretrained_backbone)

    if backbone == "mambaunet":
        from .mamba_unet import MambaUNet

        return MambaUNet(in_channels=in_channels)

    # Should never reach here due to parse_model_variant guard
    raise ValueError(f"Unknown backbone: {backbone}")
