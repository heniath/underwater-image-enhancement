"""Model architectures and registry with lazy architecture imports."""

from .registry import ALL_MODEL_NAMES, ModelSpec, build_model, parse_model_variant

__all__ = [
    "ALL_MODEL_NAMES",
    "ASPPUNet",
    "ASPPFusionUNet",
    "ContextUNet",
    "MambaASPPUNet",
    "MambaBottleneckUNet",
    "MambaUNet",
    "MambaVisionUNet",
    "MBConvUNet",
    "PCFMBConvUNet",
    "PCFTinyUNet",
    "HybridMambaUNet",
    "MobileNetUNet",
    "FusionUNet",
    "DenseASPPFusionUNet",
    "ModelSpec",
    "ResNetUNet",
    "M20566RepLight",
    "M20566LMF",
    "M20566LCS",
    "UNet5ch",
    "build_model",
    "parse_model_variant",
]


_ARCHITECTURES = {
    "M20566RepLight": ("m20566_replight", "M20566RepLight"),
    "M20566LMF": ("m20566_replight", "M20566LMF"),
    "M20566LCS": ("m20566_replight", "M20566LCS"),
    "ASPPUNet": ("context_unet", "ASPPUNet"),
    "ContextUNet": ("context_unet", "ContextUNet"),
    "MambaASPPUNet": ("context_unet", "MambaASPPUNet"),
    "MambaBottleneckUNet": ("context_unet", "MambaBottleneckUNet"),
    "MambaUNet": ("mamba_unet", "MambaUNet"),
    "MambaVisionUNet": ("mambavision_unet", "MambaVisionUNet"),
    "MBConvUNet": ("mbconv_unet", "MBConvUNet"),
    "PCFMBConvUNet": ("pcf_mbconv_unet", "PCFMBConvUNet"),
    "PCFTinyUNet": ("pcf_tiny_unet", "PCFTinyUNet"),
    "HybridMambaUNet": ("hybrid_mamba_unet", "HybridMambaUNet"),
    "MobileNetUNet": ("mobilenet_unet", "MobileNetUNet"),
    "FusionUNet": ("fusion_unet", "FusionUNet"),
    "ASPPFusionUNet": ("fusion_unet", "ASPPFusionUNet"),
    "DenseASPPFusionUNet": ("fusion_unet", "DenseASPPFusionUNet"),
    "ResNetUNet": ("resnet_unet", "ResNetUNet"),
    "UNet5ch": ("unet", "UNet5ch"),
}


def __getattr__(name):
    if name in _ARCHITECTURES:
        from importlib import import_module

        module_name, attribute = _ARCHITECTURES[name]
        return getattr(import_module(f"{__name__}.{module_name}"), attribute)
    raise AttributeError(name)
