"""Method-preserving adapters used by the UIEB/LSUI reference benchmark."""

from .base import ReferenceMethodAdapter
from .funie_gan import FUnIEGANAdapter
from .ucolor import UColorAdapter
from .unet import UNetAdapter
from .uwformer import UWFormerAdapter
from .waternet import WaterNetAdapter

REFERENCE_METHODS = {
    "funie_gan": FUnIEGANAdapter,
    "ucolor": UColorAdapter,
    "unet": UNetAdapter,
    "water_net": WaterNetAdapter,
    "uwformer": UWFormerAdapter,
}

__all__ = ["REFERENCE_METHODS", "ReferenceMethodAdapter"]
