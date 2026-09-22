"""Method-preserving adapters used by the UIEB/LSUI reference benchmark."""

from .base import ReferenceMethodAdapter
from .funie_gan import FUnIEGANAdapter
from .lpd_net import LPDNetAdapter
from .ucolor import UColorAdapter
from .unet import UNetAdapter
from .uwformer import UWFormerAdapter
from .waternet import WaterNetAdapter

# Default execution order is evidence-based: released/source-backed methods run
# before the paper-only reconstruction. Dict order is intentionally public here
# because the benchmark CLI uses it as its default queue.
CODE_BACKED_REFERENCE_METHODS = {
    "funie_gan": FUnIEGANAdapter,
    "ucolor": UColorAdapter,
    "unet": UNetAdapter,
    "water_net": WaterNetAdapter,
    "uwformer": UWFormerAdapter,
}
SUPPLEMENTAL_REIMPLEMENTATIONS = {"lpd_net": LPDNetAdapter}
REFERENCE_METHODS = {**CODE_BACKED_REFERENCE_METHODS, **SUPPLEMENTAL_REIMPLEMENTATIONS}

__all__ = [
    "CODE_BACKED_REFERENCE_METHODS",
    "REFERENCE_METHODS",
    "SUPPLEMENTAL_REIMPLEMENTATIONS",
    "ReferenceMethodAdapter",
]
