"""Structured physics features used by physics-fusion restoration models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for robust UDCP feature extraction.

    Spatial parameters are specified for a 256-pixel short image side and are
    scaled for native-resolution extraction.
    """

    patch_size: int = 15
    guided_filter_radius: int = 15
    guided_filter_eps: float = 1e-3
    min_transmission: float = 0.1
    omega: float = 0.95
    background_percentile: float = 0.1
    background_trim_fraction: float = 0.1
    min_background_candidates: int = 16
    reference_size: int = 256


@dataclass(frozen=True)
class PhysicsFeatures:
    """Color-preserving physical priors for one RGB image."""

    transmission: np.ndarray
    background_rgb: np.ndarray
    veiling_rgb: np.ndarray

    def as_7ch_maps(self) -> np.ndarray:
        """Return ``[t, V_r, V_g, V_b]`` as an HxWx4 array."""
        return np.concatenate([self.transmission[..., None], self.veiling_rgb], axis=2)


def scaled_spatial_parameter(value: int, shape: tuple[int, int], reference_size: int) -> int:
    """Scale a reference-size radius/window to a native image resolution."""
    scale = min(shape) / float(reference_size)
    return max(1, int(round(value * scale)))


def scaled_odd_window(value: int, shape: tuple[int, int], reference_size: int) -> int:
    """Scale a window and force an odd positive size."""
    result = scaled_spatial_parameter(value, shape, reference_size)
    return result if result % 2 else result + 1
