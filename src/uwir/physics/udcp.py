"""
physics.py
----------
Physics-guided channel extraction for underwater image restoration.

Implements the Underwater Dark Channel Prior (UDCP) pipeline to estimate
two spatially-varying maps from a raw underwater RGB image:

  t(x)  – transmission map  ∈ [0.1, 1]   (how much light reaches sensor)
  B_map – background light  (scalar per image, broadcast to H×W)

These are concatenated to the RGB tensor to form the 5-channel input
[R, G, B, t(x), B_map] fed into UNet5ch.

References
----------
- He et al. (2011) "Single Image Haze Removal Using Dark Channel Prior"
- Drews et al. (2013) "Transmission Estimation in Underwater Single Images"
- He et al. (2013) "Guided Image Filtering"
"""

import cv2
import numpy as np
from scipy.ndimage import minimum_filter

from .features import (
    PhysicsConfig,
    PhysicsFeatures,
    scaled_odd_window,
    scaled_spatial_parameter,
)

# ---------------------------------------------------------------------------
# Background Light Estimation
# ---------------------------------------------------------------------------


def estimate_background_light(
    image_np: np.ndarray,
    percentile: float = 0.1,
) -> np.ndarray:
    """
    Estimate global background light B via the UDCP strategy.

    In underwater imagery the red channel attenuates fastest, so only the
    green and blue channels are used to locate the dark-channel candidates.

    Args:
        image_np  (ndarray): (H, W, 3) float32 RGB in [0, 1].
        percentile (float):  Top-% brightest dark-channel pixels used as
                             candidates. Default: 0.1.

    Returns:
        ndarray: shape (3,) float32 – estimated background light.
    """
    # Use only G & B since red attenuates fastest underwater
    dark_gb = np.min(image_np[:, :, 1:], axis=2)  # (H, W)

    n_pixels = dark_gb.size
    n_top = max(1, int(n_pixels * percentile / 100.0))
    flat_idx = np.argsort(dark_gb.flatten())[-n_top:]
    h_idx, w_idx = np.unravel_index(flat_idx, dark_gb.shape)

    # Among candidates, pick the brightest overall pixel
    candidate_intensity = np.mean(image_np[h_idx, w_idx, :], axis=1)
    best = np.argmax(candidate_intensity)
    return image_np[h_idx[best], w_idx[best], :].astype(np.float32)


# ---------------------------------------------------------------------------
# Guided Filter (edge-preserving smoothing)
# ---------------------------------------------------------------------------


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Edge-preserving guided image filter (He et al., 2013).

    Args:
        guide  (ndarray): (H, W) float32 guidance image.
        src    (ndarray): (H, W) float32 input to be filtered.
        radius (int):     Filter radius. Default: 15.
        eps    (float):   Regularisation constant. Default: 1e-3.

    Returns:
        ndarray: (H, W) float32 filtered output.
    """
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(img: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(img, -1, ksize)

    N = box(np.ones_like(guide))
    mI = box(guide) / N
    mp = box(src) / N
    mIp = box(guide * src) / N
    covIp = mIp - mI * mp

    mII = box(guide * guide) / N
    varI = mII - mI * mI

    a = covIp / (varI + eps)
    b = mp - a * mI

    ma = box(a) / N
    mb = box(b) / N
    return (ma * guide + mb).astype(np.float32)


# ---------------------------------------------------------------------------
# Transmission Map Estimation
# ---------------------------------------------------------------------------


def estimate_transmission_udcp(
    image_np: np.ndarray,
    B: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """
    Estimate spatially-varying transmission map t(x) ∈ [0.1, 1].

    Args:
        image_np   (ndarray): (H, W, 3) float32 RGB in [0, 1].
        B          (ndarray): (3,) float32 background light estimate.
        omega      (float):   Controls how much haze to remove. Default: 0.95.
        patch_size (int):     Dark-channel minimum-filter patch size. Default: 15.

    Returns:
        ndarray: (H, W) float32 transmission map clipped to [0.1, 1].
    """
    B_safe = np.maximum(B, 1e-6)
    normalized = np.clip(image_np / B_safe, 0.0, 1.0)

    # UDCP: dark channel over green & blue only
    dark = np.min(normalized[:, :, 1:], axis=2)
    dark_ch = minimum_filter(dark, size=patch_size)

    t_rough = np.clip(1.0 - omega * dark_ch, 0.1, 1.0)

    # Edge-preserving refinement via guided filter
    guide = np.mean(image_np, axis=2).astype(np.float32)
    t_ref = _guided_filter(guide, t_rough, radius=15, eps=1e-3)
    return np.clip(t_ref, 0.1, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_physics_maps(
    image_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the two physics-guided extra channels from a float32 RGB image.

    This is the single call used inside ``EUVPDataset.__getitem__`` when
    ``USE_PHYSICS=True``.  The result is converted to 1-channel tensors
    and concatenated onto the RGB tensor to form the 5-channel model input.

    Args:
        image_np (ndarray): (H, W, 3) float32 RGB image in [0, 1].

    Returns:
        t_map (ndarray): (H, W) float32 – per-pixel transmission map.
        b_map (ndarray): (H, W) float32 – spatially broadcast scalar
                         background light (mean of B across channels).

    Example::

        t_map, b_map = compute_physics_maps(img_np)
        t_t = torch.from_numpy(t_map).unsqueeze(0)  # (1, H, W)
        b_t = torch.from_numpy(b_map).unsqueeze(0)  # (1, H, W)
        inp_5ch = torch.cat([rgb_tensor, t_t, b_t], dim=0)  # (5, H, W)
    """
    B = estimate_background_light(image_np)
    t_map = estimate_transmission_udcp(image_np, B)
    b_map = np.full(t_map.shape, float(np.mean(B)), dtype=np.float32)
    return t_map, b_map


def estimate_background_light_robust(
    image_np: np.ndarray,
    config: PhysicsConfig | None = None,
) -> np.ndarray:
    """Estimate RGB background light using robust aggregation of UDCP candidates."""
    config = config or PhysicsConfig()
    image = np.asarray(image_np, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
    if image.size and image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)

    dark_gb = np.min(image[:, :, 1:], axis=2).reshape(-1)
    requested = int(dark_gb.size * config.background_percentile / 100.0)
    count = min(dark_gb.size, max(config.min_background_candidates, requested, 1))
    indices = np.argpartition(dark_gb, -count)[-count:]
    candidates = image.reshape(-1, 3)[indices]

    trim = int(count * config.background_trim_fraction)
    if trim * 2 >= count:
        trim = 0
    ordered = np.sort(candidates, axis=0)
    selected = ordered[trim : count - trim] if trim else ordered
    return np.mean(selected, axis=0, dtype=np.float64).astype(np.float32)


def compute_physics_features(
    image_np: np.ndarray,
    config: PhysicsConfig | None = None,
) -> PhysicsFeatures:
    """Compute robust, color-preserving UDCP features for fusion models."""
    config = config or PhysicsConfig()
    image = np.asarray(image_np, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
    if image.size and image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    patch_size = scaled_odd_window(
        config.patch_size, image.shape[:2], config.reference_size
    )
    radius = scaled_spatial_parameter(
        config.guided_filter_radius, image.shape[:2], config.reference_size
    )
    background = estimate_background_light_robust(image, config)
    safe_background = np.maximum(background, 1e-6)
    normalized = np.clip(image / safe_background.reshape(1, 1, 3), 0.0, 1.0)
    dark = np.min(normalized[:, :, 1:], axis=2)
    dark_channel = minimum_filter(dark, size=patch_size, mode="reflect")
    rough = np.clip(
        1.0 - config.omega * dark_channel,
        config.min_transmission,
        1.0,
    )
    guide = np.mean(image, axis=2).astype(np.float32)
    transmission = _guided_filter(
        guide,
        rough,
        radius=radius,
        eps=config.guided_filter_eps,
    )
    transmission = np.clip(
        transmission, config.min_transmission, 1.0
    ).astype(np.float32)
    veiling = (
        (1.0 - transmission[..., None]) * background.reshape(1, 1, 3)
    ).astype(np.float32)
    return PhysicsFeatures(transmission, background, veiling)
