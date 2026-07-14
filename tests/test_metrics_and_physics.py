import numpy as np
import torch

from uwir.metrics import (
    compute_ciede2000,
    compute_psnr,
    compute_ssim,
    compute_uciqe,
    compute_uiqm,
    tiled_predict,
)
from uwir.physics import PhysicsConfig, compute_physics_features, compute_physics_maps


def test_reference_metrics_on_identical_images():
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)

    assert np.isinf(compute_psnr(image, image))
    assert compute_ssim(image, image) == 1.0
    assert compute_ciede2000(image, image) == 0.0


def test_underwater_metrics_are_finite():
    gradient = np.linspace(0, 1, 32, dtype=np.float32)
    image = np.stack(np.meshgrid(gradient, gradient), axis=-1)
    image = np.concatenate([image, image[..., :1]], axis=-1)

    assert np.isfinite(compute_uciqe(image))
    assert np.isfinite(compute_uiqm(image))


def test_udcp_physics_map_contract():
    image = np.random.default_rng(42).random((32, 32, 3), dtype=np.float32)
    transmission, background = compute_physics_maps(image)

    assert transmission.shape == (32, 32)
    assert background.shape == (32, 32)
    assert np.all(np.isfinite(transmission))
    assert np.all(np.isfinite(background))


def test_structured_physics_is_color_preserving_and_bounded():
    image = np.zeros((48, 80, 3), dtype=np.float32)
    image[..., 0] = 0.2
    image[..., 1] = np.linspace(0.3, 0.9, 80)
    image[..., 2] = 0.8
    features = compute_physics_features(image)

    assert features.transmission.shape == (48, 80)
    assert features.background_rgb.shape == (3,)
    assert features.veiling_rgb.shape == (48, 80, 3)
    assert np.ptp(features.background_rgb) > 0.05
    assert np.all((features.transmission >= 0.1) & (features.transmission <= 1.0))
    assert np.all(np.isfinite(features.as_7ch_maps()))


def test_physics_configuration_changes_guided_result():
    image = np.random.default_rng(4).random((64, 96, 3), dtype=np.float32)
    narrow = compute_physics_features(image, PhysicsConfig(guided_filter_radius=2))
    wide = compute_physics_features(image, PhysicsConfig(guided_filter_radius=20))
    assert not np.allclose(narrow.transmission, wide.transmission)


def test_tiled_predict_matches_direct_path_for_small_images():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x[:, :3]

    image = torch.rand(1, 3, 85, 99)
    output = tiled_predict(Identity(), image, tile_size=64, overlap=8)
    assert output.shape == image.shape
    assert torch.allclose(output, image)
