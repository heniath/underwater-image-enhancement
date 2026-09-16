from pathlib import Path

import numpy as np
import pytest
import torch

from uwir.cli.train import _add_physics_channels, load_ckpt, save_ckpt
from uwir.models import ALL_MODEL_NAMES, ModelSpec, build_model, parse_model_variant
from uwir.models.uwlyt import UWLYTV2, LowResolutionAttention
from uwir.physics import compute_physics_maps, compute_physics_maps_rgb

V2_VARIANTS = (
    ("uwlytv2_3ch", 3, "none", 35_000),
    ("uwlytv2_4ch_t", 4, "t", 40_000),
    ("uwlytv2_6ch_b", 6, "b_rgb", 40_000),
    ("uwlytv2_7ch", 7, "tb_rgb", 40_000),
    ("uwlytv2tiny_3ch", 3, "none", 15_000),
    ("uwlytv2tiny_4ch_t", 4, "t", 18_000),
    ("uwlytv2tiny_6ch_b", 6, "b_rgb", 18_000),
    ("uwlytv2tiny_7ch", 7, "tb_rgb", 18_000),
)


@pytest.mark.parametrize(("name", "channels", "schema", "cap"), V2_VARIANTS)
def test_v2_registry_forward_identity_and_parameter_ceiling(name, channels, schema, cap):
    spec = parse_model_variant(name)
    assert spec == ModelSpec(name.split("_")[0], channels, schema)
    model = build_model(name).eval()
    inputs = torch.rand(1, channels, 19, 23)
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (1, 3, 19, 23)
    assert torch.isfinite(output).all()
    assert torch.equal(output, inputs[:, :3])
    assert sum(parameter.numel() for parameter in model.parameters()) <= cap


@pytest.mark.parametrize("width, expected", [(24, (9, 15)), (40, (15, 25))])
def test_v2_fixed_asymmetric_chrominance_budget(width, expected):
    model = UWLYTV2(width=width)
    assert (model.cb_channels, model.cr_channels) == expected
    assert model.cb_channels + model.cr_channels == width
    assert sum(isinstance(module, LowResolutionAttention) for module in model.modules()) == 1
    assert model.fusion[0].in_channels == width * 2


def test_v2_physics_collation_preserves_rgb_background_without_averaging():
    rgb = torch.rand(3, 7, 9)
    transmission = np.full((7, 9), 0.4, dtype=np.float32)
    background = np.array([0.1, 0.35, 0.9], dtype=np.float32)

    def extractor(_image):
        return transmission, background

    b_input = _add_physics_channels(rgb, "b_rgb", extractor)
    tb_input = _add_physics_channels(rgb, "tb_rgb", extractor)
    assert b_input.shape == (6, 7, 9)
    assert tb_input.shape == (7, 7, 9)
    assert torch.equal(tb_input[3], torch.full((7, 9), 0.4))
    for channel, expected in zip(b_input[3:], background, strict=True):
        assert torch.allclose(channel, torch.full((7, 9), float(expected)))


def test_rgb_physics_api_complements_unchanged_legacy_contract():
    image = np.random.default_rng(7).random((17, 21, 3), dtype=np.float32)
    legacy_t, legacy_b = compute_physics_maps(image)
    rgb_t, background_rgb = compute_physics_maps_rgb(image)
    assert legacy_t.shape == rgb_t.shape == (17, 21)
    assert background_rgb.shape == (3,)
    assert np.allclose(legacy_t, rgb_t)
    assert np.allclose(legacy_b, background_rgb.mean())
    assert not np.shares_memory(background_rgb, legacy_b)


def test_every_legacy_registry_name_and_scalar_background_schema_remains_available():
    expected = {
        f"{family}_{variant}"
        for family in ("unet", "uwlyt", "uwlyttiny")
        for variant in ("3ch", "4ch_t", "4ch_b", "5ch")
    }
    assert expected.issubset(ALL_MODEL_NAMES)
    for name in expected:
        spec = parse_model_variant(name)
        assert spec.physics_mode in ("none", "t", "b", "tb")
    assert parse_model_variant("uwlyt_4ch_b") == ModelSpec("uwlyt", 4, "b")
    assert parse_model_variant("uwlyt_5ch") == ModelSpec("uwlyt", 5, "tb")


def test_v2_gradients_reach_every_trainable_path_after_nonzero_test_initialization():
    model = UWLYTV2("tb_rgb", width=24)
    with torch.no_grad():
        model.residual_head.weight.fill_(1e-3)
        model.transmission_gate.project_out.weight.fill_(1e-2)
        model.background_conditioner.project_out.weight.fill_(1e-2)
    inputs = torch.rand(2, 7, 17, 21)
    model(inputs).mean().backward()

    for branch_name in (
        "cb_path",
        "cr_path",
        "transmission_gate",
        "background_conditioner",
        "fusion",
        "residual_head",
    ):
        branch = getattr(model, branch_name)
        gradients = [p.grad for p in branch.parameters() if p.requires_grad]
        assert gradients and any(g is not None and torch.count_nonzero(g) for g in gradients), (
            branch_name
        )
        assert all(g is None or torch.isfinite(g).all() for g in gradients)


def test_v2_noop_conditioner_initialization_and_schema_failures():
    model = UWLYTV2("tb_rgb")
    assert torch.count_nonzero(model.transmission_gate.project_out.weight) == 0
    assert torch.count_nonzero(model.background_conditioner.project_out.weight) == 0
    assert torch.count_nonzero(model.residual_head.weight) == 0
    with pytest.raises(ValueError, match="expects NCHW.*7 channels"):
        model(torch.rand(1, 5, 16, 16))
    with pytest.raises(ValueError, match="unknown UWLYTV2 physics schema"):
        UWLYTV2("b")
    with pytest.raises(ValueError, match="requires RGB background light"):
        _add_physics_channels(
            torch.rand(3, 8, 8),
            "b_rgb",
            lambda _image: (
                np.ones((8, 8), dtype=np.float32),
                np.ones((8, 8), dtype=np.float32),
            ),
        )
    with pytest.raises(ValueError, match="Unknown model"):
        parse_model_variant("uwlytv2_5ch")


def test_v2_checkpoint_round_trip(tmp_path: Path):
    model = build_model("uwlytv2tiny_7ch")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        model.residual_head.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    path = tmp_path / "v2" / "best_model.pth"
    save_ckpt(model, optimizer, 4, {"psnr": 22.5}, str(path))
    restored = build_model("uwlytv2tiny_7ch")
    epoch, metrics = load_ckpt(str(path), restored)
    assert (epoch, metrics) == (4, {"psnr": 22.5})
    assert torch.equal(restored.residual_head.bias, model.residual_head.bias)
