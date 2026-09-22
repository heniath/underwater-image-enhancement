import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from uwir.datasets.lsui import prepare_lsui_splits
from uwir.datasets.uieb import prepare_uieb_splits
from uwir.evaluation.quality_metrics import evaluate_pair
from uwir.reference_methods.funie_gan import FUnIEDiscriminator, FUnIEGenerator
from uwir.reference_methods.ucolor import UColorNet, _gdcp_transmission
from uwir.reference_methods.uwformer import UWFormer
from uwir.reference_methods.waternet import WaterNet, _waternet_inputs
from uwir.training.seed import capture_rng_state, restore_rng_state, seed_everything


def _image(path: Path, value: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def test_uieb_manifest_is_exact_and_model_seed_independent(tmp_path):
    root = tmp_path / "UIEB"
    for index in range(890):
        filename = f"{index:04d}.png"
        _image(root / "raw-890" / filename)
        _image(root / "reference-890" / filename)
    manifest = tmp_path / "uieb.json"
    first = prepare_uieb_splits(root, manifest)
    second = prepare_uieb_splits(root, manifest)
    assert {name: len(records) for name, records in first.items()} == {
        "train": 720,
        "val": 80,
        "test": 90,
    }
    assert [[item.identity for item in first[name]] for name in first] == [
        [item.identity for item in second[name]] for name in second
    ]
    payload = json.loads(manifest.read_text())
    assert payload["split_seed"] == 42


def test_lsui_preserves_official_test_and_exact_stem_pairing(tmp_path):
    root = tmp_path / "LSUI"
    for split, count in (("train", 12), ("test", 3)):
        for index in range(count):
            filename = f"{split}_{index}.png"
            _image(root / split / "input" / filename)
            _image(root / split / "GT" / filename)
    splits = prepare_lsui_splits(root, tmp_path / "lsui.json")
    assert len(splits["test"]) == 3
    assert all(record.split_hint == "test" for record in splits["test"])
    identities = [{record.identity for record in splits[name]} for name in ("train", "val", "test")]
    assert not (
        identities[0] & identities[1]
        or identities[0] & identities[2]
        or identities[1] & identities[2]
    )


def test_lsui_rejects_sorted_zip_pairing(tmp_path):
    root = tmp_path / "LSUI"
    _image(root / "input" / "a.png")
    _image(root / "GT" / "b.png")
    with pytest.raises(ValueError, match="refusing to zip"):
        prepare_lsui_splits(root, tmp_path / "manifest.json")


def test_common_metrics_guard_rgb_range_and_identity():
    image = torch.full((3, 16, 16), 0.5)
    metrics = evaluate_pair(image, image)
    assert np.isinf(metrics["psnr"])
    assert metrics["ssim"] == 1.0
    assert metrics["ciede2000"] == 0.0
    with pytest.raises(ValueError, match="range"):
        evaluate_pair(image * 3, image)


def test_rng_state_round_trip_supports_exact_resume():
    seed_everything(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize("name", ["funie", "water", "ucolor", "uwformer"])
def test_reference_network_paths_are_finite_rgb(name):
    image = torch.rand(1, 3, 32, 32)
    if name == "funie":
        model, args = FUnIEGenerator(), (image.mul(2).sub(1),)
    elif name == "water":
        model, args = WaterNet(), (image, *_waternet_inputs(image))
    elif name == "ucolor":
        model, args = UColorNet(), (image, _gdcp_transmission(image))
    else:
        model, args = UWFormer(), (image,)
    model.eval()
    with torch.no_grad():
        output = model(*args)
    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_funie_discriminator_is_a_training_only_separate_network():
    discriminator = FUnIEDiscriminator()
    result = discriminator(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    assert result.shape[1] == 1


def test_funie_adapter_updates_both_networks_and_round_trips(monkeypatch):
    import uwir.reference_methods.funie_gan as funie_module

    class TinyContent(torch.nn.Module):
        def forward(self, prediction, target):
            return torch.nn.functional.mse_loss(prediction, target)

    monkeypatch.setattr(funie_module, "VGGFeatureLoss", lambda *_args, **_kwargs: TinyContent())
    adapter = funie_module.FUnIEGANAdapter("cpu", amp=False)
    adapter.zero_grad()
    before_generator = next(adapter.modules["generator"].parameters()).detach().clone()
    before_discriminator = next(adapter.modules["discriminator"].parameters()).detach().clone()
    batch = {"degraded": torch.rand(1, 3, 32, 32), "reference": torch.rand(1, 3, 32, 32)}
    adapter.train_step(batch, update=True)
    assert not torch.equal(before_generator, next(adapter.modules["generator"].parameters()))
    assert not torch.equal(
        before_discriminator, next(adapter.modules["discriminator"].parameters())
    )

    state = adapter.state_dict()
    restored = funie_module.FUnIEGANAdapter("cpu", amp=False)
    restored.load_state_dict(state)
    assert state.keys() == restored.state_dict().keys()
    assert set(state["optimizers"]) == {"generator", "discriminator"}
