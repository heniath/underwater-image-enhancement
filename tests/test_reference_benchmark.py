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
from uwir.reference_methods import (
    CODE_BACKED_REFERENCE_METHODS,
    REFERENCE_METHODS,
    SUPPLEMENTAL_REIMPLEMENTATIONS,
)
from uwir.reference_methods.base import ReferenceMethodAdapter, backward_and_step
from uwir.reference_methods.funie_gan import FUnIEDiscriminator, FUnIEGenerator
from uwir.reference_methods.lpd_net import LPDNet, _msrcr_prior
from uwir.reference_methods.ucolor import UColorAdapter, UColorNet, _gdcp_transmission
from uwir.reference_methods.uwformer import UWFormer, _FourierResidual
from uwir.reference_methods.waternet import WaterNet, _waternet_inputs
from uwir.training.checkpoint import load_checkpoint, save_checkpoint
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


def test_code_backed_methods_are_prioritized_over_paper_only_reimplementations():
    assert list(CODE_BACKED_REFERENCE_METHODS) == [
        "funie_gan",
        "ucolor",
        "unet",
        "water_net",
        "uwformer",
    ]
    assert list(SUPPLEMENTAL_REIMPLEMENTATIONS) == ["lpd_net"]
    assert list(REFERENCE_METHODS) == [*CODE_BACKED_REFERENCE_METHODS, "lpd_net"]


def test_scheduler_advances_only_after_a_completed_optimizer_update():
    class ScheduledAdapter(ReferenceMethodAdapter):
        def build(self):
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            self.modules = {"restoration": model}
            self.optimizers = {"restoration": optimizer}
            self.schedulers = {
                "restoration": torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
            }

        def train_step(self, batch, *, accumulation_steps=1, update=True):
            raise NotImplementedError

        def _inference_native(self, degraded):
            raise NotImplementedError

        @classmethod
        def provenance(cls):
            return {}

        @classmethod
        def training_config(cls):
            return {}

    adapter = ScheduledAdapter("cpu", amp=False)
    adapter.step_schedulers()
    assert adapter.optimizers["restoration"].param_groups[0]["lr"] == 1.0

    loss = adapter.modules["restoration"](torch.ones(1, 1)).sum()
    backward_and_step(
        adapter,
        loss,
        adapter.optimizers["restoration"],
        accumulation_steps=1,
        update=True,
    )
    adapter.step_schedulers()
    assert adapter.optimizers["restoration"].param_groups[0]["lr"] == 0.5


def test_funie_uses_released_pytorch_topology():
    model = FUnIEGenerator()
    assert model.down1[0].kernel_size == (4, 4)
    assert model.down1[0].stride == (2, 2)
    assert model.down2[0].out_channels == 128
    assert model.down5[0].out_channels == 256
    assert isinstance(model.up1.model[0], torch.nn.ConvTranspose2d)


def test_restore_rng_state_handles_checkpoint_and_device_normalization(tmp_path):
    seed_everything(42)
    state = capture_rng_state()
    ckpt_path = tmp_path / "test_ckpt.pth"
    save_checkpoint(
        ckpt_path,
        {
            "adapter": {},
            "epoch": 1,
            "best_val_psnr": 20.0,
            "best_epoch": 1,
            "history": [],
            "run_config": {},
            "rng_state": state,
        },
    )
    loaded = load_checkpoint(ckpt_path, map_location="cpu")
    restore_rng_state(loaded["rng_state"])
    val1 = torch.rand(1)

    seed_everything(42)
    state2 = capture_rng_state()
    state2["torch_cpu"] = state2["torch_cpu"].clone()
    restore_rng_state(state2)
    val2 = torch.rand(1)
    assert torch.equal(val1, val2)


@pytest.mark.parametrize("name", ["funie", "water", "ucolor", "uwformer", "lpd_net"])
def test_reference_network_paths_are_finite_rgb(name):
    image = torch.rand(1, 3, 32, 32)
    if name == "funie":
        model, args = FUnIEGenerator(), (image.mul(2).sub(1),)
    elif name == "water":
        model, args = WaterNet(), (image, *_waternet_inputs(image))
    elif name == "ucolor":
        model, args = UColorNet(), (image, _gdcp_transmission(image))
    elif name == "lpd_net":
        model, args = LPDNet(), (image, _msrcr_prior(image))
    else:
        model, args = UWFormer(), (image,)
    model.eval()
    with torch.no_grad():
        output = model(*args)
    assert output.shape == image.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("module_name", "adapter_name"),
    [
        ("uwir.reference_methods.funie_gan", "FUnIEGANAdapter"),
        ("uwir.reference_methods.ucolor", "UColorAdapter"),
    ],
)
def test_pooled_reference_adapters_preserve_odd_native_resolution(
    monkeypatch, module_name, adapter_name
):
    module = __import__(module_name, fromlist=[adapter_name])

    class TinyContent(torch.nn.Module):
        def forward(self, prediction, target):
            return torch.nn.functional.mse_loss(prediction, target)

    monkeypatch.setattr(module, "VGGFeatureLoss", lambda *_args, **_kwargs: TinyContent())
    adapter = getattr(module, adapter_name)("cpu", amp=False)
    adapter.eval()
    image = torch.rand(1, 3, 33, 35)

    prediction = adapter.inference(image)

    assert prediction.shape == image.shape
    assert torch.isfinite(prediction).all()


def test_uwformer_fourier_path_keeps_fft_in_float32_under_autocast():
    block = _FourierResidual(4).eval()
    image = torch.rand(1, 4, 15, 20)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = block(image)

    assert prediction.shape == image.shape
    assert torch.isfinite(prediction).all()


def test_ucolor_uses_tiled_native_resolution_inference():
    class IdentityRestoration(torch.nn.Module):
        def forward(self, rgb, transmission):
            assert transmission.shape[1] == 1
            return rgb

    adapter = object.__new__(UColorAdapter)
    adapter.modules = {"restoration": IdentityRestoration()}
    image = torch.rand(1, 3, 257, 259)

    prediction = adapter._inference_native(image)

    assert prediction.shape == image.shape
    assert torch.allclose(prediction, image, atol=1e-6)


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
