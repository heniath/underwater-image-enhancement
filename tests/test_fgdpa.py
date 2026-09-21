"""
test_fgdpa.py
-------------
Unit and integration tests for FGDPA (Real-Time Underwater Image Enhancement
via Frequency-Guided Dual-Path Attention - ICME 2026).
"""

import pytest
import torch
import torch.nn.functional as F

from uwir.models import ModelSpec, build_model, parse_model_variant
from uwir.models.fgdpa import build_fgdpa, build_fgdpaslim


FGDPA_VARIANTS = [
    ("fgdpa_3ch", "fgdpa", 3, 60_000),
    ("fgdpa_4ch_t", "fgdpa", 4, 60_000),
    ("fgdpa_4ch_b", "fgdpa", 4, 60_000),
    ("fgdpa_5ch", "fgdpa", 5, 60_000),
    ("fgdpaslim_3ch", "fgdpaslim", 3, 5_000),
    ("fgdpaslim_4ch_t", "fgdpaslim", 4, 5_000),
    ("fgdpaslim_4ch_b", "fgdpaslim", 4, 5_000),
    ("fgdpaslim_5ch", "fgdpaslim", 5, 5_000),
]


@pytest.mark.parametrize(("name", "backbone", "channels", "cap"), FGDPA_VARIANTS)
def test_fgdpa_registry_and_params(name, backbone, channels, cap):
    spec = parse_model_variant(name)
    assert spec == ModelSpec(backbone, channels, spec.physics_mode)
    model = build_model(name, pretrained_backbone=False)
    params = sum(p.numel() for p in model.parameters())
    assert params <= cap, f"{name} parameters {params} exceeded cap {cap}"
    if name == "fgdpaslim_3ch":
        assert params == 4_234, f"Expected exactly 4234 params for fgdpaslim_3ch, got {params}"


@pytest.mark.parametrize("name", ["fgdpa_3ch", "fgdpa_5ch", "fgdpaslim_3ch"])
def test_fgdpa_forward_backward_gradients(name):
    model = build_model(name, pretrained_backbone=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    in_ch = 5 if "5ch" in name else 3
    inputs = torch.rand(2, in_ch, 64, 64)
    targets = torch.rand(2, 3, 64, 64)

    outputs = model(inputs)
    assert outputs.shape == (2, 3, 64, 64)
    assert outputs.min() >= 0.0 and outputs.max() <= 1.0

    loss = F.l1_loss(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", ["fgdpa_3ch", "fgdpaslim_3ch"])
def test_fgdpa_odd_dimension_preservation(name):
    model = build_model(name, pretrained_backbone=False).eval()
    inputs = torch.rand(1, 3, 31, 37)
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (1, 3, 31, 37)


def test_fgdpa_slim_reparameterization_equivalence():
    """Verify that FGDPA.slim() produces functionally equivalent outputs to the multi-branch model."""
    model_train = build_fgdpa(in_channels=3, channels=12, rep_scale=4).eval()
    model_slim = model_train.slim().eval()

    inputs = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        out_train = model_train(inputs)
        out_slim = model_slim(inputs)

    # Check maximum absolute difference between multi-branch and re-parameterized single-conv
    max_diff = (out_train - out_slim).abs().max().item()
    assert max_diff < 1e-4, f"Structural re-param discrepancy too high: {max_diff}"


def test_fgdpaslim_pretrained_checkpoint():
    """Verify that fgdpaslim_3ch loads official pretrained weights properly."""
    model = build_model("fgdpaslim_3ch", pretrained_backbone=True).eval()
    inputs = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (1, 3, 256, 256)
    assert torch.isfinite(outputs).all()
