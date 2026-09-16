"""
test_tournament_models.py
-------------------------
Unit tests for the 5 lightweight tournament models:
1. NAFNet-Tiny
2. FA-Net (FA*Net)
3. MobileIE
4. LiteEnhanceNet
5. LSNet
"""

import pytest
import torch

from uwir.models import ModelSpec, build_model, parse_model_variant

TOURNAMENT_MODELS = [
    ("nafnettiny_3ch", "nafnettiny", 3, 110_000),
    ("fanet_3ch", "fanet", 3, 95_000),
    ("fanet_5ch", "fanet", 5, 95_000),
    ("mobileie_3ch", "mobileie", 3, 45_000),
    ("mobileie_5ch", "mobileie", 5, 45_000),
    ("liteenhancenet_3ch", "liteenhancenet", 3, 30_000),
    ("liteenhancenet_5ch", "liteenhancenet", 5, 30_000),
    ("lsnet_3ch", "lsnet", 3, 30_000),
    ("lsnet_5ch", "lsnet", 5, 30_000),
]



@pytest.mark.parametrize(("name", "backbone", "channels", "cap"), TOURNAMENT_MODELS)
def test_tournament_models_registry_and_params(name, backbone, channels, cap):
    spec = parse_model_variant(name)
    assert spec == ModelSpec(backbone, channels, spec.physics_mode)
    model = build_model(name, pretrained_backbone=False)
    params = sum(p.numel() for p in model.parameters())
    assert params <= cap, f"{name} parameters {params} exceeded cap {cap}"


@pytest.mark.parametrize("name", [
    "nafnettiny_3ch",
    "fanet_3ch",
    "mobileie_3ch",
    "liteenhancenet_3ch",
    "lsnet_3ch",
])
def test_forward_backward_finite_gradients(name):
    model = build_model(name, pretrained_backbone=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    inputs = torch.rand(2, 3, 32, 32)
    targets = torch.rand(2, 3, 32, 32)

    outputs = model(inputs)
    assert outputs.shape == (2, 3, 32, 32)
    assert outputs.min() >= 0.0 and outputs.max() <= 1.0

    loss = torch.nn.functional.l1_loss(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", [
    "fanet_3ch",
    "mobileie_3ch",
    "liteenhancenet_3ch",
    "lsnet_3ch",
])
def test_odd_dimension_preservation(name):
    model = build_model(name, pretrained_backbone=False).eval()
    inputs = torch.rand(1, 3, 31, 37)
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (1, 3, 31, 37)
