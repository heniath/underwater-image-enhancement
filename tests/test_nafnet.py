"""Tests for NAFNet architectures adapted for lightweight underwater restoration."""

import pytest
import torch

from uwir.models import ModelSpec, build_model, parse_model_variant
from uwir.models.nafnet import (
    LayerNorm2d,
    NAFBlock,
    NAFNet,
    SimpleGate,
    SimplifiedChannelAttention,
    build_nafnet,
    build_nafnetmicro,
    build_nafnettiny,
)

NAFNET_VARIANTS = [
    ("nafnettiny_3ch", "nafnettiny", 3, 110_000),
    ("nafnettiny_4ch_t", "nafnettiny", 4, 110_000),
    ("nafnettiny_4ch_b", "nafnettiny", 4, 110_000),
    ("nafnettiny_5ch", "nafnettiny", 5, 110_000),
    ("nafnetmicro_3ch", "nafnetmicro", 3, 50_000),
    ("nafnet_3ch", "nafnet", 3, 300_000),
]


@pytest.mark.parametrize(("model_name", "family", "channels", "cap"), NAFNET_VARIANTS)
def test_nafnet_variants_registered_and_respect_caps(model_name, family, channels, cap):
    spec = parse_model_variant(model_name)
    assert spec == ModelSpec(family, channels, spec.physics_mode)
    model = build_model(model_name, pretrained_backbone=False)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count <= cap, f"{model_name} params {param_count} exceeded cap {cap}"


def test_nafnet_blocks_components():
    # Test SimpleGate
    gate = SimpleGate()
    x = torch.randn(2, 8, 16, 16)
    out = gate(x)
    assert out.shape == (2, 4, 16, 16)

    # Test SimplifiedChannelAttention
    sca = SimplifiedChannelAttention(channels=16)
    x = torch.randn(2, 16, 16, 16)
    out = sca(x)
    assert out.shape == (2, 16, 16, 16)

    # Test LayerNorm2d
    ln = LayerNorm2d(channels=16)
    x = torch.randn(2, 16, 16, 16)
    out = ln(x)
    assert out.shape == (2, 16, 16, 16)
    assert torch.allclose(out.mean(dim=1), torch.zeros(2, 16, 16), atol=1e-5)

    # Test NAFBlock
    block = NAFBlock(c=16)
    out = block(x)
    assert out.shape == (2, 16, 16, 16)


@pytest.mark.parametrize("channels", [3, 4, 5])
def test_nafnettiny_odd_dimension_padding_and_identity_init(channels):
    model = build_nafnettiny(in_channels=channels).eval()
    inputs = torch.rand(2, channels, 31, 35)
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (2, 3, 31, 35)
    assert output.min() >= 0.0 and output.max() <= 1.0
    assert torch.equal(output, inputs[:, :3])


def test_nafnet_finite_gradients_backward():
    model = build_model("nafnettiny_3ch")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    inputs = torch.rand(2, 3, 32, 32, requires_grad=False)
    target = torch.rand(2, 3, 32, 32)

    output = model(inputs)
    loss = torch.nn.functional.l1_loss(output, target)
    optimizer.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
