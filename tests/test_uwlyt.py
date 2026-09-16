from pathlib import Path

import pytest
import torch

from uwir.cli.train import load_ckpt, save_ckpt
from uwir.models import ModelSpec, build_model, parse_model_variant
from uwir.models.uwlyt import (
    UWLYT,
    AxialDepthwiseBlock,
    LowResolutionAttention,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)

VARIANTS = [
    (f"{family}_{variant}", family, channels, cap)
    for family, cap in (("uwlyt", 50_000), ("uwlyttiny", 20_000))
    for variant, channels in (("3ch", 3), ("4ch_t", 4), ("4ch_b", 4), ("5ch", 5))
]


def test_ycbcr_round_trip_is_accurate_and_differentiable():
    rgb = torch.rand(2, 3, 17, 19, requires_grad=True)
    restored = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
    assert torch.allclose(restored, rgb, atol=1e-5, rtol=1e-5)
    restored.mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()


@pytest.mark.parametrize(("model_name", "family", "channels", "cap"), VARIANTS)
def test_variants_are_registered_and_respect_parameter_caps(model_name, family, channels, cap):
    spec = parse_model_variant(model_name)
    assert spec == ModelSpec(family, channels, spec.physics_mode)
    model = build_model(model_name, pretrained_backbone=False)
    assert sum(parameter.numel() for parameter in model.parameters()) <= cap


@pytest.mark.parametrize("channels", [3, 4, 5])
def test_output_shape_range_odd_padding_and_identity_initialization(channels):
    model = UWLYT(in_channels=channels).eval()
    inputs = torch.rand(2, channels, 31, 35)
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (2, 3, 31, 35)
    assert output.min() >= 0 and output.max() <= 1
    assert torch.equal(output, inputs[:, :3])


def test_architecture_contains_required_lightweight_components():
    model = UWLYT(in_channels=5)
    assert any(isinstance(module, AxialDepthwiseBlock) for module in model.modules())
    assert any(isinstance(module, LowResolutionAttention) for module in model.modules())
    assert model.prior_projection is not None
    assert torch.count_nonzero(model.residual_head.weight) == 0
    assert torch.count_nonzero(model.residual_head.bias) == 0


def test_finite_backward_gradients_and_cpu_training_step():
    model = build_model("uwlyt_5ch", pretrained_backbone=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    inputs = torch.rand(2, 5, 17, 21)
    target = torch.rand(2, 3, 17, 21)
    before = model.residual_head.weight.detach().clone()
    loss = torch.nn.functional.l1_loss(model(inputs), target)
    optimizer.zero_grad()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    assert not torch.equal(model.residual_head.weight, before)


def test_checkpoint_restoration(tmp_path: Path):
    model = build_model("uwlyttiny_4ch_t", pretrained_backbone=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with torch.no_grad():
        model.residual_head.bias.fill_(0.125)
    checkpoint = tmp_path / "uwlyt" / "best_model.pth"
    save_ckpt(model, optimizer, 9, {"psnr": 21.0}, str(checkpoint))

    restored = build_model("uwlyttiny_4ch_t", pretrained_backbone=False)
    epoch, metrics = load_ckpt(str(checkpoint), restored)
    assert epoch == 9 and metrics == {"psnr": 21.0}
    assert torch.equal(restored.residual_head.bias, model.residual_head.bias)
