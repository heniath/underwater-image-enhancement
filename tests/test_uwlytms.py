import pytest
import torch

from uwir.models import UWLYTMS, ModelSpec, build_model, parse_model_variant
from uwir.models.uwlyt import LowResolutionAttention, SeparableDownsample, SkipFusion


@pytest.mark.parametrize(
    ("name", "channels", "physics_mode"),
    (
        ("uwlytms_3ch", 3, "none"),
        ("uwlytms_4ch_t", 4, "t"),
        ("uwlytms_4ch_b", 4, "b"),
        ("uwlytms_5ch", 5, "tb"),
    ),
)
def test_multiscale_variants_are_registered_and_preserve_identity(name, channels, physics_mode):
    assert parse_model_variant(name) == ModelSpec("uwlytms", channels, physics_mode)
    model = build_model(name).eval()
    inputs = torch.rand(1, channels, 31, 35)
    with torch.no_grad():
        output = model(inputs)

    assert output.shape == (1, 3, 31, 35)
    assert torch.equal(output, inputs[:, :3])
    assert sum(parameter.numel() for parameter in model.parameters()) <= 125_000


def test_multiscale_architecture_has_three_encoder_levels_and_skip_fusion():
    model = UWLYTMS()
    assert sum(isinstance(module, SeparableDownsample) for module in model.modules()) == 3
    assert sum(isinstance(module, SkipFusion) for module in model.modules()) == 3
    assert sum(isinstance(module, LowResolutionAttention) for module in model.modules()) == 1

    spatial_shapes = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: spatial_shapes.append(output.shape[-2:])
        )
        for layer in (model.down1, model.down2, model.down3)
    ]
    try:
        model(torch.rand(1, 3, 64, 80))
    finally:
        for handle in handles:
            handle.remove()
    assert spatial_shapes == [(32, 40), (16, 20), (8, 10)]


def test_multiscale_model_has_finite_gradients_after_identity_head_opens():
    model = UWLYTMS(width=32)
    with torch.no_grad():
        model.residual_head.weight.fill_(1e-3)
        attention = next(
            module for module in model.modules() if isinstance(module, LowResolutionAttention)
        )
        attention.gate.fill_(1e-2)

    inputs = torch.rand(2, 3, 33, 39)
    target = torch.rand(2, 3, 33, 39)
    torch.nn.functional.l1_loss(model(inputs), target).backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
