from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from uwir.models import ALL_MODEL_NAMES, ModelSpec, build_model, parse_model_variant
from uwir.models.fusion_unet import ZeroGatedFusion
from uwir.models.hybrid_mamba_unet import (
    ECAResidual,
    GatedDecoderBlock,
    HybridMambaBlock,
    HybridMambaUNet,
    WindowAttentionBlock,
    window_partition,
    window_reverse,
)

HYBRID_VARIANTS = {
    "hybridmamba_core_3ch": ModelSpec("hybridmamba_core", 3, "none"),
    "hybridmamba_local_3ch": ModelSpec("hybridmamba_local", 3, "none"),
    "hybridmamba_attn_3ch": ModelSpec("hybridmamba_attn", 3, "none"),
    "hybridmambafusion_7ch_tv": ModelSpec("hybridmambafusion", 7, "tv"),
}


class _ZeroScan(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def _small_model(**kwargs) -> HybridMambaUNet:
    defaults = {
        "widths": (8, 12, 16, 24, 32),
        "d_state": 2,
        "num_heads": 4,
        "window_size": 4,
        "use_checkpoint": False,
    }
    defaults.update(kwargs)
    return HybridMambaUNet(**defaults)


@pytest.mark.parametrize(("name", "spec"), HYBRID_VARIANTS.items())
def test_hybrid_registry_names_channel_contracts_and_outputs(monkeypatch, name, spec):
    monkeypatch.setattr(
        "uwir.models.hybrid_mamba_unet.SS2D.forward", lambda _self, x: torch.zeros_like(x)
    )
    assert name in ALL_MODEL_NAMES
    assert parse_model_variant(name) == spec
    model = build_model(name, pretrained_backbone=False).eval()
    assert model.in_channels == spec.in_channels
    sample = torch.rand(1, spec.in_channels, 32, 32)
    with torch.no_grad():
        output = model(sample)
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, sample[:, :3], atol=1e-4)


@pytest.mark.parametrize(
    ("physics", "channels"),
    [(False, 3), (True, 7)],
)
def test_hybrid_output_is_finite_rgb_identity(monkeypatch, physics, channels):
    monkeypatch.setattr(
        "uwir.models.hybrid_mamba_unet.SS2D.forward", lambda _self, x: torch.zeros_like(x)
    )
    model = _small_model(in_channels=channels, use_physics=physics).eval()
    sample = torch.rand(1, channels, 35, 43)
    with torch.no_grad():
        output = model(sample)
    assert output.shape == (1, 3, 35, 43)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, sample[:, :3], atol=1e-4)


def test_hybrid_rejects_wrong_input_channels():
    model = _small_model().eval()
    with pytest.raises(ValueError, match="expects 3 channels"):
        model(torch.rand(1, 7, 32, 32))
    with pytest.raises(ValueError, match="requires in_channels=7"):
        _small_model(in_channels=3, use_physics=True)


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "uwir.models.hybrid_mamba_unet.SS2D.forward", lambda _self, x: torch.zeros_like(x)
    )
    source = _small_model().eval()
    sample = torch.rand(1, 3, 32, 32)
    expected = source(sample)
    checkpoint = tmp_path / "hybrid.pt"
    torch.save(source.state_dict(), checkpoint)
    restored = _small_model().eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(expected, restored(sample))


@pytest.mark.parametrize("use_local_branch", [False, True])
def test_hybrid_block_scales_receive_gradients(use_local_branch):
    torch.manual_seed(3)
    block = HybridMambaBlock(4, d_state=2, use_local_branch=use_local_branch)
    block.ss2d = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    output = block(torch.randn(2, 4, 3, 5)).square().mean()
    output.backward()
    assert block.mamba_scale.grad is not None
    assert torch.isfinite(block.mamba_scale.grad).all()
    if use_local_branch:
        assert block.local_scale.grad is not None
        assert torch.isfinite(block.local_scale.grad).all()
    else:
        assert block.local_scale is None


def test_hybrid_block_keeps_ss2d_stable_under_autocast():
    block = HybridMambaBlock(4, d_state=2, use_local_branch=False)
    sample = torch.randn(2, 4, 3, 5, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = block(sample)
        loss = output.square().mean()
    loss.backward()
    assert output.dtype == sample.dtype
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )


@pytest.mark.parametrize("shape", [(2, 5, 11, 8), (1, 9, 6, 8)])
def test_window_partition_round_trip_for_odd_nonsquare_maps(shape):
    sample = torch.randn(shape)
    windows, metadata = window_partition(sample, 4)
    restored = window_reverse(windows, metadata, 4)
    assert restored.shape == sample.shape
    assert torch.equal(restored, sample)


def test_window_attention_preserves_odd_nonsquare_shape_and_gradients():
    block = WindowAttentionBlock(16, window_size=4, num_heads=4, mlp_ratio=2)
    sample = torch.randn(2, 16, 5, 11, requires_grad=True)
    output = block(sample)
    output.mean().backward()
    assert output.shape == sample.shape
    assert block.attention_scale.grad is not None
    assert block.mlp_scale.grad is not None


def test_eca_shape_and_decoder_gate_range_and_initial_value():
    sample = torch.randn(2, 8, 9, 7)
    assert ECAResidual(8)(sample).shape == sample.shape
    decoder = GatedDecoderBlock(16, 8, 8)
    output = decoder(torch.randn(2, 16, 5, 4), sample)
    assert output.shape == sample.shape
    assert decoder.last_gate is not None
    assert torch.all((decoder.last_gate >= 0) & (decoder.last_gate <= 1))
    assert torch.allclose(decoder.last_gate, torch.sigmoid(torch.tensor(2.0)))


def test_physics_fusion_is_exactly_inactive_at_initialization():
    fusion = ZeroGatedFusion(8, 4).eval()
    image = torch.randn(2, 8, 9, 11)
    physics_a = torch.randn(2, 4, 9, 11)
    physics_b = torch.randn(2, 4, 9, 11)
    assert torch.equal(fusion(image, physics_a), image)
    assert torch.equal(fusion(image, physics_b), image)


def test_real_ss2d_forward_backward_smoke_at_64():
    model = _small_model(use_local_branch=True, use_attention=True).train()
    output = model(torch.rand(1, 3, 64, 64, requires_grad=True))
    output.mean().backward()
    assert output.shape == (1, 3, 64, 64)
    assert model.dec_full.gate.bias.grad is not None
