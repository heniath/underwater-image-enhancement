"""
test_fanetplus.py
-----------------
Unit tests for FA*Net-Plus (FANetPlus) architecture with Haar Wavelet
multi-frequency processing and Dual-Dilation RFAB.
"""

import pytest
import torch

from uwir.models import ModelSpec, build_model, parse_model_variant
from uwir.models.fanetplus import (
    DualDilationConv,
    EnhancedRFAB,
    FANetPlus,
    HaarWavelet2D,
    InverseHaarWavelet2D,
    WaveletFrequencyBranch,
    build_fanetplus,
)


def test_haar_wavelet_round_trip_perfect_reconstruction():
    dwt = HaarWavelet2D()
    idwt = InverseHaarWavelet2D()

    # Even spatial dimension
    x_even = torch.randn(2, 32, 64, 64)
    freq, pad_h, pad_w = dwt(x_even)
    assert freq.shape == (2, 32 * 4, 32, 32)
    x_rec = idwt(freq, pad_h, pad_w)
    assert torch.allclose(x_rec, x_even, atol=1e-5, rtol=1e-5)

    # Odd spatial dimension
    x_odd = torch.randn(2, 16, 31, 35)
    freq, pad_h, pad_w = dwt(x_odd)
    assert pad_h == 1 and pad_w == 1
    x_rec_odd = idwt(freq, pad_h, pad_w)
    assert torch.allclose(x_rec_odd, x_odd, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name,channels,cap", [
    ("fanetplus_3ch", 3, 125_000),
    ("fanetplus_4ch_t", 4, 125_000),
    ("fanetplus_5ch", 5, 125_000),
])

def test_fanetplus_registration_and_params(name, channels, cap):
    spec = parse_model_variant(name)
    assert spec == ModelSpec("fanetplus", channels, spec.physics_mode)
    model = build_model(name, pretrained_backbone=False)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count <= cap, f"{name} parameters {param_count} exceeded cap {cap}"


@pytest.mark.parametrize("in_channels", [3, 5])
def test_fanetplus_shape_preservation_and_identity_init(in_channels):
    model = build_fanetplus(in_channels=in_channels).eval()

    # Standard size
    x = torch.rand(2, in_channels, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Identity residual initialization guarantees initially output == input[:, :3]
    assert torch.equal(out, x[:, :3])

    # Odd dimensions
    x_odd = torch.rand(1, in_channels, 33, 37)
    with torch.no_grad():
        out_odd = model(x_odd)
    assert out_odd.shape == (1, 3, 33, 37)


def test_fanetplus_finite_gradients_backward():
    model = build_model("fanetplus_5ch", pretrained_backbone=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    inputs = torch.rand(2, 5, 32, 32)
    targets = torch.rand(2, 3, 32, 32)

    outputs = model(inputs)
    loss = torch.nn.functional.l1_loss(outputs, targets)

    optimizer.zero_grad()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
