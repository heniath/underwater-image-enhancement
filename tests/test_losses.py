"""
test_losses.py
--------------
Unit tests for advanced loss functions:
1. CharbonnierLoss
2. ColorAngleLoss
3. WaveletLoss
4. CompositeLoss
"""

import pytest
import torch

from uwir.losses import (
    CharbonnierLoss,
    ColorAngleLoss,
    CompositeLoss,
    EdgeLoss,
    LocalVarianceWeightedLoss,
    MobileIELoss,
    SSIMLoss,
    TotalVariationLoss,
    UIQMLoss,
    WaveletLoss,
)


def test_charbonnier_loss_basic():
    loss_fn = CharbonnierLoss(eps=1e-3)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    val = loss_fn(pred, target)
    assert val.ndim == 0
    assert val.item() > 0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_charbonnier_zero_diff():
    loss_fn = CharbonnierLoss(eps=1e-3)
    x = torch.ones(2, 3, 16, 16)
    val = loss_fn(x, x)
    # When diff == 0, sqrt(0 + eps^2) = eps
    assert torch.isclose(val, torch.tensor(1e-3), atol=1e-6)


def test_color_angle_loss_basic():
    loss_fn = ColorAngleLoss()
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    val = loss_fn(pred, target)
    assert val.ndim == 0
    assert 0.0 <= val.item() <= 2.0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_color_angle_loss_scale_invariance():
    """Color angle loss measures chromaticity direction; scaling intensity should yield ~0 loss."""
    loss_fn = ColorAngleLoss()
    x = torch.rand(2, 3, 16, 16).clamp(min=0.1)
    scaled = x * 0.5  # Darker version of exact same color hue

    val = loss_fn(x, scaled)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-4)


def test_wavelet_loss_basic():
    loss_fn = WaveletLoss(hf_weight=0.5)

    # Even spatial dimension
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    val = loss_fn(pred, target)
    assert val.ndim == 0
    assert val.item() > 0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Odd spatial dimension (tests padding logic)
    pred_odd = torch.rand(2, 3, 31, 33, requires_grad=True)
    target_odd = torch.rand(2, 3, 31, 33)
    val_odd = loss_fn(pred_odd, target_odd)
    assert torch.isfinite(val_odd)
    val_odd.backward()
    assert torch.isfinite(pred_odd.grad).all()


def test_composite_loss_all_components():
    """Test CompositeLoss with Charbonnier, SSIM, Color, and Wavelet losses active."""
    criterion = CompositeLoss(
        lambda_l1=1.0,
        lambda_perc=0.0,
        lambda_ssim=0.3,
        lambda_color=0.2,
        lambda_wavelet=0.1,
        use_charbonnier=True,
    )

    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    total, parts = criterion(pred, target)
    assert torch.isfinite(total)
    assert total.item() > 0

    expected_keys = {
        "l1",
        "perceptual",
        "ssim_loss",
        "color",
        "wavelet",
        "lvw",
        "edge",
        "tv",
        "uiqm",
        "total",
    }
    assert set(parts.keys()) == expected_keys
    assert parts["perceptual"] == 0.0
    assert parts["ssim_loss"] > 0
    assert parts["color"] > 0
    assert parts["wavelet"] > 0
    assert parts["lvw"] == 0.0
    assert parts["edge"] == 0.0
    assert parts["tv"] == 0.0
    assert parts["uiqm"] == 0.0

    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_composite_loss_backward_compatibility():
    """Verify default CompositeLoss reproduces legacy interface."""
    criterion = CompositeLoss(lambda_l1=1.0, lambda_perc=0.0, lambda_ssim=0.5)

    pred = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)

    total, parts = criterion(pred, target)
    assert torch.isfinite(total)
    assert parts["color"] == 0.0
    assert parts["wavelet"] == 0.0
    assert parts["lvw"] == 0.0
    assert parts["edge"] == 0.0
    assert parts["tv"] == 0.0
    assert parts["uiqm"] == 0.0


def test_local_variance_weighted_loss_basic():
    """Test LocalVarianceWeightedLoss (MobileIE LVW / OutlierAware)."""
    loss_fn = LocalVarianceWeightedLoss()
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    val = loss_fn(pred, target)
    assert val.ndim == 0
    assert val.item() >= 0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Verify alias
    assert MobileIELoss is LocalVarianceWeightedLoss


def test_edge_loss_basic():
    """Test Sobel-based EdgeLoss."""
    loss_fn = EdgeLoss()
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    val = loss_fn(pred, target)
    assert val.ndim == 0
    assert val.item() > 0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_edge_loss_identical_inputs():
    """Gradient difference for identical inputs must be zero."""
    loss_fn = EdgeLoss()
    x = torch.rand(2, 3, 32, 32)
    val = loss_fn(x, x)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_total_variation_loss_basic():
    """Test TotalVariationLoss."""
    loss_fn = TotalVariationLoss()
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)

    val = loss_fn(pred)
    assert val.ndim == 0
    assert val.item() > 0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_total_variation_loss_flat_image():
    """TV loss for constant flat image must be zero."""
    loss_fn = TotalVariationLoss()
    flat = torch.full((2, 3, 32, 32), 0.5)
    val = loss_fn(flat)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-6)


def test_uiqm_loss_basic():
    """Test Differentiable UIQMLoss."""
    loss_fn = UIQMLoss()
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)

    val = loss_fn(pred)
    assert val.ndim == 0
    assert torch.isfinite(val)

    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_composite_loss_with_new_losses():
    """Test CompositeLoss with all newly introduced losses active."""
    criterion = CompositeLoss(
        lambda_l1=0.5,
        lambda_perc=0.0,
        lambda_ssim=0.2,
        lambda_color=0.1,
        lambda_wavelet=0.1,
        lambda_lvw=0.5,
        lambda_edge=0.2,
        lambda_tv=0.01,
        lambda_uiqm=0.05,
    )

    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    total, parts = criterion(pred, target)
    assert torch.isfinite(total)
    assert total.item() != 0.0

    expected_keys = {
        "l1",
        "perceptual",
        "ssim_loss",
        "color",
        "wavelet",
        "lvw",
        "edge",
        "tv",
        "uiqm",
        "total",
    }
    assert set(parts.keys()) == expected_keys
    assert parts["lvw"] > 0
    assert parts["edge"] > 0
    assert parts["tv"] > 0

    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_edge_and_uiqm_loss_amp_bfloat16():
    """Verify EdgeLoss and UIQMLoss accept bfloat16/float16 tensors under AMP without dtype mismatch."""
    edge_fn = EdgeLoss()
    uiqm_fn = UIQMLoss()

    for dt in (torch.bfloat16, torch.float16):
        pred = torch.rand(2, 3, 32, 32, dtype=dt, requires_grad=True)
        target = torch.rand(2, 3, 32, 32, dtype=dt)

        l_edge = edge_fn(pred, target)
        assert l_edge.dtype == dt
        assert torch.isfinite(l_edge)

        l_uiqm = uiqm_fn(pred)
        assert l_uiqm.dtype == dt
        assert torch.isfinite(l_uiqm)

