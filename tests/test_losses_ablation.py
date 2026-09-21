"""
test_losses_ablation.py
------------------------
Unit tests for the modular loss ablation suite:
  1. TVLoss (Total Variation)
  2. EdgeLoss (Gaussian-Laplacian Pyramid)
  3. LocalVarianceLoss (MobileIE LVW)
  4. UIQMLoss (Differentiable Underwater Image Quality Measure)
  5. CompositeLoss 0/1 toggle verification
"""

try:
    import pytest
except ImportError:
    pytest = None
import torch

from uwir.losses import (
    CompositeLoss,
    EdgeLoss,
    GradientDifferenceLoss,
    HVILoss,
    LaplacianPyramidLoss,
    LocalVarianceLoss,
    SSIMLoss,
    TVLoss,
    UIQMLoss,
)


def test_tv_loss():
    tv = TVLoss(loss_weight=1.0)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    loss = tv(pred)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_edge_loss():
    edge = EdgeLoss(loss_weight=1.0)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss = edge(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_local_variance_loss():
    # Spatial mode (exact MobileIE Eq 8)
    lvw_spatial = LocalVarianceLoss(mode="spatial", loss_weight=1.0)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss_s = lvw_spatial(pred, target)
    assert loss_s.dim() == 0
    assert torch.isfinite(loss_s)
    assert loss_s.item() >= 0.0

    loss_s.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    # Window mode (sliding K=7)
    pred_w = torch.rand(2, 3, 64, 64, requires_grad=True)
    lvw_window = LocalVarianceLoss(mode="window", kernel_size=7, loss_weight=1.0)
    loss_w = lvw_window(pred_w, target)
    assert loss_w.dim() == 0
    assert torch.isfinite(loss_w)
    assert loss_w.item() >= 0.0

    loss_w.backward()
    assert pred_w.grad is not None
    assert torch.isfinite(pred_w.grad).all()


def test_uiqm_loss():
    uiqm_loss = UIQMLoss(loss_weight=1.0)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    loss = uiqm_loss(pred)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_hvi_loss():
    hvi_loss = HVILoss(density_k=0.2, loss_weight=1.0)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss = hvi_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # Test identical inputs yield zero loss
    zero_loss = hvi_loss(target, target)
    assert abs(zero_loss.item()) < 1e-6

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_ssim_loss():
    ssim_loss = SSIMLoss()
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss = ssim_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_laplacian_pyramid_loss():
    lap_pyr = LaplacianPyramidLoss(num_levels=3, loss_weight=1.0)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss = lap_pyr(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_gradient_difference_loss():
    gd_loss = GradientDifferenceLoss(loss_weight=1.0, alpha=1)
    pred = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    loss = gd_loss(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # Identical images yield zero gradient difference
    zero_loss = gd_loss(target, target)
    assert abs(zero_loss.item()) < 1e-6

    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_composite_loss_toggles():
    # Only base L1
    comp = CompositeLoss(
        lambda_l1=1.0,
        lambda_perc=0.0,
        use_l1=1,
        use_perc=0,
        use_ssim=0,
        use_tv=0,
        use_edge=0,
        use_gd=0,
        use_lvw=0,
        use_uiqm=0,
        use_hvi=0,
        use_lap_pyr=0,
        device="cpu",
    )
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)
    tot, parts = comp(pred, target)
    assert parts["l1"] > 0.0
    assert parts["perceptual"] == 0.0
    assert parts["ssim_loss"] == 0.0
    assert parts["tv"] == 0.0
    assert parts["edge"] == 0.0
    assert parts["gd"] == 0.0
    assert parts["lvw"] == 0.0
    assert parts["uiqm"] == 0.0
    assert parts["hvi"] == 0.0
    assert parts["lap_pyr"] == 0.0

    # Toggle on TV, Edge, GD, LVW, UIQM, SSIM, HVI, LapPyr
    comp_all = CompositeLoss(
        lambda_l1=1.0,
        lambda_perc=0.0,
        lambda_ssim=0.1,
        lambda_tv=0.001,
        lambda_edge=0.1,
        lambda_gd=1.0,
        lambda_lvw=0.1,
        lambda_uiqm=0.05,
        lambda_hvi=0.5,
        lambda_lap_pyr=1.0,
        use_l1=1,
        use_perc=0,
        use_ssim=1,
        use_tv=1,
        use_edge=1,
        use_gd=1,
        use_lvw=1,
        use_uiqm=1,
        use_hvi=1,
        use_lap_pyr=1,
        device="cpu",
    )
    tot, parts = comp_all(pred, target)
    assert parts["l1"] > 0.0
    assert parts["ssim_loss"] > 0.0
    assert parts["tv"] > 0.0
    assert parts["edge"] > 0.0
    assert parts["gd"] > 0.0
    assert parts["lvw"] > 0.0
    assert parts["uiqm"] > 0.0
    assert parts["hvi"] > 0.0
    assert parts["lap_pyr"] > 0.0
    assert torch.isfinite(tot)

