"""
tests/test_pcf_mbconv.py
------------------------
Unit tests for PCF-MBConv model components, re-parameterization equivalence,
loss computation, and registry integration.
"""

import pytest
import torch
import torch.nn as nn

from src.uwir.losses import CompositeLoss, HSVCSLoss
from src.uwir.models import PCFMBConvUNet, PCFTinyUNet, build_model, parse_model_variant
from src.uwir.models.pcf_modules import (
    CSAGF,
    ColorBiasAwareModule,
    RepDepthwiseConv2d,
    ValueConfidenceModule,
    rgb_to_hsv_cs,
)


def test_hsv_cs_conversion_properties():
    """Verify that rgb_to_hsv_cs produces continuous values in [0, 1] without NaN/Inf."""
    # Random RGB inputs
    x = torch.rand(4, 3, 64, 64)
    hsv_cs = rgb_to_hsv_cs(x)

    assert hsv_cs.shape == (4, 4, 64, 64)
    assert not torch.isnan(hsv_cs).any()
    assert not torch.isinf(hsv_cs).any()
    assert (hsv_cs >= 0.0).all() and (hsv_cs <= 1.0).all()

    # Extreme corner cases: black, white, pure primaries
    corners = torch.tensor([
        [0.0, 0.0, 0.0],  # black
        [1.0, 1.0, 1.0],  # white
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
    ]).view(5, 3, 1, 1)

    hsv_corners = rgb_to_hsv_cs(corners)
    assert not torch.isnan(hsv_corners).any()
    assert (hsv_corners >= 0.0).all() and (hsv_corners <= 1.0).all()


def test_color_bias_and_confidence():
    """Verify ColorBiasAwareModule and ValueConfidenceModule ranges."""
    cb = ColorBiasAwareModule(in_channels=3, mid_channels=16)
    vc = ValueConfidenceModule(mid_channels=16)

    rgb = torch.rand(2, 3, 32, 32)
    w_cb = cb(rgb)
    assert w_cb.shape == (2, 1, 32, 32)
    assert (w_cb >= 0.0).all() and (w_cb <= 1.0).all()

    v = torch.rand(2, 1, 32, 32)
    c_conf = vc(v)
    assert c_conf.shape == (2, 1, 32, 32)
    assert (c_conf >= 0.0).all() and (c_conf <= 1.0).all()


def test_csagf_fusion():
    """Verify CSAGF fusion preserves shape and blends modalities."""
    csagf = CSAGF(channels=64)
    f_rgb = torch.randn(2, 64, 16, 16)
    f_hsv = torch.randn(2, 64, 16, 16)

    f_fused = csagf(f_rgb, f_hsv)
    assert f_fused.shape == (2, 64, 16, 16)


def test_rep_depthwise_conv_equivalence():
    """Verify that multi-branch training conv folds into a single Conv2d with < 1e-4 diff."""
    rep = RepDepthwiseConv2d(channels=32)
    rep.eval()

    x = torch.randn(2, 32, 24, 24)
    with torch.no_grad():
        out_train = rep(x)
        rep.switch_to_deploy()
        out_deploy = rep(x)

    diff = (out_train - out_deploy).abs().max().item()
    assert diff < 1e-4, f"Re-parameterization difference too high: {diff}"
    assert rep.is_deployed
    assert isinstance(rep.deployed_conv, nn.Conv2d)


def test_pcf_mbconv_forward_shape():
    """Verify full model forward pass with 5-channel input."""
    model = PCFMBConvUNet(in_channels=5, out_channels=3)
    x = torch.rand(2, 5, 128, 128)

    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 3, 128, 128)
    assert (y >= 0.0).all() and (y <= 1.0).all()


def test_pcf_mbconv_model_reparam_equivalence():
    """Verify full PCFMBConvUNet switch_to_deploy() numerical equivalence."""
    model = PCFMBConvUNet(in_channels=5, out_channels=3)
    model.eval()
    x = torch.rand(2, 5, 64, 64)

    with torch.no_grad():
        y1 = model(x)
        model.switch_to_deploy()
        y2 = model(x)

    diff = (y1 - y2).abs().max().item()
    assert diff < 1e-4, f"Full model deployment difference too high: {diff}"


def test_pcf_mbconv_param_budget():
    """Verify lightweight parameter budget <= 4.5M params."""
    model = PCFMBConvUNet(in_channels=5, out_channels=3)
    train_params = sum(p.numel() for p in model.parameters())
    model.switch_to_deploy()
    deploy_params = sum(p.numel() for p in model.parameters())

    assert train_params < 4_500_000, f"Train params exceed 4.5M: {train_params}"
    assert deploy_params < 4_000_000, f"Deploy params exceed 4.0M: {deploy_params}"
    assert deploy_params <= train_params


def test_composite_loss_hsvcs_backward():
    """Verify HSVCSLoss and CompositeLoss compute gradients properly."""
    criterion = CompositeLoss(
        lambda_l1=1.0,
        lambda_perc=0.0,
        lambda_ssim=0.0,
        lambda_hsvcs=0.5,
    )

    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)

    loss, parts = criterion(pred, target)
    assert "hsvcs" in parts
    assert parts["hsvcs"] > 0.0
    assert loss.item() > 0.0

    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_registry_integration():
    """Verify that build_model registers and constructs all PCF variants."""
    for variant, expected_cls in [
        ("pcf_mbconv_5ch", PCFMBConvUNet),
        ("pcf_mbconv_3ch", PCFMBConvUNet),
        ("pcf_tiny_3ch", PCFTinyUNet),
        ("pcf_tiny_5ch", PCFTinyUNet),
    ]:
        spec = parse_model_variant(variant)
        assert spec is not None
        model = build_model(variant)
        assert isinstance(model, expected_cls)


def test_pcf_tiny_unet_architecture():
    """Verify original PCF-Net scale model (< 0.2M params) and numerical re-parameterization."""
    model = PCFTinyUNet(in_channels=3, out_channels=3)
    params = sum(p.numel() for p in model.parameters())
    assert params < 200_000, f"Expected < 200K params for tiny PCF-Net, got: {params}"

    x = torch.rand(2, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        y1 = model(x)
        model.switch_to_deploy()
        y2 = model(x)

    assert y1.shape == (2, 3, 64, 64)
    diff = (y1 - y2).abs().max().item()
    assert diff < 1e-4, f"PCFTinyUNet deployment diff too high: {diff}"


def test_pcf_mbconv_3ch_forward():
    """Verify 3-channel input forward pass and fallback handling."""
    model_3ch = PCFMBConvUNet(in_channels=3, out_channels=3)
    x = torch.rand(2, 3, 64, 64)
    y = model_3ch(x)
    assert y.shape == (2, 3, 64, 64)

    # Test fallback: passing 3 channels to 5-channel model automatically pads without crashing
    model_5ch = PCFMBConvUNet(in_channels=5, out_channels=3)
    y_fallback = model_5ch(x)
    assert y_fallback.shape == (2, 3, 64, 64)

