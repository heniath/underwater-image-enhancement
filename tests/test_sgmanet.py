"""
test_sgmanet.py
---------------
Unit tests for SGMA-Net (A Lightweight Mamba-Attention Network)
adapted for Underwater Image Restoration.
"""

import pytest
import torch

from uwir.models import ModelSpec, build_model, parse_model_variant
from uwir.models.sgmanet import (
    DSConv2d,
    LGFI,
    MDC,
    MDSA,
    MHSA,
    RHMA,
    SAG,
    SGMANet,
    SGWL,
    MAMBA_CUDA_AVAILABLE,
    MambaSelectiveScan,
    SelectiveScanPureTorch,
    build_sgmanet,
)


def test_mamba_selective_scan():
    """Verify S6 Selective Scan executes cleanly and outputs matching shape."""
    s6 = MambaSelectiveScan(d_model=64, d_state=32)
    x = torch.randn(2, 64, 64)  # (B, L, D)
    out = s6(x)
    assert out.shape == (2, 64, 64)
    assert torch.isfinite(out).all()


def test_lgfi_recursive_attention():
    """Verify LGFI channel-affinity module runs recursively."""
    lgfi = LGFI(channels=64, recursion_depth=2)
    x = torch.randn(2, 64, 64)
    out = lgfi(x)
    assert out.shape == (2, 64, 64)
    assert torch.isfinite(out).all()


def test_rhma_bottleneck():
    """Verify RHMA bottleneck block handles 4D feature map (B, C, H, W)."""
    rhma = RHMA(channels=64, d_state=32, num_heads=2)
    x = torch.randn(2, 64, 8, 8)
    out = rhma(x)
    assert out.shape == (2, 64, 8, 8)
    assert torch.isfinite(out).all()


def test_sag_skip_filtering():
    """Verify SAG filters shallow feature using deeper semantic feature."""
    sag = SAG(channels=32)
    x_i = torch.randn(2, 32, 32, 32)
    h_i = torch.randn(2, 32, 32, 32)
    y_i = sag(x_i, h_i)
    assert y_i.shape == (2, 32, 32, 32)
    assert torch.isfinite(y_i).all()


def test_mdsa_multi_dimensional_attention():
    """Verify MDSA captures responses across multi-dilated views."""
    mdsa = MDSA(channels=32)
    f = torch.randn(2, 32, 32, 32)
    out = mdsa(f)
    assert out.shape == (2, 32, 32, 32)
    assert torch.isfinite(out).all()


def test_sgwl_prediction_fusion():
    """Verify SGWL produces valid Softmax-normalized weights and fused map."""
    sgwl = SGWL(num_levels=3, in_channels_per_pred=3)
    s1 = torch.rand(2, 3, 64, 64)
    s2 = torch.rand(2, 3, 64, 64)
    s3 = torch.rand(2, 3, 64, 64)
    s_final, alphas = sgwl(s1, s2, s3)

    assert s_final.shape == (2, 3, 64, 64)
    assert alphas.shape == (2, 3)
    # Weights must sum to 1.0
    assert torch.allclose(alphas.sum(dim=-1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("name,channels,cap", [
    ("sgmanet_3ch", 3, 1_250_000),
    ("sgmanet_4ch_t", 4, 1_250_000),
    ("sgmanet_4ch_b", 4, 1_250_000),
    ("sgmanet_5ch", 5, 1_250_000),
])
def test_sgmanet_registration_and_params(name, channels, cap):
    spec = parse_model_variant(name)
    assert spec == ModelSpec("sgmanet", channels, spec.physics_mode)
    model = build_model(name, pretrained_backbone=False)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count <= cap, f"{name} parameters {param_count} exceeded cap {cap}"


@pytest.mark.parametrize("in_channels", [3, 5])
def test_sgmanet_shape_preservation_and_resolutions(in_channels):
    model = build_sgmanet(in_channels=in_channels).eval()

    # Standard patch size
    x1 = torch.rand(1, in_channels, 64, 64)
    y1 = model(x1)
    assert y1.shape == (1, 3, 64, 64)
    assert y1.min() >= 0.0 and y1.max() <= 1.0

    # Training crop size
    x2 = torch.rand(1, in_channels, 256, 256)
    y2 = model(x2)
    assert y2.shape == (1, 3, 256, 256)
    assert y2.min() >= 0.0 and y2.max() <= 1.0

    # Non-square input
    x3 = torch.rand(1, in_channels, 192, 256)
    y3 = model(x3)
    assert y3.shape == (1, 3, 192, 256)
    assert y3.min() >= 0.0 and y3.max() <= 1.0


def test_sgmanet_forward_backward_gradients():
    model = build_sgmanet(in_channels=3)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    inputs = torch.rand(2, 3, 64, 64)
    targets = torch.rand(2, 3, 64, 64)

    outputs = model(inputs)
    assert outputs.shape == (2, 3, 64, 64)

    loss = torch.nn.functional.l1_loss(outputs, targets)
    optimizer.zero_grad()
    loss.backward()

    # Verify all gradients are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"


def test_sgmanet_5ch_full_composite_loss():
    """Verify sgmanet_5ch backward with all loss components (Charbonnier, SSIM, Color, Wavelet)."""
    from uwir.losses import CompositeLoss

    model = build_model("sgmanet_5ch", pretrained_backbone=False)
    model.train()
    criterion = CompositeLoss(
        lambda_l1=1.0,
        lambda_perc=0.05,
        lambda_ssim=0.3,
        lambda_color=0.2,
        lambda_wavelet=0.1,
        use_charbonnier=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    inputs = torch.rand(2, 5, 64, 64)
    targets = torch.rand(2, 3, 64, 64)

    outputs = model(inputs)
    assert outputs.shape == (2, 3, 64, 64)

    loss, comps = criterion(outputs, targets)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in comps.values())

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist and are finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"5ch: Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"5ch: Parameter {name} has non-finite gradient"

