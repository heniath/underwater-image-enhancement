import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("kornia")

from uwir.cli.train import train_epoch
from uwir.losses import CompositeLoss, PhysicsConsistentLoss
from uwir.models import LearnablePhysicsUNet, PhysicsOutput, build_model, parse_model_variant


def test_registry_builds_rgb_end_to_end_model():
    spec = parse_model_variant("learnable_physics_unet")
    assert (spec.in_channels, spec.physics_mode) == (3, "none")
    assert isinstance(build_model("learnable_physics_unet"), LearnablePhysicsUNet)


def test_wavelength_maps_and_reconstruction_contract():
    model = LearnablePhysicsUNet(extractor_width=8).eval()
    image = torch.rand(2, 3, 32, 48)

    with torch.no_grad():
        output = model(image, return_physics=True)

    assert isinstance(output, PhysicsOutput)
    assert output.enhanced.shape == image.shape
    assert output.transmission.shape == image.shape
    assert output.background.shape == image.shape
    assert torch.all(output.transmission >= 0.1)
    assert torch.all(output.transmission <= 1.0)
    assert torch.all(output.background >= 0.0)
    assert torch.all(output.background <= 1.0)
    assert torch.allclose(output.background, output.background[..., :1, :1].expand_as(image))
    expected = output.enhanced * output.transmission + output.background * (
        1.0 - output.transmission
    )
    assert torch.allclose(output.reconstructed, expected)
    assert torch.equal(model(image), output.enhanced)


def test_physics_loss_updates_both_extractors():
    model = LearnablePhysicsUNet(extractor_width=8)
    image = torch.rand(1, 3, 32, 32)
    target = torch.rand_like(image)
    criterion = PhysicsConsistentLoss(
        CompositeLoss(lambda_perc=0, lambda_ssim=0),
        lambda_reconstruction=1.0,
    )

    loss, parts = criterion(model(image, return_physics=True), target, image)
    loss.backward()

    assert parts["reconstruction"] >= 0.0
    assert any(
        parameter.grad is not None for parameter in model.transmission_extractor.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.background_extractor.parameters())


def test_training_loop_accepts_physics_model():
    model = LearnablePhysicsUNet(extractor_width=8)
    inputs = torch.rand(2, 3, 32, 32)
    targets = torch.rand_like(inputs)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    criterion = PhysicsConsistentLoss(CompositeLoss(lambda_perc=0, lambda_ssim=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss, parts = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    assert loss >= 0.0
    assert parts["reconstruction"] >= 0.0
