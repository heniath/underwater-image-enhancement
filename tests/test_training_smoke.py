from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("kornia")

from uwir.cli.train import (
    EarlyStopping,
    _split_train_validation,
    load_ckpt,
    save_ckpt,
    train_epoch,
    val_loss_epoch,
)
from uwir.losses import CompositeLoss
from uwir.models import build_model


def _loader(channels=5):
    inputs = torch.rand(2, channels, 32, 32)
    targets = torch.rand(2, 3, 32, 32)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def test_forward_and_small_training_step():
    model = build_model("unet_5ch", pretrained_backbone=False)
    criterion = CompositeLoss(lambda_perc=0, lambda_ssim=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loss, _ = train_epoch(model, _loader(), optimizer, criterion, torch.device("cpu"))
    val_loss = val_loss_epoch(model, _loader(), criterion, torch.device("cpu"))

    assert train_loss >= 0
    assert val_loss >= 0


def test_checkpoint_round_trip(tmp_path: Path):
    model = build_model("unet_3ch", pretrained_backbone=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "nested" / "model.pth"
    save_ckpt(model, optimizer, 7, {"psnr": 28.5}, str(checkpoint))

    restored = build_model("unet_3ch", pretrained_backbone=False)
    epoch, metrics = load_ckpt(str(checkpoint), restored)
    assert epoch == 7
    assert metrics == {"psnr": 28.5}


def test_early_stopping():
    stopping = EarlyStopping(patience=2, min_delta=0.01)
    assert not stopping(1.0)
    assert not stopping(0.9)
    assert stopping(0.8)


def test_validation_split_disables_augmentation_and_is_stable():
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.augment = True
            self.input_files = [f"root/subset/trainA/{index}.png" for index in range(20)]

        def __len__(self):
            return len(self.input_files)

        def __getitem__(self, index):
            return index, self.augment

    first_train, first_val = _split_train_validation(DummyDataset(), seed=42)
    second_train, second_val = _split_train_validation(DummyDataset(), seed=42)
    assert first_train.indices == second_train.indices
    assert first_val.indices == second_val.indices
    assert first_val.dataset.augment is False
    assert first_train.dataset.augment is True


def test_amp_gradient_overflow_skips_step_and_reduces_scale():
    class NanBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, torch.nan)

    class UnstableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            return NanBackward.apply(value * self.weight)

    class FakeScaler:
        def __init__(self):
            self.current_scale = 1024.0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            return None

        def get_scale(self):
            return self.current_scale

        def update(self, new_scale=None):
            self.current_scale = float(new_scale)

        def step(self, optimizer):
            raise AssertionError("overflowed optimizer step must be skipped")

    model = UnstableModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.ones(1, 1)
    targets = torch.zeros(1, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    scaler = FakeScaler()

    def criterion(prediction, _target):
        return prediction.mean(), {"l1": 1.0}

    train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        torch.device("cpu"),
        scaler=scaler,
    )

    assert model.weight.item() == 1.0
    assert scaler.get_scale() == 512.0
