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
