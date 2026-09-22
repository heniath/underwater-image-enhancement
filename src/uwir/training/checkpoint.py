"""Atomic, multi-network reference benchmark checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    required = {"adapter", "epoch", "best_val_psnr", "best_epoch", "history", "run_config"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: missing {sorted(missing)}")
    return payload
