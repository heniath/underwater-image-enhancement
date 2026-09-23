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
    if "rng_state" in payload and isinstance(payload["rng_state"], dict):
        rng = payload["rng_state"]
        if "torch_cpu" in rng and isinstance(rng["torch_cpu"], torch.Tensor):
            rng["torch_cpu"] = rng["torch_cpu"].to(device="cpu", dtype=torch.uint8)
        if "torch_cuda" in rng and isinstance(rng["torch_cuda"], (list, tuple)):
            rng["torch_cuda"] = [
                t.to(device="cpu", dtype=torch.uint8) if isinstance(t, torch.Tensor) else t
                for t in rng["torch_cuda"]
            ]
    return payload
