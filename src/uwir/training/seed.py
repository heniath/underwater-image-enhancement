"""Reproducibility helpers; model seeds never participate in split generation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def capture_rng_state() -> dict[str, object]:
    """Capture stochastic state needed to continue a run after a process restart."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu = state["torch_cpu"]
    if isinstance(torch_cpu, torch.Tensor):
        torch_cpu = torch_cpu.to(device="cpu", dtype=torch.uint8)
    elif torch_cpu is not None:
        torch_cpu = torch.as_tensor(torch_cpu, dtype=torch.uint8, device="cpu")
    if torch_cpu is not None:
        torch.set_rng_state(torch_cpu)
    if torch.cuda.is_available() and state.get("torch_cuda"):
        devices = torch.cuda.device_count()
        cuda_states = [
            t.to(device="cpu", dtype=torch.uint8) if isinstance(t, torch.Tensor) else t
            for t in state["torch_cuda"][:devices]
        ]
        for i, s in enumerate(cuda_states):
            torch.cuda.set_rng_state(s, i)
