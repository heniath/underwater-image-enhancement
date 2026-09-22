"""Adapter contract that standardizes orchestration without standardizing methods."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from torch import nn


class ReferenceMethodAdapter(ABC):
    method_name = "abstract"
    display_name = "Abstract"

    def __init__(self, device: str | torch.device, *, amp: bool = True) -> None:
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.modules: dict[str, nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.method_state: dict[str, Any] = {}
        self._stepped_optimizer_ids: set[int] = set()
        self.build()

    @abstractmethod
    def build(self) -> None:
        """Construct all inference and training-only state."""

    @abstractmethod
    def train_step(
        self, batch: dict[str, Any], *, accumulation_steps: int = 1, update: bool = True
    ) -> dict[str, float]:
        """Run one method-specific microbatch and optionally apply optimizer updates."""

    @abstractmethod
    def _inference_native(self, degraded: torch.Tensor) -> torch.Tensor:
        """Run the method's native inference path and return nominal [0,1] RGB."""

    def autocast(self):
        return torch.autocast(device_type="cuda", enabled=True) if self.amp else nullcontext()

    def train(self) -> None:
        for module in self.modules.values():
            module.train()

    def eval(self) -> None:
        for module in self.modules.values():
            module.eval()

    def prepare_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            batch["degraded"].to(self.device, non_blocking=True),
            batch["reference"].to(self.device, non_blocking=True),
        )

    @torch.no_grad()
    def inference(self, degraded: torch.Tensor) -> torch.Tensor:
        degraded = degraded.to(self.device)
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW RGB input, got {tuple(degraded.shape)}")
        with self.autocast():
            prediction = self._inference_native(degraded)
        prediction = prediction.float()
        if prediction.shape != degraded.shape:
            raise ValueError(
                f"{self.display_name} changed spatial/image shape: "
                f"{tuple(degraded.shape)} -> {tuple(prediction.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{self.display_name} produced non-finite inference output")
        return prediction.clamp(0.0, 1.0)

    def zero_grad(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def step_schedulers(self) -> None:
        for scheduler in self.schedulers.values():
            optimizer = getattr(scheduler, "optimizer", None)
            if optimizer is None or id(optimizer) in self._stepped_optimizer_ids:
                scheduler.step()
        self._stepped_optimizer_ids.clear()

    def record_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Record a completed update so its epoch scheduler may advance."""
        self._stepped_optimizer_ids.add(id(optimizer))

    def network_benchmark_call(self, degraded: torch.Tensor) -> tuple[nn.Module, tuple[Any, ...]]:
        """Return the inference network and already-prepared network inputs."""
        module = self.modules.get("generator", self.modules.get("restoration"))
        if module is None:
            raise RuntimeError(f"{self.display_name} did not register an inference module")
        return module, (degraded,)

    def state_dict(self) -> dict[str, Any]:
        return {
            "modules": {name: module.state_dict() for name, module in self.modules.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "schedulers": {
                name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
            },
            "scaler": self.scaler.state_dict(),
            "method_state": self.method_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, value in state["modules"].items():
            self.modules[name].load_state_dict(value)
        for name, value in state.get("optimizers", {}).items():
            self.optimizers[name].load_state_dict(value)
        for name, value in state.get("schedulers", {}).items():
            self.schedulers[name].load_state_dict(value)
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.method_state = state.get("method_state", {})

    @classmethod
    @abstractmethod
    def provenance(cls) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def training_config(cls) -> dict[str, Any]:
        pass

    @classmethod
    def config_id(cls) -> str:
        commit = cls.provenance().get("source_commit_sha", "unknown")
        return f"{cls.method_name}:{commit}"


class VGGFeatureLoss(nn.Module):
    """Frozen VGG feature MSE with an explicit, recorded ImageNet dependency."""

    def __init__(self, architecture: str, stop: int) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, VGG19_Weights, vgg16, vgg19

        if architecture == "vgg19":
            weights, constructor = VGG19_Weights.IMAGENET1K_V1, vgg19
        else:
            weights, constructor = VGG16_Weights.IMAGENET1K_V1, vgg16
        default_checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        )
        if not default_checkpoint.exists():
            # Some cluster homes are read-only. Keep missing auxiliary weights in a
            # configurable benchmark cache while still reusing existing global weights.
            cache_root = Path(
                os.environ.get(
                    "UWIR_TORCH_HOME",
                    Path(__file__).resolve().parents[3]
                    / "outputs"
                    / "reference_methods"
                    / "cache"
                    / "torch",
                )
            )
            cache_root.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(cache_root / "hub"))
        features = constructor(weights=weights).features[:stop]
        self.features = features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.features(prediction), self.features(target))


def backward_and_step(
    adapter: ReferenceMethodAdapter,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    accumulation_steps: int,
    update: bool,
) -> None:
    adapter.scaler.scale(loss / accumulation_steps).backward()
    if update:
        previous_scale = adapter.scaler.get_scale()
        adapter.scaler.step(optimizer)
        adapter.scaler.update()
        # GradScaler lowers its scale when non-finite gradients cause it to
        # skip optimizer.step(). Do not advance the LR schedule in that case.
        if adapter.scaler.get_scale() >= previous_scale:
            adapter.record_optimizer_step(optimizer)
        optimizer.zero_grad(set_to_none=True)
