"""One guarded metric path shared by every reference method."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from uwir.metrics import compute_ciede2000, compute_psnr, compute_ssim, compute_uciqe, compute_uiqm

METRIC_KEYS = ("psnr", "ssim", "ciede2000", "uciqe", "uiqm")


def _rgb01(value: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be HxWx3 or 3xHxW RGB, got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype in [0,1], got {value.dtype}")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")
    minimum, maximum = float(value.min()), float(value.max())
    if minimum < -1e-6 or maximum > 1 + 1e-6:
        raise ValueError(f"{name} range must be [0,1], got [{minimum}, {maximum}]")
    return value.astype(np.float32, copy=False).clip(0, 1)


def evaluate_pair(prediction: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray):
    prediction, target = _rgb01(prediction, "prediction"), _rgb01(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(f"Metric inputs differ: {prediction.shape} vs {target.shape}")
    return {
        "psnr": compute_psnr(prediction, target),
        "ssim": compute_ssim(prediction, target),
        "ciede2000": compute_ciede2000(prediction, target),
        "uciqe": compute_uciqe(prediction),
        "uiqm": compute_uiqm(prediction),
    }


@torch.no_grad()
def evaluate_adapter(adapter, loader, *, max_samples: int | None = None) -> dict[str, Any]:
    adapter.eval()
    totals: dict[str, list[float]] = defaultdict(list)
    per_image = []
    seen = 0
    for batch in loader:
        degraded = batch["degraded"].to(adapter.device)
        target = batch["reference"]
        prediction = adapter.inference(degraded).cpu()
        for index in range(prediction.shape[0]):
            metrics = evaluate_pair(prediction[index], target[index])
            identity = batch["identity"][index]
            per_image.append({"identity": identity, **metrics})
            for name, value in metrics.items():
                totals[name].append(value)
            seen += 1
            if max_samples is not None and seen >= max_samples:
                break
        if max_samples is not None and seen >= max_samples:
            break
    if not seen:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        **{name: float(np.mean(values, dtype=np.float64)) for name, values in totals.items()},
        "num_samples": seen,
        "per_image": per_image,
    }
