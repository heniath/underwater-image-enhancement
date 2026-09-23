"""Outer reference protocol: fixed data, validation selection, resume, one-shot test."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from uwir.evaluation.quality_metrics import evaluate_adapter
from uwir.reference_methods import REFERENCE_METHODS

from .checkpoint import load_checkpoint, save_checkpoint
from .seed import (
    capture_rng_state,
    dataloader_generator,
    restore_rng_state,
    seed_everything,
    seed_worker,
)

MODEL_SEEDS = (0, 1, 2)
SPLIT_SEED = 42


@dataclass(frozen=True)
class BenchmarkConfig:
    epochs: int = 100
    crop_size: int = 256
    effective_batch_size: int = 4
    split_seed: int = SPLIT_SEED
    model_seeds: tuple[int, ...] = MODEL_SEEDS
    model_selection_metric: str = "val_psnr"
    validation_frequency: int = 1
    amp: bool = True
    cache_data: bool | str = "auto"


PHYSICAL_BATCH_SIZE = {
    "funie_gan": 4,
    "lpd_net": 2,
    "ucolor": 1,
    "unet": 4,
    "water_net": 4,
    "uwformer": 1,
}


def repository_state(root: Path) -> dict[str, Any]:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()

    return {
        "git_commit_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def environment_metadata(root: Path) -> dict[str, Any]:
    result = {
        **repository_state(root),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    return result


def _json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def write_benchmark_metadata(output_root: Path, config: BenchmarkConfig, repository_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    _json(output_root / "benchmark_config.json", asdict(config))
    _json(output_root / "environment.json", environment_metadata(repository_root))
    _json(
        output_root / "method_provenance.json",
        {name: cls.provenance() for name, cls in REFERENCE_METHODS.items()},
    )
    _json(
        output_root / "method_training_configs.json",
        {name: cls.training_config() for name, cls in REFERENCE_METHODS.items()},
    )


def _loader(dataset, *, batch_size, shuffle, seed, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0 if getattr(dataset, "_cache", None) is not None else 2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=dataloader_generator(seed),
    )


def _checkpoint_payload(adapter, epoch, best_psnr, best_epoch, history, run_config):
    return {
        "adapter": adapter.state_dict(),
        "epoch": epoch,
        "best_val_psnr": best_psnr,
        "best_epoch": best_epoch,
        "history": history,
        "run_config": run_config,
        "rng_state": capture_rng_state(),
    }


def _append_result(path: Path, row: dict[str, Any]):
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            item
            for item in rows
            if not (
                item["dataset"] == str(row["dataset"])
                and item["method"] == str(row["method"])
                and int(item["seed"]) == int(row["seed"])
            )
        ]
    rows.append(row)
    fieldnames = list(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_reference_experiment(
    *,
    dataset_name: str,
    method_name: str,
    datasets: dict[str, Any],
    output_root: str | Path,
    repository_root: str | Path,
    model_seed: int,
    config: BenchmarkConfig | None = None,
    device: str | torch.device | None = None,
    smoke: bool = False,
    max_train_batches: int | None = None,
    max_eval_samples: int | None = None,
    resume: bool = True,
):
    config = config or BenchmarkConfig()
    if method_name not in REFERENCE_METHODS:
        raise ValueError(f"Unknown reference method {method_name!r}")
    if config.split_seed != SPLIT_SEED:
        raise ValueError("Reference split seed is frozen at 42")
    seed_everything(model_seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    physical = PHYSICAL_BATCH_SIZE[method_name]
    if config.effective_batch_size % physical:
        raise ValueError("Effective batch size must be divisible by physical batch size")
    accumulation = config.effective_batch_size // physical
    epochs = 1 if smoke else config.epochs
    run_dir = (
        Path(output_root)
        / ("smoke" if smoke else "")
        / dataset_name.upper()
        / method_name
        / f"seed_{model_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_config_path = run_dir / "run_config.json"
    if (run_dir / "test_metrics.json").exists() and existing_config_path.exists() and resume:
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if existing_config.get("completed"):
            print(
                f"[{'SMOKE' if smoke else 'FULL'}] {dataset_name} | {method_name} | seed={model_seed} already completed. Skipping.",
                flush=True,
            )
            return json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    adapter = REFERENCE_METHODS[method_name](device, amp=config.amp)
    repo = repository_state(Path(repository_root))
    run_config = {
        "dataset": dataset_name.upper(),
        "method": method_name,
        "seed": model_seed,
        "split_seed": config.split_seed,
        "epochs": epochs,
        "full_budget_epochs": config.epochs,
        "smoke": smoke,
        "crop_size": config.crop_size,
        "physical_batch_size": physical,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": config.effective_batch_size,
        "model_selection_metric": config.model_selection_metric,
        "amp_requested": config.amp,
        "amp_enabled": adapter.amp,
        "repository_git_sha": repo["git_commit_sha"],
        "repository_dirty": repo["git_dirty"],
        "method_config_id": adapter.config_id(),
        "method_provenance": adapter.provenance(),
        "method_training_config": adapter.training_config(),
        "data_cache_mode": config.cache_data,
        "ram_cache_enabled": {
            name: getattr(dataset, "_cache", None) is not None for name, dataset in datasets.items()
        },
        "completed": False,
    }
    _json(run_dir / "run_config.json", run_config)

    val_loader = _loader(datasets["val"], batch_size=1, shuffle=False, seed=model_seed)
    test_loader = _loader(datasets["test"], batch_size=1, shuffle=False, seed=model_seed)
    start_epoch, best_psnr, best_epoch, history = 1, -float("inf"), 0, []
    last_path = run_dir / "last_model.pth"
    if resume and last_path.exists():
        state = load_checkpoint(last_path, map_location=device)
        adapter.load_state_dict(state["adapter"])
        start_epoch, best_psnr, best_epoch, history = (
            state["epoch"] + 1,
            state["best_val_psnr"],
            state["best_epoch"],
            state["history"],
        )
        if state.get("rng_state"):
            restore_rng_state(state["rng_state"])

    mode_str = "SMOKE" if smoke else "FULL"
    print(
        f"\n[{mode_str}] >>> Starting {dataset_name} | {method_name} | seed={model_seed} | "
        f"epochs={epochs} (start={start_epoch}) | physical_bs={physical} | device={device}",
        flush=True,
    )

    adapter.zero_grad()
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        # An epoch-specific seed makes sample order and worker-side paired
        # augmentation invariant to whether earlier epochs ran in this process.
        train_loader = _loader(
            datasets["train"],
            batch_size=physical,
            shuffle=True,
            seed=model_seed + epoch,
            drop_last=not smoke,
        )
        adapter.train()
        limit_batches = max_train_batches
        if limit_batches is None and not smoke and accumulation > 1:
            limit_batches = (len(train_loader) // accumulation) * accumulation

        total_batches = limit_batches if limit_batches is not None else len(train_loader)
        logs = []
        for batch_index, batch in enumerate(train_loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            update = (batch_index + 1) % accumulation == 0
            step_log = adapter.train_step(batch, accumulation_steps=accumulation, update=update)
            logs.append(step_log)
            if not smoke and ((batch_index + 1) % 100 == 0 or (batch_index + 1) == total_batches):
                step_loss = step_log.get("loss", 0.0)
                print(
                    f"  [{dataset_name}-{method_name}-s{model_seed}] Ep {epoch:3d}/{epochs:3d} | "
                    f"Batch {batch_index + 1:4d}/{total_batches:4d} | step_loss={step_loss:.4f}",
                    flush=True,
                )
        if not logs:
            raise ValueError("No training batches were processed")
        # Smoke mode may intentionally cap before an accumulation boundary; force callers to provide enough batches.
        if len(logs) % accumulation:
            if not smoke:
                pass
            else:
                raise ValueError(
                    f"Processed {len(logs)} microbatches, not divisible by accumulation={accumulation}"
                )
        adapter.step_schedulers()
        validation = evaluate_adapter(adapter, val_loader, max_samples=max_eval_samples)
        epoch_record = {
            "epoch": epoch,
            "train": {key: float(np.mean([item[key] for item in logs])) for key in logs[0]},
            "validation": {key: value for key, value in validation.items() if key != "per_image"},
        }
        history.append(epoch_record)
        is_best = False
        if validation["psnr"] > best_psnr:
            best_psnr, best_epoch = validation["psnr"], epoch
            is_best = True
            save_checkpoint(
                run_dir / "best_model.pth",
                _checkpoint_payload(adapter, epoch, best_psnr, best_epoch, history, run_config),
            )
        save_checkpoint(
            last_path,
            _checkpoint_payload(adapter, epoch, best_psnr, best_epoch, history, run_config),
        )
        _json(run_dir / "history.json", history)

        elapsed = time.time() - epoch_t0
        train_loss = epoch_record["train"].get("loss", next(iter(epoch_record["train"].values()), 0.0))
        val_psnr = validation.get("psnr", 0.0)
        val_ssim = validation.get("ssim", 0.0)
        best_tag = " [*BEST*]" if is_best else ""
        print(
            f"[{mode_str}][{dataset_name}|{method_name}|s{model_seed}] "
            f"Epoch {epoch:3d}/{epochs:3d} ({elapsed:5.1f}s) | "
            f"train_loss={train_loss:.4f} | val_psnr={val_psnr:.2f}dB val_ssim={val_ssim:.4f} "
            f"(best={best_psnr:.2f}dB @ ep {best_epoch}){best_tag}",
            flush=True,
        )

    print(
        f"[{mode_str}][{dataset_name}|{method_name}|s{model_seed}] "
        f"Training complete. Evaluating test set with best checkpoint (ep {best_epoch})...",
        flush=True,
    )
    best_path = run_dir / "best_model.pth"
    if not best_path.exists() and last_path.exists():
        best_path = last_path
    best = load_checkpoint(best_path, map_location=device)
    adapter.load_state_dict(best["adapter"])
    test_metrics = evaluate_adapter(adapter, test_loader, max_samples=max_eval_samples)
    _json(run_dir / "test_metrics.json", test_metrics)
    run_config["completed"] = True
    run_config["best_epoch"] = best_epoch
    _json(run_dir / "run_config.json", run_config)
    print(
        f"[{mode_str}][{dataset_name}|{method_name}|s{model_seed}] "
        f"FINAL TEST RESULT: PSNR={test_metrics['psnr']:.2f}dB | SSIM={test_metrics['ssim']:.4f} | "
        f"UCIQE={test_metrics.get('uciqe', 0.0):.4f} | UIQM={test_metrics.get('uiqm', 0.0):.4f}\n",
        flush=True,
    )
    row = {
        "dataset": dataset_name.upper(),
        "method": method_name,
        "seed": model_seed,
        "best_epoch": best_epoch,
        **{
            f"test_{key}": test_metrics[key]
            for key in ("psnr", "ssim", "ciede2000", "uciqe", "uiqm")
        },
        "epochs": epochs,
        "physical_batch_size": physical,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": config.effective_batch_size,
        "crop_size": config.crop_size,
        "method_config_id": adapter.config_id(),
        "git_commit_sha": repo["git_commit_sha"],
    }
    _append_result(
        Path(output_root) / ("smoke_results.csv" if smoke else "per_run_results.csv"), row
    )
    return test_metrics


def aggregate_results(per_run_path: str | Path, output_path: str | Path):
    with Path(per_run_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    output = []
    for (dataset, method), items in sorted(groups.items()):
        record: dict[str, Any] = {"dataset": dataset, "method": method, "num_seeds": len(items)}
        for metric in ("psnr", "ssim", "ciede2000", "uciqe", "uiqm"):
            values = np.array([float(item[f"test_{metric}"]) for item in items], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output[0]) if output else ["dataset", "method"]
        )
        writer.writeheader()
        writer.writerows(output)
    return output


def evaluate_cross_dataset(
    *,
    source_dataset: str,
    target_dataset: str,
    method_name: str,
    model_seed: int,
    target_test_dataset,
    output_root: str | Path,
    device: str | torch.device | None = None,
):
    """Evaluate an existing best checkpoint on the other dataset; never trains."""
    output_root = Path(output_root)
    checkpoint = (
        output_root / source_dataset.upper() / method_name / f"seed_{model_seed}" / "best_model.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Cross-dataset evaluation requires existing checkpoint: {checkpoint}"
        )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    adapter = REFERENCE_METHODS[method_name](device)
    adapter.load_state_dict(load_checkpoint(checkpoint, map_location=device)["adapter"])
    loader = _loader(target_test_dataset, batch_size=1, shuffle=False, seed=model_seed)
    metrics = evaluate_adapter(adapter, loader)
    path = output_root / "cross_dataset_results.csv"
    row = {
        "source_dataset": source_dataset.upper(),
        "target_dataset": target_dataset.upper(),
        "method": method_name,
        "seed": model_seed,
        **{key: metrics[key] for key in ("psnr", "ssim", "ciede2000", "uciqe", "uiqm")},
    }
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [
            item
            for item in rows
            if not (
                item["source_dataset"] == row["source_dataset"]
                and item["target_dataset"] == row["target_dataset"]
                and item["method"] == method_name
                and int(item["seed"]) == model_seed
            )
        ]
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    return metrics
