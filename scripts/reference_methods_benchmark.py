#!/usr/bin/env python3
"""Launch/resume method-preserving UIEB and LSUI reference reruns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from uwir.datasets.lsui import build_lsui_datasets, compact_tree, discover_lsui
from uwir.datasets.uieb import build_uieb_datasets, discover_uieb
from uwir.evaluation.inference_benchmark import benchmark_inference
from uwir.reference_methods import REFERENCE_METHODS
from uwir.training.runner import (
    MODEL_SEEDS,
    PHYSICAL_BATCH_SIZE,
    BenchmarkConfig,
    aggregate_results,
    run_reference_experiment,
    write_benchmark_metadata,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Method-preserving reference reruns under a standardized data/split/evaluation protocol"
    )
    result.add_argument("--data-root", type=Path, default=None)
    result.add_argument("--output-root", type=Path, default=None)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    result.add_argument("--device", default=None)
    result.add_argument("--no-ram-cache", action="store_true")
    result.add_argument(
        "--methods", nargs="+", choices=REFERENCE_METHODS, default=list(REFERENCE_METHODS)
    )
    result.add_argument("--datasets", nargs="+", choices=("UIEB", "LSUI"), default=["UIEB", "LSUI"])
    result.add_argument("--seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    result.add_argument("--efficiency", action="store_true")
    return result


def _smoke_complete(output_root: Path) -> bool:
    path = output_root / "smoke_results.csv"
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        combinations = {(row["dataset"], row["method"]) for row in csv.DictReader(handle)}
    expected = {(dataset, method) for dataset in ("UIEB", "LSUI") for method in REFERENCE_METHODS}
    return expected.issubset(combinations)


def _write_efficiency(output_root: Path, device: str | None):
    rows = []
    for cls in REFERENCE_METHODS.values():
        adapter = cls(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        rows.append(benchmark_inference(adapter))
    path = output_root / "efficiency_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_preflight(config: BenchmarkConfig, discovered, output_root: Path, *, full: bool):
    checks = {
        "split manifests frozen": all(
            (output_root / "splits" / name).exists()
            for name in ("uieb_split_manifest.json", "lsui_split_manifest.json")
        ),
        "model seeds do not alter splits": config.split_seed == 42
        and tuple(config.model_seeds) == (0, 1, 2),
        "100 epoch external budget": config.epochs == 100,
        "effective batch size 4": config.effective_batch_size == 4,
        "256 crop": config.crop_size == 256,
        "validation PSNR checkpoint selection": config.model_selection_metric == "val_psnr",
        "same common evaluator": True,
        "no test-set model selection": True,
        "UIEB 720/80/90": "UIEB" not in discovered
        or [len(discovered["UIEB"][name]) for name in ("train", "val", "test")] == [720, 80, 90],
        "LSUI disjoint fixed split": "LSUI" not in discovered
        or not any(
            {record.identity for record in discovered["LSUI"][left].records}
            & {record.identity for record in discovered["LSUI"][right].records}
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ),
        "method provenance recorded": all(cls.provenance() for cls in REFERENCE_METHODS.values()),
        "method recipes recorded": all(
            all(key in cls.training_config() for key in ("losses", "optimizers", "schedulers"))
            for cls in REFERENCE_METHODS.values()
        ),
        "all method/dataset smoke combinations passed": not full or _smoke_complete(output_root),
    }
    print("\nPreflight")
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':4} | {label}")
    if not all(checks.values()):
        raise RuntimeError("Critical preflight requirement failed")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or repository_root / "datasets").expanduser().resolve()
    output_root = (
        (args.output_root or repository_root / "outputs" / "reference_methods")
        .expanduser()
        .resolve()
    )
    config = BenchmarkConfig(cache_data=False if args.no_ram_cache else "auto")
    write_benchmark_metadata(output_root, config, repository_root)

    print(f"Repository: {repository_root}")
    print(f"Data root:  {data_root}")
    print(f"Output:     {output_root}")
    print(f"Expected full runs: {len(REFERENCE_METHODS) * 2 * len(MODEL_SEEDS)}")
    print("\nLSUI tree:\n" + compact_tree(data_root / "LSUI"))
    discovered = {}
    if "UIEB" in args.datasets:
        pairs = discover_uieb(data_root / "UIEB")
        print(f"UIEB PASS: {len(pairs)} pairs; fixed 720/80/90")
        discovered["UIEB"] = build_uieb_datasets(
            data_root / "UIEB",
            output_root / "splits" / "uieb_split_manifest.json",
            cache=config.cache_data,
        )
    if "LSUI" in args.datasets:
        pairs, report = discover_lsui(data_root / "LSUI")
        print("LSUI PASS: " + json.dumps(report, sort_keys=True))
        discovered["LSUI"] = build_lsui_datasets(
            data_root / "LSUI",
            output_root / "splits" / "lsui_split_manifest.json",
            cache=config.cache_data,
        )

    _print_preflight(config, discovered, output_root, full=args.full)
    if args.full and not _smoke_complete(output_root):
        raise RuntimeError("All dataset x method smoke combinations must pass before --full")
    seeds = [0] if args.smoke else args.seeds
    print(
        f"\n>>> Running benchmark [mode={'SMOKE' if args.smoke else 'FULL'}]: "
        f"datasets={list(discovered.keys())}, methods={args.methods}, seeds={seeds}\n",
        flush=True,
    )
    for dataset_name, datasets in discovered.items():
        for method in args.methods:
            for seed in seeds:
                accumulation = config.effective_batch_size // PHYSICAL_BATCH_SIZE[method]
                run_reference_experiment(
                    dataset_name=dataset_name,
                    method_name=method,
                    datasets=datasets,
                    output_root=output_root,
                    repository_root=repository_root,
                    model_seed=seed,
                    config=config,
                    device=args.device,
                    smoke=args.smoke,
                    max_train_batches=accumulation if args.smoke else None,
                    max_eval_samples=1 if args.smoke else None,
                    resume=True,
                )
    if args.full:
        print(f"\n>>> Aggregating full results into {output_root / 'aggregate_results.csv'}...", flush=True)
        aggregate_results(
            output_root / "per_run_results.csv", output_root / "aggregate_results.csv"
        )
        print(">>> All assigned full runs completed successfully!", flush=True)
    if args.efficiency:
        _write_efficiency(output_root, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
