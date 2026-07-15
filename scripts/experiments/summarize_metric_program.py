"""Summarize paired-seed metric-program results and apply promotion thresholds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SEED_PATTERN = re.compile(r"_full_seed(\d+)")
METRICS = ("psnr", "ssim", "ciede2000", "uciqe", "uiqm", "inference_ms_per_img")


def _load(path: Path):
    return json.loads(path.read_text())["runs"]


def _rows(runs: dict, model: str, metric_key: str = "test_metrics"):
    rows = {}
    for run_name, result in runs.items():
        match = SEED_PATTERN.search(run_name)
        if result.get("model_name") == model and match and result.get(metric_key):
            rows[int(match.group(1))] = result[metric_key]
    return rows


def _summary(rows: dict[int, dict]):
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows.values()])),
            "std": float(np.std([row[metric] for row in rows.values()], ddof=1)),
        }
        for metric in METRICS
        if all(metric in row for row in rows.values())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("winner")
    args = parser.parse_args()

    baseline = "unet_3ch"
    report = {"baseline": baseline, "candidate": args.winner, "benchmarks": {}}
    promotion_values = {}
    for benchmark in ("euvp", "uieb"):
        runs = _load(args.root / "results" / benchmark / "test_results_all.json")
        base_native = _rows(runs, baseline)
        candidate_native = _rows(runs, args.winner)
        base_legacy = _rows(runs, baseline, "test_metrics_legacy_256")
        candidate_legacy = _rows(runs, args.winner, "test_metrics_legacy_256")
        seeds = sorted(set(base_native) & set(candidate_native))
        if seeds != [42, 123, 3407]:
            raise RuntimeError(f"{benchmark}: expected paired seeds 42, 123, 3407; got {seeds}")
        deltas = {
            metric: [candidate_native[seed][metric] - base_native[seed][metric] for seed in seeds]
            for metric in ("psnr", "ssim", "ciede2000")
        }
        mean_deltas = {metric: float(np.mean(values)) for metric, values in deltas.items()}
        report["benchmarks"][benchmark] = {
            "seeds": seeds,
            "native": {
                "baseline": _summary(base_native),
                "candidate": _summary(candidate_native),
                "paired_deltas": deltas,
                "mean_deltas": mean_deltas,
            },
            "legacy_256": {
                "baseline": _summary(base_legacy),
                "candidate": _summary(candidate_legacy),
            },
        }
        promotion_values[benchmark] = mean_deltas

    rules = {
        "euvp_psnr": promotion_values["euvp"]["psnr"] >= 0.20,
        "euvp_ssim": promotion_values["euvp"]["ssim"] >= 0.002,
        "uieb_psnr": promotion_values["uieb"]["psnr"] >= 0.10,
        "uieb_ssim": promotion_values["uieb"]["ssim"] >= 0.0,
        "euvp_ciede2000": promotion_values["euvp"]["ciede2000"] <= 0.10,
        "uieb_ciede2000": promotion_values["uieb"]["ciede2000"] <= 0.10,
    }
    report["promotion_rules"] = rules
    report["promoted"] = all(rules.values())

    histories = {}
    for path in (args.root / "checkpoints").glob("*/training_history.json"):
        history = json.loads(path.read_text())
        match = SEED_PATTERN.search(path.parent.name)
        if match:
            key = f"{history['_meta']['model']}_seed{match.group(1)}"
            histories[key] = {
                "training_time_min": history["_meta"].get("training_time_min"),
                "peak_gpu_memory_mb": history["_meta"].get("peak_gpu_memory_mb"),
            }
    report["resources"] = histories

    output = args.root / "results" / "promotion_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Promotion report: {output}")


if __name__ == "__main__":
    main()
