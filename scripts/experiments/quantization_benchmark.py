"""
quantization_benchmark.py
=========================
Comprehensive Benchmark for FP32, FP16, INT8, and INT4 Precision.

Evaluates 4 models:
  - uwlytmsv2_3ch (125k params)
  - uwlytms_3ch   (118k params)
  - uwlytv2_3ch   (27k params)
  - unet_5ch      (31.4M params)

Measures:
  1. Model Size on Disk (MB)
  2. Peak Memory Occupancy (MiB)
  3. Pure Model Inference Latency (ms/image) & Throughput (FPS)
  4. Restoration Quality on Test Benchmark: PSNR (dB), SSIM, CIEDE2000, UIQM

Usage:
------
    # Benchmark all 4 models across FP32, FP16, INT8, INT4
    python -m scripts.experiments.quantization_benchmark \\
        --data_root ./datasets/EUVP \\
        --device cuda \\
        --output_dir ./results/quant_benchmark
"""

import argparse
import copy
import csv
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tabulate import tabulate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from uwir.cli.evaluate import collect_test_pairs, TestDataset
from uwir.cli.train import PhysicsCollate, _resolve_physics_extractor, load_ckpt
from uwir.metrics import evaluate_loader
from uwir.models import ALL_MODEL_NAMES, build_model, parse_model_variant

DEFAULT_MODELS = ["uwlytmsv2_3ch", "uwlytms_3ch", "uwlytv2_3ch", "unet_5ch"]
PRECISION_FORMATS = ["fp32", "fp16", "int8", "int4"]


# ---------------------------------------------------------------------------
# Quantization Engines (Simulation & Native Conversion)
# ---------------------------------------------------------------------------

def quantize_model_weights(model: nn.Module, bits: int) -> nn.Module:
    """
    Simulate per-tensor symmetric uniform weight quantization to `bits` precision.
    Simulates exact quantization noise on weights:
        scale = max(|W|) / (2^(bits-1) - 1)
        W_q = clamp(round(W / scale), -2^(bits-1), 2^(bits-1) - 1)
        W_dequant = W_q * scale
    """
    q_model = copy.deepcopy(model)
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if param.ndim >= 2:  # weights (Conv2d, Linear), exclude 1D biases and norms
                max_val = param.abs().max()
                scale = (max_val / qmax).clamp(min=1e-8)
                q_weight = torch.clamp(torch.round(param / scale), qmin, qmax)
                param.copy_(q_weight * scale)
    return q_model


def prepare_model_precision(model: nn.Module, precision: str, device: torch.device) -> nn.Module:
    """Prepare model for specific precision format."""
    precision = precision.lower()
    if precision == "fp32":
        return model.float().to(device)
    elif precision == "fp16":
        if device.type == "cpu":
            # Some CPU kernels don't support float16 Conv2d natively; use float32 fallback or keep float16 if supported
            try:
                return model.half().to(device)
            except Exception:
                print("  [WARN] CPU does not natively support FP16 Conv2d, running in FP32 simulation.")
                return model.float().to(device)
        return model.half().to(device)
    elif precision == "int8":
        # Weight-Quantized INT8 model
        q_model = quantize_model_weights(model, bits=8)
        return q_model.to(device)
    elif precision == "int4":
        # Weight-Quantized INT4 model
        q_model = quantize_model_weights(model, bits=4)
        return q_model.to(device)
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def compute_model_size_mb(model: nn.Module, precision: str) -> float:
    """Compute exact theoretical model size in Megabytes based on parameter bitwidth."""
    total_params = sum(p.numel() for p in model.parameters())
    bits_map = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}
    bits = bits_map.get(precision.lower(), 32)
    size_bytes = total_params * (bits / 8.0)
    return size_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Precision-Aware DataLoader & Evaluation
# ---------------------------------------------------------------------------

class PrecisionCollateWrapper:
    def __init__(self, base_collate, precision: str):
        self.base_collate = base_collate
        self.precision = precision.lower()

    def __call__(self, batch):
        inps, gts = self.base_collate(batch)
        if self.precision == "fp16":
            inps = inps.half()
        return inps, gts


def evaluate_precision_model(
    model_name: str,
    precision: str,
    checkpoint_path: str | None,
    data_root: str,
    device: torch.device,
    max_eval_samples: int | None = None,
) -> dict:
    """Evaluate one model in one specific precision format."""
    _, in_channels, physics_mode = parse_model_variant(model_name)
    physics_extractor = _resolve_physics_extractor("udcp") if physics_mode != "none" else None

    # 1. Build & load base model
    base_model = build_model(model_name, pretrained_backbone=False)
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_ckpt(checkpoint_path, base_model, device="cpu")
        print(f"  [Loaded checkpoint] {checkpoint_path}")
    else:
        print(f"  [Using fresh model weights (profiling mode)]")

    # 2. Convert to target precision
    model = prepare_model_precision(base_model, precision, device)
    model.eval()

    # 3. Model size on disk
    size_mb = compute_model_size_mb(base_model, precision)
    n_params = sum(p.numel() for p in base_model.parameters())

    # 4. Measure pure inference latency & peak VRAM
    dummy_input = torch.rand(1, in_channels, 256, 256).to(device)
    if precision == "fp16":
        dummy_input = dummy_input.half()

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    timed_runs = 20
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(timed_runs):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_mem_mb = size_mb

    latency_ms = (time.perf_counter() - t_start) / timed_runs * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    # 5. Measure restoration metrics on test set
    test_pairs = collect_test_pairs(data_root)
    if not test_pairs:
        psnr, ssim, ciede, uiqm = 0.0, 0.0, 0.0, 0.0
    else:
        test_ds = TestDataset(test_pairs, img_size=256)
        collate_base = PhysicsCollate(physics_mode, physics_extractor)
        collate_fn = PrecisionCollateWrapper(collate_base, precision)
        loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


        eval_desc = f"[{model_name}|{precision.upper()}] Test"
        metrics, n = evaluate_loader(model, loader, device, max_samples=max_eval_samples, desc=eval_desc)
        psnr = metrics.get("psnr", 0.0)
        ssim = metrics.get("ssim", 0.0)
        ciede = metrics.get("ciede2000", 0.0)
        uiqm = metrics.get("uiqm", 0.0)

    return {
        "model": model_name,
        "precision": precision.upper(),
        "params": n_params,
        "size_mb": round(size_mb, 3),
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "peak_mem_mb": round(peak_mem_mb, 2),
        "psnr": round(psnr, 3),
        "ssim": round(ssim, 4),
        "ciede2000": round(ciede, 3),
        "uiqm": round(uiqm, 3),
    }


# ---------------------------------------------------------------------------
# CLI & Runner
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Precision Benchmark: FP32, FP16, INT8, INT4.")
    p.add_argument("--data_root", default="./datasets/EUVP", help="Path to EUVP dataset root.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of model variants to test.")
    p.add_argument("--precisions", nargs="+", default=PRECISION_FORMATS, help="List of precisions: fp32 fp16 int8 int4.")
    p.add_argument("--checkpoints_dir", default="./checkpoints", help="Directory with saved checkpoints.")
    p.add_argument("--output_dir", default="./results/quantization_benchmark", help="Directory to save CSV report.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device.")
    p.add_argument("--max_samples", type=int, default=50, help="Max test samples to evaluate for metrics (None for all 515).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("  MULTI-PRECISION BENCHMARK: FP32 vs FP16 vs INT8 vs INT4")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"Models       : {args.models}")
    print(f"Precisions   : {[p.upper() for p in args.precisions]}")
    print(f"Dataset      : {args.data_root}")
    print(f"Max Samples  : {args.max_samples} test images")
    print(f"Output Dir   : {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for model_name in args.models:
        # Search for available checkpoint if exists
        ckpt_path = None
        if os.path.exists(args.checkpoints_dir):
            for root, _, files in os.walk(args.checkpoints_dir):
                if model_name in root and "best_model.pth" in files:
                    ckpt_path = os.path.join(root, "best_model.pth")
                    break

        print(f"\n---> Benchmarking Model: {model_name} (Checkpoint: {ckpt_path or 'Initialised'})")
        for prec in args.precisions:
            res = evaluate_precision_model(
                model_name=model_name,
                precision=prec,
                checkpoint_path=ckpt_path,
                data_root=args.data_root,
                device=device,
                max_eval_samples=args.max_samples,
            )
            rows.append(res)

    # -----------------------------------------------------------------------
    # Summary Display & Export
    # -----------------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"precision_benchmark_{stamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 95)
    print("  MULTI-PRECISION QUANTIZATION BENCHMARK REPORT")
    print("=" * 95)
    table_data = [
        [
            r["model"],
            r["precision"],
            f"{r['size_mb']:.2f} MB",
            f"{r['latency_ms']:.2f} ms",
            f"{r['fps']:.1f}",
            f"{r['peak_mem_mb']:.1f} MB",
            f"{r['psnr']:.2f} dB",
            f"{r['ssim']:.4f}",
            f"{r['uiqm']:.3f}",
        ]
        for r in rows
    ]
    headers = ["Model", "Format", "Size", "Latency", "FPS", "Peak VRAM", "PSNR", "SSIM", "UIQM"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print(f"\n[INFO] Detailed CSV report saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
