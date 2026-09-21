#!/usr/bin/env python3
"""
ablation_loss.py
----------------
Systematic Ablation Study on Loss Functions for Underwater Image Enhancement.

Compares the quantitative contribution of candidate loss functions against baseline:
  - 01_l1_baseline     : Pure L1 pixel loss (reference baseline)
  - 02_plus_perceptual : Baseline + VGG-16 Perceptual loss
  - 03_plus_ssim       : Baseline + SSIM loss
  - 04_plus_color      : Baseline + Color Angle (cosine distance) loss
  - 05_plus_wavelet    : Baseline + 2D Haar Wavelet domain loss
  - 06_plus_edge       : Baseline + Sobel-based Edge loss
  - 07_plus_tv         : Baseline + Total Variation (TV) loss
  - 08_lvw_mobileie    : MobileIE Local Variance-Weighted (LVW) loss
  - 09_plus_uiqm       : Baseline + Differentiable UIQM loss
  - 10_best_combo      : Balanced multi-component composite loss

Usage:
  # Run loss ablation on SGMA-Net with EUVP in WSL:
  python scripts/experiments/ablation_loss.py --model sgmanet_3ch --dataset euvp --nEpochs 20

  # Run specific loss experiments:
  python scripts/experiments/ablation_loss.py --model sgmanet_3ch --experiments 01_l1_baseline 06_plus_edge 08_lvw_mobileie
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, ToTensor

# Adjust project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from uwir.cli.train import (
    EarlyStopping,
    PhysicsCollate,
    _resolve_physics_extractor,
    _split_train_validation,
    build_scheduler,
    load_ckpt,
    save_ckpt,
    train_epoch,
    val_loss_epoch,
)
from uwir.cli.evaluate import TestDataset, collect_test_pairs
from uwir.data.datasets import UIEBDataset
from uwir.data.factory import get_euvp_training_set, get_uieb_training_set
from uwir.losses import CompositeLoss
from uwir.metrics import evaluate_loader
from uwir.models import ALL_MODEL_NAMES, build_model, parse_model_variant


# ---------------------------------------------------------------------------
# Loss Configurations Matrix
# ---------------------------------------------------------------------------

LOSS_EXPERIMENTS: dict[str, dict] = {
    "01_l1_baseline": {
        "desc": "Baseline L1 pixel loss",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
            "use_charbonnier": False,
        },
    },
    "02_plus_perceptual": {
        "desc": "L1 + VGG Perceptual (lambda=0.1)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.1,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
        },
    },
    "03_plus_ssim": {
        "desc": "L1 + SSIM (lambda=0.5)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.5,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
        },
    },
    "04_plus_color": {
        "desc": "L1 + Color Angle (lambda=0.2)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.2,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
        },
    },
    "05_plus_wavelet": {
        "desc": "L1 + Haar Wavelet (lambda=0.1)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.1,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
        },
    },
    "06_plus_edge": {
        "desc": "L1 + Sobel Edge (lambda=0.1)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.1,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
        },
    },
    "07_plus_tv": {
        "desc": "L1 + Total Variation (lambda=0.01)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.01,
            "lambda_uiqm": 0.0,
        },
    },
    "08_lvw_mobileie": {
        "desc": "MobileIE Local Variance-Weighted (LVW) only",
        "params": {
            "lambda_l1": 0.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 1.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.0,
        },
    },
    "09_plus_uiqm": {
        "desc": "L1 + Differentiable UIQM (lambda=0.01)",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.0,
            "lambda_ssim": 0.0,
            "lambda_color": 0.0,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.0,
            "lambda_tv": 0.0,
            "lambda_uiqm": 0.01,
        },
    },
    "10_best_combo": {
        "desc": "Balanced Combo: L1 + Perc + SSIM + Edge + Color",
        "params": {
            "lambda_l1": 1.0,
            "lambda_perc": 0.1,
            "lambda_ssim": 0.5,
            "lambda_color": 0.1,
            "lambda_wavelet": 0.0,
            "lambda_lvw": 0.0,
            "lambda_edge": 0.1,
            "lambda_tv": 0.005,
            "lambda_uiqm": 0.0,
        },
    },
}

METRIC_KEYS = ("psnr", "ssim", "ciede2000", "uciqe", "uiqm")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ablation Study on Loss Functions for SGMA-Net / UWIR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="sgmanet_3ch",
        choices=ALL_MODEL_NAMES,
        help="Model variant to train in the ablation.",
    )
    p.add_argument(
        "--dataset",
        default="euvp",
        choices=["euvp", "uieb"],
        help="Dataset to run ablation on.",
    )
    p.add_argument("--data_euvp", default="./datasets/EUVP", help="EUVP dataset root.")
    p.add_argument("--data_uieb", default="./datasets/UIEB", help="UIEB dataset root.")
    p.add_argument("--euvp_subset", default="all", help="EUVP subset.")
    p.add_argument(
        "--experiments",
        nargs="+",
        default=list(LOSS_EXPERIMENTS.keys()),
        choices=list(LOSS_EXPERIMENTS.keys()),
        help="List of loss configurations to run.",
    )
    p.add_argument("--nEpochs", type=int, default=20, help="Training epochs per experiment.")
    p.add_argument("--batchSize", type=int, default=16, help="Mini-batch size.")
    p.add_argument("--cropSize", type=int, default=256, help="Spatial resolution.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--threads", type=int, default=4, help="DataLoader threads.")
    p.add_argument(
        "--prior_method",
        default="udcp",
        choices=["udcp"],
        help="Physics prior for 4ch/5ch variants.",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="./checkpoints/ablation_loss",
        help="Directory to save checkpoints.",
    )
    p.add_argument(
        "--val_folder",
        default="./results/ablation_loss",
        help="Directory to save final comparison JSON report.",
    )
    p.add_argument(
        "--scheduler_step", type=int, default=15, help="StepLR decay step size."
    )
    p.add_argument("--scheduler_gamma", type=float, default=0.5, help="StepLR decay factor.")
    p.add_argument(
        "--in_memory",
        action="store_true",
        default=False,
        help="Pre-load entire dataset images into RAM.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision (FP16/BF16) on CUDA for maximum GPU utilization.",
    )
    p.add_argument(
        "--no_amp",
        action="store_false",
        dest="amp",
        help="Disable Automatic Mixed Precision.",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs.",
    )
    return p


# ---------------------------------------------------------------------------
# Training one loss configuration
# ---------------------------------------------------------------------------


def train_single_experiment(
    exp_name: str,
    exp_cfg: dict,
    model_name: str,
    train_ds,
    val_ds,
    test_ds,
    args,
    device: torch.device,
) -> dict:
    print(f"\n{'='*75}")
    print(f"  RUNNING LOSS EXPERIMENT: {exp_name}")
    print(f"  Description: {exp_cfg['desc']}")
    print(f"  Parameters : {exp_cfg['params']}")
    print(f"{'='*75}")

    # Set deterministic seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _, in_channels, physics_mode = parse_model_variant(model_name)
    physics_extractor = _resolve_physics_extractor(args.prior_method)

    collate_tr = PhysicsCollate(physics_mode, physics_extractor)
    collate_vl = PhysicsCollate(physics_mode, physics_extractor)

    use_persistent = args.threads > 0 and device.type == "cuda"
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if args.threads > 0 else None,
        collate_fn=collate_tr,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.threads,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=use_persistent,
        prefetch_factor=2 if args.threads > 0 else None,
        collate_fn=collate_vl,
    )

    # Instantiate model
    model = build_model(model_name, pretrained_backbone=False).to(device)

    # Instantiate Criterion
    criterion_args = dict(exp_cfg["params"])
    criterion_args["device"] = device
    criterion = CompositeLoss(**criterion_args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )

    # Precision & AMP settings for maximum GPU utilization
    amp_enabled = getattr(args, "amp", True) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (amp_enabled and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    es = EarlyStopping(patience=args.early_stop_patience, mode="max")

    ckpt_exp_dir = Path(args.checkpoint_dir) / f"{model_name}_{args.dataset}_{exp_name}"
    ckpt_exp_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_exp_dir / "best_model.pth"

    best_psnr = -1.0
    best_epoch = 0

    start_time = time.perf_counter()

    for epoch in range(1, args.nEpochs + 1):
        ep_start = time.perf_counter()
        tr_loss, comps = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            desc=f"[{exp_name}] Ep {epoch:02d}/{args.nEpochs}",
        )
        vl_loss = val_loss_epoch(
            model, val_loader, criterion, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype
        )
        scheduler.step()

        # Validation PSNR / SSIM evaluation
        val_metrics, _ = evaluate_loader(
            model, val_loader, device, desc=f"[{exp_name}] Val Ep {epoch:02d}"
        )
        val_psnr = val_metrics["psnr"]
        val_ssim = val_metrics["ssim"]

        flag = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            flag = " [BEST]"
            save_ckpt(
                model,
                optimizer,
                epoch,
                {"psnr": val_psnr, "ssim": val_ssim},
                str(best_ckpt_path),
            )

        ep_duration = time.perf_counter() - ep_start
        print(
            f"Ep {epoch:02d}/{args.nEpochs:02d} | "
            f"Tr Loss: {tr_loss:.4f} | Vl Loss: {vl_loss:.4f} | "
            f"Val PSNR: {val_psnr:.3f} dB | Val SSIM: {val_ssim:.4f} | "
            f"Time: {ep_duration:.1f}s{flag}"
        )

        if es(val_psnr):
            print(f"[EARLY STOPPING] triggered at epoch {epoch}")
            break

    total_time_min = (time.perf_counter() - start_time) / 60.0

    # -----------------------------------------------------------------------
    # Final Evaluation on Test Set
    # -----------------------------------------------------------------------
    print(f"\n[EVALUATION] Loading best model (epoch {best_epoch}) on test set...")
    load_ckpt(str(best_ckpt_path), model, device=str(device))
    model.eval()

    test_loader = data.DataLoader(
        test_ds,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.threads,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_vl,
    )

    test_metrics, n_test = evaluate_loader(
        model, test_loader, device, desc=f"[{exp_name}] Eval Test"
    )

    res = {
        "exp_name": exp_name,
        "desc": exp_cfg["desc"],
        "loss_params": exp_cfg["params"],
        "best_epoch": best_epoch,
        "val_best_psnr": best_psnr,
        "training_time_min": total_time_min,
        "test_metrics": {k: float(test_metrics.get(k, 0.0)) for k in METRIC_KEYS},
    }

    print(f"Results on Test Set:")
    for k in METRIC_KEYS:
        print(f"  {k.upper():<10}: {res['test_metrics'][k]:.4f}")

    return res


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------


def main():
    parser = make_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model variant : {args.model}")
    print(f"Dataset       : {args.dataset}")
    print(f"Epochs/run    : {args.nEpochs}")
    print(f"Batch size    : {args.batchSize}")
    print(f"AMP enabled   : {args.amp}")
    print(f"In memory     : {args.in_memory}")
    print(f"Experiments   : {len(args.experiments)} to run -> {args.experiments}")

    # Load Training & Validation Datasets
    if args.dataset == "euvp":
        print(f"Loading EUVP training set from {args.data_euvp}...")
        full_train_ds = get_euvp_training_set(
            args.data_euvp,
            img_size=args.cropSize,
            subset=args.euvp_subset,
            in_memory=args.in_memory,
        )
        test_pairs = collect_test_pairs(args.data_euvp)
        if not test_pairs:
            raise FileNotFoundError(
                f"No EUVP test samples found in {args.data_euvp}/test_samples/Inp and GTr"
            )
        test_ds = TestDataset(test_pairs, img_size=args.cropSize)
    elif args.dataset == "uieb":
        print(f"Loading UIEB dataset from {args.data_uieb} (in_memory={args.in_memory})...")
        to_tensor = Compose([ToTensor()])
        uieb_train_ds = UIEBDataset(
            args.data_uieb,
            transform=to_tensor,
            augment=True,
            img_size=args.cropSize,
            in_memory=args.in_memory,
        )
        uieb_eval_ds = copy.copy(uieb_train_ds)
        uieb_eval_ds.augment = False
        total_n = len(uieb_train_ds)
        train_val_n = min(800, total_n - 90) if total_n > 90 else int(total_n * 0.8)
        full_train_ds = data.Subset(uieb_train_ds, list(range(train_val_n)))
        test_ds = data.Subset(uieb_eval_ds, list(range(train_val_n, total_n)))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_ds, val_ds = _split_train_validation(full_train_ds, seed=args.seed, fraction=0.10)
    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")
    print(f"Test samples  : {len(test_ds)}")

    all_results: dict[str, dict] = {}

    for exp_name in args.experiments:
        cfg = LOSS_EXPERIMENTS[exp_name]
        res = train_single_experiment(
            exp_name=exp_name,
            exp_cfg=cfg,
            model_name=args.model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            args=args,
            device=device,
        )
        all_results[exp_name] = res

    # -----------------------------------------------------------------------
    # Comparative Summary Report
    # -----------------------------------------------------------------------
    out_dir = Path(args.val_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"loss_ablation_{args.model}_{args.dataset}_{datetime.now():%Y%m%d_%H%M%S}.json"

    # Save full JSON report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "model": args.model,
                    "dataset": args.dataset,
                    "epochs": args.nEpochs,
                    "seed": args.seed,
                    "device": str(device),
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    # Print Formatted Markdown Comparison Table
    base_metrics = all_results.get("01_l1_baseline", {}).get("test_metrics", {})
    W = 110
    print(f"\n{'='*W}")
    print(f"  LOSS ABLATION STUDY RESULTS ({args.model.upper()} on {args.dataset.upper()})")
    print(f"{'='*W}")
    header = f"{'Experiment':<22} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'CIEDE2000':<12} | {'UCIQE':<10} | {'UIQM':<10} | {'Δ PSNR':<10}"
    print(header)
    print("-" * W)

    for exp_name, r in all_results.items():
        tm = r["test_metrics"]
        psnr = tm.get("psnr", 0.0)
        ssim = tm.get("ssim", 0.0)
        ciede = tm.get("ciede2000", 0.0)
        uciqe = tm.get("uciqe", 0.0)
        uiqm = tm.get("uiqm", 0.0)

        delta_str = "0.00 (Base)"
        if base_metrics and "psnr" in base_metrics:
            delta = psnr - base_metrics["psnr"]
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f} dB"

        row = (
            f"{exp_name:<22} | {psnr:>10.3f}   | {ssim:>8.4f} | "
            f"{ciede:>10.3f}   | {uciqe:>8.4f} | {uiqm:>8.4f} | {delta_str:>10}"
        )
        print(row)

    print("-" * W)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
