#!/usr/bin/env python3
"""
scripts/experiments/evaluate_benchmarks.py
=========================================
End-to-end benchmark evaluation script for underwater image enhancement models.
Evaluates trained checkpoints on 4 test sets:
  1. EUVP test_samples Benchmark (Official test_samples Inp & GTr pairs)
  2. EUVP Scenes Held-out Test Set (218 paired test images, seed=42)
  3. EUVP Dark Benchmark (Held-out paired test split)
  4. UIEB T90 Cross-Dataset Benchmark (90 standard test pairs)

Computes the 5 core underwater metrics in the codebase:
  1. PSNR (Full-reference dB, higher better)
  2. SSIM (Full-reference [0, 1], higher better)
  3. CIEDE2000 (Full-reference color diff, lower better)
  4. UCIQE (Underwater Color Image Quality Evaluation, higher better)
  5. UIQM (Underwater Image Quality Measure, higher better)
plus inference latency (ms), parameters, and GFLOPs.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

# Ensure src is in python path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import thop
except ImportError:
    thop = None

from uwir.models import ALL_MODEL_NAMES, build_model, parse_model_variant
from uwir.metrics import (
    compute_ciede2000,
    compute_psnr,
    compute_ssim,
    compute_uciqe,
    compute_uiqm,
    evaluate_loader,
)
from uwir.cli.train import _collate_val, _resolve_physics_extractor, load_ckpt
from uwir.physics import PhysicsConfig, compute_physics_features

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def auto_detect_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """
    Detects checkpoint_dir, euvp_root, uieb_raw_dir, uieb_ref_dir across local and Kaggle environments.
    """
    ckpt_candidates = [
        Path("/kaggle/input/datasets/thung192/uwir-trained-checkpoints"),
        Path("/kaggle/input/datasets/tthanh13/uwir-trained-checkpoints"),
        Path("/kaggle/input/datasets/uwir-trained-checkpoints"),
        Path("/kaggle/input/uwir-trained-checkpoints"),
        Path("/kaggle/input/thung192/uwir-trained-checkpoints"),
        Path("/kaggle/input/tthanh13/uwir-trained-checkpoints"),
        _REPO_ROOT / "checkpoints" / "m20566_variants_v2",
        _REPO_ROOT / "scratch" / "uwir_checkpoints_dataset",
        _REPO_ROOT / "checkpoints",
        _REPO_ROOT / "kaggle_runner_ui" / "outputs" / "underwater_image_enhancement",
    ]
    ckpt_dir = find_first_existing(ckpt_candidates)
    if ckpt_dir is None:
        matches = list(Path("/kaggle/input").glob("**/best_model.pth"))
        if matches:
            ckpt_dir = matches[0].parent.parent

    # 2. EUVP Root
    euvp_candidates = [
        Path("/kaggle/input/datasets/pamuduranasinghe/euvp-dataset/EUVP"),
        Path("/kaggle/input/pamuduranasinghe/euvp-dataset/EUVP"),
        Path("/kaggle/input/euvp-dataset/EUVP"),
        Path("/kaggle/input/pamuduranasinghe/euvp-dataset"),
        Path("/kaggle/input/euvp-dataset"),
        _REPO_ROOT / "datasets" / "EUVP",
        Path("D:/Dataset/EUVP"),
    ]
    euvp_dir = find_first_existing(euvp_candidates)
    if euvp_dir is None:
        matches = list(Path("/kaggle/input").glob("**/Paired/underwater_dark"))
        if matches:
            euvp_dir = matches[0].parents[1]

    # 3. UIEB Raw & Reference
    uieb_raw_candidates = [
        Path("/kaggle/input/uieb-dataset-raw/raw-890"),
        Path("/kaggle/input/larjeck/uieb-dataset-raw/raw-890"),
        Path("/kaggle/input/datasets/larjeck/uieb-dataset-raw/raw-890"),
        Path("/kaggle/input/uieb-dataset/raw-890"),
        _REPO_ROOT / "datasets" / "UIEB" / "raw-890",
        Path("D:/Dataset/UIEB/raw-890"),
    ]
    uieb_ref_candidates = [
        Path("/kaggle/input/uieb-dataset-reference/reference-890"),
        Path("/kaggle/input/larjeck/uieb-dataset-reference/reference-890"),
        Path("/kaggle/input/datasets/larjeck/uieb-dataset-reference/reference-890"),
        Path("/kaggle/input/uieb-dataset/reference-890"),
        _REPO_ROOT / "datasets" / "UIEB" / "reference-890",
        Path("D:/Dataset/UIEB/reference-890"),
    ]
    uieb_raw = find_first_existing(uieb_raw_candidates)
    uieb_ref = find_first_existing(uieb_ref_candidates)

    if uieb_raw is None:
        raw_matches = list(Path("/kaggle/input").glob("**/raw-890"))
        if raw_matches:
            uieb_raw = raw_matches[0]
    if uieb_ref is None:
        ref_matches = list(Path("/kaggle/input").glob("**/reference-890"))
        if ref_matches:
            uieb_ref = ref_matches[0]

    return ckpt_dir, euvp_dir, uieb_raw, uieb_ref


class SimplePairDataset(data.Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: int = 256):
        self.pairs = pairs
        self.resize = transforms.Resize((img_size, img_size), antialias=True)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, gt_path = self.pairs[idx]
        inp_img = Image.open(inp_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.resize:
            inp_img = self.resize(inp_img)
            gt_img = self.resize(gt_img)

        inp_t = self.to_tensor(inp_img)
        gt_t = self.to_tensor(gt_img)
        return inp_t, gt_t, Path(inp_path).name, Path(gt_path).name


def collect_euvp_test_samples_pairs(euvp_root: Optional[Path]) -> List[Tuple[str, str]]:
    """
    Collects the official EUVP test_samples pairs (test_samples/Inp and test_samples/GTr).
    """
    candidates = []
    if euvp_root is not None:
        candidates.extend([
            euvp_root / "test_samples",
            euvp_root.parent / "test_samples",
            euvp_root / "EUVP" / "test_samples",
        ])
    test_dir = next((p for p in candidates if p.exists()), None)
    if test_dir is None and euvp_root is not None:
        matches = list(euvp_root.glob("**/test_samples"))
        if not matches and euvp_root.parent.exists():
            matches = list(euvp_root.parent.glob("**/test_samples"))
        if matches:
            test_dir = matches[0]

    if (test_dir is None or not test_dir.exists()) and Path("/kaggle/input").exists():
        matches = [p for p in Path("/kaggle/input").glob("**/*") if p.is_dir() and p.name.lower() == "test_samples"]
        if matches:
            test_dir = matches[0]

    if test_dir is None or not test_dir.exists():
        local_candidates = [
            _REPO_ROOT / "datasets" / "EUVP" / "test_samples",
            _REPO_ROOT / "datasets" / "test_samples",
            Path("D:/Dataset/EUVP/test_samples"),
        ]
        test_dir = next((p for p in local_candidates if p.exists()), None)

    if test_dir is None or not test_dir.exists():
        print(f"[WARN] EUVP test_samples folder not found under {euvp_root}")
        return []

    inp_dir = test_dir / "Inp"
    gt_dir = test_dir / "GTr"

    if not (inp_dir.exists() and gt_dir.exists()):
        for d in test_dir.iterdir():
            if d.is_dir():
                if d.name.lower() == "inp":
                    inp_dir = d
                elif d.name.lower() in ("gtr", "gt", "ground_truth", "gt_reference"):
                    gt_dir = d

    if not (inp_dir.exists() and gt_dir.exists()):
        print(f"[WARN] Inp or GTr folder not found in {test_dir}")
        return []

    gt_dict = {f.stem: f for f in gt_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    pairs = []
    for f in sorted(inp_dir.iterdir()):
        if f.suffix.lower() in IMG_EXTS and f.stem in gt_dict:
            pairs.append((str(f), str(gt_dict[f.stem])))

    if not pairs:
        inp_files = sorted([f for f in inp_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        gt_files = sorted([f for f in gt_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        if len(inp_files) == len(gt_files) and len(inp_files) > 0:
            print(f"  [INFO] Paired {len(inp_files)} EUVP test_samples by sorted order.")
            pairs = [(str(i), str(g)) for i, g in zip(inp_files, gt_files)]

    print(f"  [INFO] Found {len(pairs)} test_samples pairs from {test_dir}")
    return pairs


def collect_euvp_scenes_test_pairs(
    euvp_root: Path, test_ratio: float = 0.10, seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Collects the dedicated paired EUVP underwater_scenes held-out test set
    (10% deterministic split = 218 test pairs, identical split seed to train.py).
    """
    inp_dir = euvp_root / "Paired" / "underwater_scenes" / "trainA"
    gt_dir = euvp_root / "Paired" / "underwater_scenes" / "trainB"

    if not (inp_dir.exists() and gt_dir.exists()):
        print(f"[WARN] EUVP Scenes folders not found in {euvp_root}")
        return []

    gt_dict = {f.stem: f for f in gt_dir.iterdir() if f.suffix in IMG_EXTS}
    all_pairs = []
    for f in sorted(inp_dir.iterdir()):
        if f.suffix in IMG_EXTS and f.stem in gt_dict:
            all_pairs.append((str(f), str(gt_dict[f.stem])))

    if not all_pairs:
        return []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    test_size = max(1, int(len(all_pairs) * test_ratio))
    test_indices = sorted(indices[:test_size])
    return [all_pairs[i] for i in test_indices]


def collect_euvp_dark_test_pairs(
    euvp_root: Path, test_ratio: float = 0.20, seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Collects paired EUVP underwater_dark test set.
    Uses deterministic train/test split (default 20% held-out test split).
    """
    inp_dir = euvp_root / "Paired" / "underwater_dark" / "trainA"
    gt_dir = euvp_root / "Paired" / "underwater_dark" / "trainB"

    if not (inp_dir.exists() and gt_dir.exists()):
        print(f"[WARN] EUVP Dark folders not found in {euvp_root}")
        return []

    gt_dict = {f.stem: f for f in gt_dir.iterdir() if f.suffix in IMG_EXTS}
    all_pairs = []
    for f in sorted(inp_dir.iterdir()):
        if f.suffix in IMG_EXTS and f.stem in gt_dict:
            all_pairs.append((str(f), str(gt_dict[f.stem])))

    if not all_pairs:
        return []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    test_size = max(1, int(len(all_pairs) * test_ratio))
    test_indices = sorted(indices[:test_size])
    return [all_pairs[i] for i in test_indices]


def collect_uieb_t90_pairs(
    raw_dir: Path, ref_dir: Path, count: int = 90, seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Collects standard UIEB T90 cross-dataset test pairs.
    """
    if not (raw_dir.exists() and ref_dir.exists()):
        print(f"[WARN] UIEB raw or ref dir missing: {raw_dir}, {ref_dir}")
        return []

    ref_dict = {f.stem: f for f in ref_dir.iterdir() if f.suffix in IMG_EXTS}
    pairs = [
        (str(f), str(ref_dict[f.stem]))
        for f in sorted(raw_dir.iterdir())
        if f.suffix in IMG_EXTS and f.stem in ref_dict
    ]
    if not pairs:
        return []

    generator = np.random.default_rng(seed)
    selected = sorted(generator.choice(len(pairs), size=min(count, len(pairs)), replace=False))
    return [pairs[i] for i in selected]


def parse_model_name_from_dir(name: str) -> Optional[str]:
    if "no_hsvloss" in name or "hsv10" in name:
        return "pcf_mbconv_5ch"
    if "m20566_replight" in name:
        return "m20566_replight_3ch"
    if "m20566_lcs" in name:
        return "m20566_lcs_3ch"
    if "m20566_lmf" in name:
        return "m20566_lmf_3ch"
    clean = name.replace("fast50", "").replace("euvp", "").strip("_- ")
    normalized = clean.replace("3ch", "_3ch").replace("4ch", "_4ch").replace("5ch", "_5ch")
    for mn in sorted(ALL_MODEL_NAMES, key=len, reverse=True):
        if clean.startswith(mn) or normalized.startswith(mn) or mn in clean:
            return mn
    return None


def profile_model_complexity(model, in_channels: int, img_size: int = 256, device="cpu"):
    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6
    params_k = params / 1e3
    macs_g = None
    if thop is not None:
        try:
            dummy = torch.randn(1, in_channels, img_size, img_size).to(device)
            macs, _ = thop.profile(model, inputs=(dummy,), verbose=False)
            macs_g = macs / 1e9
        except Exception:
            pass
    return params, params_m, params_k, macs_g


def evaluate_dataset(
    model,
    pairs: List[Tuple[str, str]],
    model_name: str,
    device: torch.device,
    physics_mode: str,
    physics_extractor,
    fusion_extractor,
    dataset_name: str,
    img_size: int = 256,
    batch_size: int = 16,
) -> Dict[str, float]:
    if not pairs:
        return {}

    ds = SimplePairDataset(pairs, img_size=img_size)

    def collate_fn(batch):
        return _collate_val(batch, physics_mode, physics_extractor, fusion_extractor)

    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    metrics, count = evaluate_loader(model, loader, device, desc=f"{model_name} on {dataset_name}")
    metrics["num_evaluated"] = count
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate UWIR checkpoints on EUVP test_samples, EUVP Scenes, EUVP Dark, and UIEB T90 with 5 metrics")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Root folder containing checkpoint subdirs")
    parser.add_argument("--data_euvp", type=str, default=None, help="EUVP root (contains test_samples & Paired)")
    parser.add_argument("--uieb_raw", type=str, default=None, help="UIEB raw-890 directory")
    parser.add_argument("--uieb_ref", type=str, default=None, help="UIEB reference-890 directory")
    parser.add_argument("--out_dir", type=str, default="results/benchmarks", help="Output results directory")
    parser.add_argument("--img_size", type=int, default=256, help="Evaluation crop/resize size")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"================================================================")
    print(f"  UWIR BENCHMARK EVALUATION ENGINE (5 METRICS SUITE)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"================================================================")

    # 1. Path resolution
    det_ckpt, det_euvp, det_uieb_raw, det_uieb_ref = auto_detect_paths()
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else det_ckpt
    euvp_dir = Path(args.data_euvp) if args.data_euvp else det_euvp
    uieb_raw = Path(args.uieb_raw) if args.uieb_raw else det_uieb_raw
    uieb_ref = Path(args.uieb_ref) if args.uieb_ref else det_uieb_ref

    print(f"  Checkpoint Dir : {ckpt_dir}")
    print(f"  EUVP Root      : {euvp_dir}")
    print(f"  UIEB Raw Dir   : {uieb_raw}")
    print(f"  UIEB Ref Dir   : {uieb_ref}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Collect test sets
    euvp_test_samples_pairs = collect_euvp_test_samples_pairs(euvp_dir) if euvp_dir else []
    euvp_scenes_pairs = collect_euvp_scenes_test_pairs(euvp_dir) if euvp_dir else []
    euvp_dark_pairs = collect_euvp_dark_test_pairs(euvp_dir) if euvp_dir else []
    uieb_pairs = collect_uieb_t90_pairs(uieb_raw, uieb_ref) if (uieb_raw and uieb_ref) else []
    print(f"\n  Loaded Test Sets:")
    print(f"    1. EUVP test_samples Set : {len(euvp_test_samples_pairs)} pairs (official Inp/GTr)")
    print(f"    2. EUVP Scenes Test Set  : {len(euvp_scenes_pairs)} pairs (held-out)")
    print(f"    3. EUVP Dark Test Set    : {len(euvp_dark_pairs)} pairs (cross-subset)")
    print(f"    4. UIEB T90 Test Set     : {len(uieb_pairs)} pairs (cross-dataset)")

    # 3. Setup physics extractors
    physics_extractor = _resolve_physics_extractor("udcp")
    fusion_extractor = partial(
        compute_physics_features,
        config=PhysicsConfig(guided_filter_radius=15, guided_filter_eps=1e-3),
    )

    # 4. Find all checkpoints to evaluate
    all_runs = {}
    found_ckpts = []
    if ckpt_dir and ckpt_dir.exists():
        found_ckpts = list(ckpt_dir.rglob("best_model.pth"))
    if not found_ckpts:
        found_ckpts = list(Path("/kaggle/input").glob("**/best_model.pth"))
    if not found_ckpts and (_REPO_ROOT / "scratch" / "uwir_checkpoints_dataset").exists():
        found_ckpts = list((_REPO_ROOT / "scratch" / "uwir_checkpoints_dataset").rglob("best_model.pth"))

    if found_ckpts:
        print(f"\n  Found {len(found_ckpts)} checkpoints:")
        for p in found_ckpts:
            print(f"    * {p.parent.name} -> {p}")
    else:
        print(f"\n  [WARN] No checkpoints found!")

    results_table = []

    for p in sorted(found_ckpts):
        parent_name = p.parent.name
        model_name = parse_model_name_from_dir(parent_name)
        if not model_name:
            model_name = parse_model_name_from_dir(p.parent.parent.name)
        if not model_name:
            print(f"\n[SKIP] Cannot infer model variant from {parent_name}")
            continue

        print(f"\n----------------------------------------------------------------")
        print(f"  Evaluating: {parent_name} (Model: {model_name})")
        print(f"----------------------------------------------------------------")

        _, in_channels, physics_mode = parse_model_variant(model_name)
        model = build_model(model_name, pretrained_backbone=False).to(device)

        # Load weights
        ckpt_epoch, ckpt_metrics = load_ckpt(str(p), model, device=str(device))
        print(f"  Loaded Epoch: {ckpt_epoch} | Stored Val PSNR: {ckpt_metrics.get('psnr', 0.0):.4f} dB")

        # Complexity profiling
        params, params_m, params_k, macs_g = profile_model_complexity(
            model, in_channels, img_size=args.img_size, device=device
        )
        print(f"  Parameters : {params_m:.3f} M ({params_k:.1f} k)")
        print(f"  GFLOPs/MACs: {macs_g:.3f} G" if macs_g else "  GFLOPs/MACs: N/A")

        # Test 1: EUVP test_samples (official Inp / GTr)
        euvp_test_samples_metrics = {}
        if euvp_test_samples_pairs:
            print(f"  [1/4] Running EUVP test_samples ({len(euvp_test_samples_pairs)} pairs)...")
            euvp_test_samples_metrics = evaluate_dataset(
                model,
                euvp_test_samples_pairs,
                model_name,
                device,
                physics_mode,
                physics_extractor,
                fusion_extractor,
                "EUVP test_samples",
                img_size=args.img_size,
                batch_size=args.batch_size,
            )
            print(f"    -> PSNR: {euvp_test_samples_metrics.get('psnr', 0.0):.4f} dB | SSIM: {euvp_test_samples_metrics.get('ssim', 0.0):.4f} | CIEDE2000: {euvp_test_samples_metrics.get('ciede2000', 0.0):.4f} | UCIQE: {euvp_test_samples_metrics.get('uciqe', 0.0):.4f} | UIQM: {euvp_test_samples_metrics.get('uiqm', 0.0):.4f}")

        # Test 2: EUVP Scenes (Held-out 218 test pairs)
        euvp_scenes_metrics = {}
        if euvp_scenes_pairs:
            print(f"  [2/4] Running EUVP Scenes Test ({len(euvp_scenes_pairs)} pairs)...")
            euvp_scenes_metrics = evaluate_dataset(
                model,
                euvp_scenes_pairs,
                model_name,
                device,
                physics_mode,
                physics_extractor,
                fusion_extractor,
                "EUVP Scenes",
                img_size=args.img_size,
                batch_size=args.batch_size,
            )
            print(f"    -> PSNR: {euvp_scenes_metrics.get('psnr', 0.0):.4f} dB | SSIM: {euvp_scenes_metrics.get('ssim', 0.0):.4f} | CIEDE2000: {euvp_scenes_metrics.get('ciede2000', 0.0):.4f} | UCIQE: {euvp_scenes_metrics.get('uciqe', 0.0):.4f} | UIQM: {euvp_scenes_metrics.get('uiqm', 0.0):.4f}")

        # Test 3: EUVP Dark
        euvp_dark_metrics = {}
        if euvp_dark_pairs:
            print(f"  [3/4] Running EUVP Dark Test ({len(euvp_dark_pairs)} pairs)...")
            euvp_dark_metrics = evaluate_dataset(
                model,
                euvp_dark_pairs,
                model_name,
                device,
                physics_mode,
                physics_extractor,
                fusion_extractor,
                "EUVP Dark",
                img_size=args.img_size,
                batch_size=args.batch_size,
            )
            print(f"    -> PSNR: {euvp_dark_metrics.get('psnr', 0.0):.4f} dB | SSIM: {euvp_dark_metrics.get('ssim', 0.0):.4f} | CIEDE2000: {euvp_dark_metrics.get('ciede2000', 0.0):.4f} | UCIQE: {euvp_dark_metrics.get('uciqe', 0.0):.4f} | UIQM: {euvp_dark_metrics.get('uiqm', 0.0):.4f}")

        # Test 4: UIEB T90
        uieb_metrics = {}
        if uieb_pairs:
            print(f"  [4/4] Running UIEB T90 Test ({len(uieb_pairs)} pairs)...")
            uieb_metrics = evaluate_dataset(
                model,
                uieb_pairs,
                model_name,
                device,
                physics_mode,
                physics_extractor,
                fusion_extractor,
                "UIEB T90",
                img_size=args.img_size,
                batch_size=args.batch_size,
            )
            print(f"    -> PSNR: {uieb_metrics.get('psnr', 0.0):.4f} dB | SSIM: {uieb_metrics.get('ssim', 0.0):.4f} | CIEDE2000: {uieb_metrics.get('ciede2000', 0.0):.4f} | UCIQE: {uieb_metrics.get('uciqe', 0.0):.4f} | UIQM: {uieb_metrics.get('uiqm', 0.0):.4f}")

        run_info = {
            "run_name": parent_name,
            "model_name": model_name,
            "in_channels": in_channels,
            "best_epoch": ckpt_epoch,
            "params_k": params_k,
            "params_m": params_m,
            "macs_g": macs_g,
            "euvp_test_samples": euvp_test_samples_metrics,
            "euvp_scenes": euvp_scenes_metrics,
            "euvp_dark": euvp_dark_metrics,
            "uieb_t90": uieb_metrics,
        }
        all_runs[parent_name] = run_info
        results_table.append(run_info)

    # 5. Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "euvp_test_samples_samples": len(euvp_test_samples_pairs),
        "euvp_scenes_samples": len(euvp_scenes_pairs),
        "euvp_dark_samples": len(euvp_dark_pairs),
        "uieb_t90_samples": len(uieb_pairs),
        "runs": all_runs,
    }
    json_path = out_dir / "test_results_benchmarks.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] JSON benchmark report saved to {json_path}")

    # 6. Generate Markdown comparison tables for all 5 metrics
    md_lines = [
        "# Comprehensive Underwater Image Enhancement Benchmarks (5 Metrics)",
        "",
        f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Device**: {device}",
        f"- **EUVP test_samples Pairs**: {len(euvp_test_samples_pairs)}",
        f"- **EUVP Scenes Test Pairs**: {len(euvp_scenes_pairs)}",
        f"- **EUVP Dark Test Pairs**: {len(euvp_dark_pairs)}",
        f"- **UIEB T90 Test Pairs**: {len(uieb_pairs)}",
        "",
        "---",
        "",
        "## 1. EUVP test_samples Benchmark (Tập test_samples riêng của EUVP - Inp & GTr)",
        "",
        "| Model | Params (k) | MACs (G) | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in sorted(results_table, key=lambda x: x.get("euvp_test_samples", {}).get("psnr", 0.0), reverse=True):
        m = r.get("euvp_test_samples", {})
        if not m:
            continue
        m_name = r["model_name"]
        pk = f"{r['params_k']:.1f}"
        mg = f"{r['macs_g']:.3f}" if r["macs_g"] is not None else "N/A"
        psnr = f"{m.get('psnr', 0.0):.3f}"
        ssim = f"{m.get('ssim', 0.0):.4f}"
        ciede = f"{m.get('ciede2000', 0.0):.3f}"
        uciqe = f"{m.get('uciqe', 0.0):.4f}"
        uiqm = f"{m.get('uiqm', 0.0):.4f}"
        lat = f"{m.get('inference_ms_per_img', 0.0):.2f}"
        md_lines.append(f"| `{m_name}` | {pk} | {mg} | **{psnr}** | **{ssim}** | {ciede} | {uciqe} | {uiqm} | {lat} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. EUVP Scenes Test Benchmark (Held-out Test Split)",
        "",
        "| Model | Params (k) | MACs (G) | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in sorted(results_table, key=lambda x: x.get("euvp_scenes", {}).get("psnr", 0.0), reverse=True):
        m = r.get("euvp_scenes", {})
        if not m:
            continue
        m_name = r["model_name"]
        pk = f"{r['params_k']:.1f}"
        mg = f"{r['macs_g']:.3f}" if r["macs_g"] is not None else "N/A"
        psnr = f"{m.get('psnr', 0.0):.3f}"
        ssim = f"{m.get('ssim', 0.0):.4f}"
        ciede = f"{m.get('ciede2000', 0.0):.3f}"
        uciqe = f"{m.get('uciqe', 0.0):.4f}"
        uiqm = f"{m.get('uiqm', 0.0):.4f}"
        lat = f"{m.get('inference_ms_per_img', 0.0):.2f}"
        md_lines.append(f"| `{m_name}` | {pk} | {mg} | **{psnr}** | **{ssim}** | {ciede} | {uciqe} | {uiqm} | {lat} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. EUVP Dark Benchmark (Challenging Low-Light)",
        "",
        "| Model | Params (k) | MACs (G) | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in sorted(results_table, key=lambda x: x.get("euvp_dark", {}).get("psnr", 0.0), reverse=True):
        m = r.get("euvp_dark", {})
        if not m:
            continue
        m_name = r["model_name"]
        pk = f"{r['params_k']:.1f}"
        mg = f"{r['macs_g']:.3f}" if r["macs_g"] is not None else "N/A"
        psnr = f"{m.get('psnr', 0.0):.3f}"
        ssim = f"{m.get('ssim', 0.0):.4f}"
        ciede = f"{m.get('ciede2000', 0.0):.3f}"
        uciqe = f"{m.get('uciqe', 0.0):.4f}"
        uiqm = f"{m.get('uiqm', 0.0):.4f}"
        lat = f"{m.get('inference_ms_per_img', 0.0):.2f}"
        md_lines.append(f"| `{m_name}` | {pk} | {mg} | **{psnr}** | **{ssim}** | {ciede} | {uciqe} | {uiqm} | {lat} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. UIEB T90 Benchmark (Real-world Cross-Dataset)",
        "",
        "| Model | Params (k) | MACs (G) | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | UCIQE ↑ | UIQM ↑ | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in sorted(results_table, key=lambda x: x.get("uieb_t90", {}).get("psnr", 0.0), reverse=True):
        m = r.get("uieb_t90", {})
        if not m:
            continue
        m_name = r["model_name"]
        pk = f"{r['params_k']:.1f}"
        mg = f"{r['macs_g']:.3f}" if r["macs_g"] is not None else "N/A"
        psnr = f"{m.get('psnr', 0.0):.3f}"
        ssim = f"{m.get('ssim', 0.0):.4f}"
        ciede = f"{m.get('ciede2000', 0.0):.3f}"
        uciqe = f"{m.get('uciqe', 0.0):.4f}"
        uiqm = f"{m.get('uiqm', 0.0):.4f}"
        lat = f"{m.get('inference_ms_per_img', 0.0):.2f}"
        md_lines.append(f"| `{m_name}` | {pk} | {mg} | **{psnr}** | **{ssim}** | {ciede} | {uciqe} | {uiqm} | {lat} |")

    md_content = "\n".join(md_lines)
    md_path = out_dir / "benchmark_comparison_table.md"
    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"[OK] Markdown table saved to {md_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
