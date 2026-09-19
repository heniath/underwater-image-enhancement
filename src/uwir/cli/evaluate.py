# ============================================================
# eval.py  –  Test-set evaluation for checkpoint folders
# ============================================================

import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

try:
    import thop
except ImportError:
    thop = None

from uwir.cli.train import _collate_val, _resolve_physics_extractor, load_ckpt
from uwir.config import option
from uwir.models import ALL_MODEL_NAMES, build_model, parse_model_variant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def collect_test_pairs(data_root):
    """Return a list of (input_path, gt_path) tuples from test_samples/."""
    inp_dir = Path(data_root) / "test_samples" / "Inp"
    gt_dir = Path(data_root) / "test_samples" / "GTr"
    pairs = []
    if not (inp_dir.exists() and gt_dir.exists()):
        print("[WARN] test_samples not found – test set will be empty.")
        return pairs
    gt_dict = {f.stem: f for f in gt_dir.iterdir() if f.suffix in IMG_EXTS}
    for inp_file in sorted(inp_dir.iterdir()):
        if inp_file.suffix not in IMG_EXTS:
            continue
        if inp_file.stem in gt_dict:
            pairs.append((str(inp_file), str(gt_dict[inp_file.stem])))
    return pairs


def collect_uieb_test_pairs(data_root, seed: int = 42, count: int = 90):
    """Return a fixed, deterministic UIEB-90 cross-dataset benchmark."""
    input_dir = Path(data_root) / "raw-890"
    target_dir = Path(data_root) / "reference-890"
    targets = {path.stem: path for path in target_dir.iterdir() if path.suffix in IMG_EXTS}
    pairs = [
        (str(path), str(targets[path.stem]))
        for path in sorted(input_dir.iterdir())
        if path.suffix in IMG_EXTS and path.stem in targets
    ]
    generator = np.random.default_rng(seed)
    selected = sorted(generator.choice(len(pairs), size=min(count, len(pairs)), replace=False))
    return [pairs[index] for index in selected]


class TestDataset(data.Dataset):
    def __init__(self, pairs, img_size=None):
        import torchvision.transforms as transforms

        self.pairs = pairs
        self.resize = (
            transforms.Resize((img_size, img_size), antialias=True) if img_size else None
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image

        inp_path, gt_path = self.pairs[idx]
        inp_image = Image.open(inp_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        if inp_image.size != gt_image.size:
            raise ValueError(f"Pair size mismatch: {inp_path}={inp_image.size}, {gt_path}={gt_image.size}")
        if self.resize:
            inp_image, gt_image = self.resize(inp_image), self.resize(gt_image)
        inp_t = self.to_tensor(inp_image)
        gt_t = self.to_tensor(gt_image)
        # Return dummy filename strings so _collate_val (which unpacks *_) works
        return inp_t, gt_t, "", ""


# ---------------------------------------------------------------------------
# Model-name resolver
# ---------------------------------------------------------------------------


def _parse_model_name(run_name: str) -> str | None:
    """
    Infer the model variant from a checkpoint folder name.

    Tries two strategies:
    1. Normalise separators (3ch / 4ch / 5ch → _3ch …) and prefix-match
       against every registered model name (longest first).
    2. Return None if nothing matches.
    """
    normalized = run_name.replace("3ch", "_3ch").replace("4ch", "_4ch").replace("5ch", "_5ch")
    normalized = normalized.replace("__", "_")
    for mn in sorted(ALL_MODEL_NAMES, key=len, reverse=True):
        if run_name.startswith(mn) or normalized.startswith(mn):
            return mn
    return None


# ---------------------------------------------------------------------------
# Pretty summary table
# ---------------------------------------------------------------------------


def _print_summary(all_results: dict):
    """Print a ranked summary table sorted by PSNR (descending)."""
    rows = [
        (name, info) for name, info in all_results.items() if info.get("test_metrics") is not None
    ]
    if not rows:
        print("\n[INFO] No successful test results to summarise.")
        return

    rows.sort(key=lambda x: x[1]["test_metrics"]["psnr"], reverse=True)

    header = f"{'Run':<45} {'PSNR':>8} {'SSIM':>8} {'Inf(ms)':>8} {'Tr(min)':>8} {'Params(M)':>10} {'MACs(G)':>8}"
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("  RANKED SUMMARY  (sorted by PSNR descending)")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for name, info in rows:
        m = info["test_metrics"]
        tr_min = info.get("training_time_min")
        tr_min_str = f"{tr_min:>8.1f}" if tr_min is not None else f"{'N/A':>8}"

        params_m = info.get("params_m")
        params_str = f"{params_m:>10.2f}" if params_m is not None else f"{'N/A':>10}"

        macs_g = info.get("macs_g")
        macs_g_str = f"{macs_g:>8.2f}" if macs_g is not None else f"{'N/A':>8}"

        inf_ms = m.get("inference_ms_per_img")
        inf_str = f"{inf_ms:>8.1f}" if inf_ms is not None else f"{'N/A':>8}"

        print(
            f"{name:<45} "
            f"{m['psnr']:>8.4f} "
            f"{m['ssim']:>8.4f} "
            f"{inf_str} "
            f"{tr_min_str} "
            f"{params_str} "
            f"{macs_g_str}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = option()
    parser.add_argument(
        "--run_filter",
        type=str,
        default="",
        help="Optional substring filter for checkpoint directory names",
    )
    args = parser.parse_args()

    from uwir.metrics import evaluate_loader

    device = torch.device("cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu")
    print(f"Evaluation Device : {device}")
    print(f"Prior method      : {args.prior_method}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    physics_extractor = _resolve_physics_extractor(args.prior_method)

    # ------------------------------------------------------------------
    # Datasets per benchmark (evaluated separately, never merged)
    # ------------------------------------------------------------------
    benchmarks: dict = {}
    if args.eval_benchmark in ("uieb", "uieb+euvp", "all"):
        print(f"\nLoading UIEB-90 test samples from '{args.data_train_uieb}' …")
        uieb_pairs = collect_uieb_test_pairs(args.data_train_uieb, seed=42)
        benchmarks["UIEB"] = uieb_pairs
        print(f"Loaded: UIEB (n={len(uieb_pairs)})")

    if args.eval_benchmark in ("euvp", "uieb+euvp", "all"):
        print(f"\nLoading EUVP test samples from '{args.data_train_euvp}' …")
        euvp_pairs = collect_test_pairs(args.data_train_euvp)
        benchmarks["EUVP"] = euvp_pairs
        print(f"Loaded: EUVP (n={len(euvp_pairs)})")

    if not benchmarks:
        raise ValueError(f"Unknown or empty --eval_benchmark: {args.eval_benchmark}")

    data_root = f"UIEB:{args.data_train_uieb},EUVP:{args.data_train_euvp}"

    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Checkpoint directory does not exist: {checkpoint_dir}")
        return

    # ------------------------------------------------------------------
    # Per-checkpoint evaluation
    # ------------------------------------------------------------------
    all_results: dict = {}
    if os.path.isfile(os.path.join(checkpoint_dir, "best_model.pth")):
        run_name = os.path.basename(os.path.normpath(checkpoint_dir))
        checkpoint_dir = os.path.dirname(os.path.normpath(checkpoint_dir))
        run_dirs = [run_name]
    else:
        run_dirs = sorted(
            d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))
        )

    for run_name in run_dirs:
        if args.run_filter and args.run_filter not in run_name:
            continue
        run_dir = os.path.join(checkpoint_dir, run_name)
        best_path = os.path.join(run_dir, "best_model.pth")

        if not os.path.isfile(best_path):
            continue

        model_name = _parse_model_name(run_name)
        if model_name is None:
            print(f"[SKIP] {run_name}: cannot parse model variant")
            continue

        print(f"\n{'=' * 65}")
        print(f"  Evaluating : {run_name}")
        print(f"  Model      : {model_name}")
        print(f"{'=' * 65}")

        try:
            _, in_channels, physics_mode = parse_model_variant(model_name)
            model = build_model(model_name, pretrained_backbone=False).to(device)

            ckpt_epoch, ckpt_metrics = load_ckpt(best_path, model, device=str(device))
            print(f"  Loaded epoch={ckpt_epoch}  stored metrics={ckpt_metrics}")

            def collate_fn(batch, mode=physics_mode):
                return _collate_val(batch, mode, physics_extractor)

            def make_loader(dataset, batch_size):
                return data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=device.type == "cuda",
                    drop_last=False,
                    collate_fn=collate_fn,
                )

            # --- 1. Compute complexity using thop ---
            macs_g, params_m = None, None
            if thop is not None:
                try:
                    dummy_input = torch.randn(1, in_channels, args.cropSize, args.cropSize).to(
                        device
                    )
                    macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
                    macs_g = macs / 1e9
                    params_m = params / 1e6
                    print(f"  Params    : {params_m:>8.2f} M")
                    print(f"  MACs      : {macs_g:>8.2f} G")
                except Exception as e:
                    print(f"  [WARN] thop profile failed: {e}")

            # --- 2. Extract training time from log ---
            training_time_min = None
            log_path = os.path.join(args.checkpoint_dir, "..", "logs", f"{run_name}.log")
            if os.path.isfile(log_path):
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    match = re.search(r"Total training time:\s*([\d.]+)\s*min", content)
                    if match:
                        training_time_min = float(match.group(1))
                        print(f"  Train Time: {training_time_min:>8.2f} min (extracted from log)")

            bench_results: dict = {}
            for bench_name, bench_pairs in benchmarks.items():
                native_ds = TestDataset(bench_pairs)
                legacy_ds = TestDataset(bench_pairs, img_size=args.cropSize)

                legacy_loader = make_loader(legacy_ds, args.batchSize)
                native_loader = make_loader(native_ds, 1)

                legacy_metrics, n_test = evaluate_loader(
                    model, legacy_loader, device, desc=f"Legacy testing {run_name} [{bench_name}]"
                )
                if args.native_eval:
                    test_metrics, _ = evaluate_loader(
                        model,
                        native_loader,
                        device,
                        desc=f"Native testing {run_name} [{bench_name}]",
                        tile_size=args.tile_size,
                        tile_overlap=args.tile_overlap,
                    )
                else:
                    test_metrics = legacy_metrics

                print(f"\n  RESULTS ON {bench_name} (n={len(bench_pairs)})")
                print(f"  PSNR      : {test_metrics['psnr']:>8.4f}  dB")
                print(f"  SSIM      : {test_metrics['ssim']:>8.4f}")
                print(f"  CIEDE2000 : {test_metrics['ciede2000']:>8.4f}  (lower=better)")
                print(f"  UCIQE     : {test_metrics['uciqe']:>8.4f}  (higher=better)")
                print(f"  UIQM      : {test_metrics['uiqm']:>8.4f}  (higher=better)")

                bench_results[bench_name.lower()] = test_metrics

            primary_bench = "uieb" if "uieb" in bench_results else list(bench_results.keys())[0]

            all_results[run_name] = {
                "model_name": model_name,
                "in_channels": in_channels,
                "physics_mode": physics_mode,
                "best_epoch": ckpt_epoch,
                "ckpt_metrics": ckpt_metrics,
                "test_metrics": bench_results[primary_bench],
                "benchmarks": bench_results,
                "macs_g": macs_g,
                "params_m": params_m,
                "training_time_min": training_time_min,
                "status": "ok",
            }

        except Exception:
            print(f"\n  [ERROR] Evaluation failed for {run_name}:")
            traceback.print_exc()
            all_results[run_name] = {
                "model_name": model_name,
                "status": "error",
                "error": traceback.format_exc(),
            }

    # ------------------------------------------------------------------
    # Ranked summary table per benchmark
    # ------------------------------------------------------------------
    for b_name in benchmarks:
        b_key = b_name.lower()
        sub_results = {}
        for r_name, r_info in all_results.items():
            if r_info.get("status") == "ok" and b_key in r_info.get("benchmarks", {}):
                copy_info = dict(r_info)
                copy_info["test_metrics"] = r_info["benchmarks"][b_key]
                sub_results[r_name] = copy_info
        if sub_results:
            print(f"\n{'=' * 65}")
            print(f"  RANKED SUMMARY — BENCHMARK: {b_name} (n={len(benchmarks[b_name])})")
            print(f"{'=' * 65}")
            _print_summary(sub_results)

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    os.makedirs(args.val_folder, exist_ok=True)
    out_path = os.path.join(args.val_folder, "test_results_all.json")

    # Attach run-level metadata for reproducibility
    output = {
        "_meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "data_root": data_root,
            "checkpoint_dir": checkpoint_dir,
            "prior_method": args.prior_method,
            "crop_size": args.cropSize,
            "seed": args.seed,
            "benchmark": args.eval_benchmark,
            "pair_manifest": {b_name: len(pairs) for b_name, pairs in benchmarks.items()},
            "metric_version": 2,
        },
        "runs": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  Results saved -> {out_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
