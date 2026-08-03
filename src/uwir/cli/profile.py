"""
net_test.py
-----------
Benchmark the paper U-Net and retained UW-LYT channel variants.

Usage
-----
    uwir-profile                 # test all models
    uwir-profile unet uwlyt      # filter registered models

The script gracefully skips a model if it cannot be instantiated so the
remaining models are still reported.

Output files
------------
Results are written to  results/net_benchmark_<YYYYMMDD_HHMMSS>.txt  (full log)
and                      results/net_benchmark_<YYYYMMDD_HHMMSS>.csv  (summary table).
"""

import argparse
import csv
import os
import resource
import sys
import time
import traceback
from datetime import datetime

import torch

try:
    from thop import profile as thop_profile

    _THOP_AVAILABLE = True
except ImportError:
    _THOP_AVAILABLE = False
    print(
        "[WARNING] thop not installed – FLOPs will not be reported. "
        "Install with:  pip install thop\n"
    )

from uwir.models import ALL_MODEL_NAMES, build_model, parse_model_variant

# ---------------------------------------------------------------------------
# Tee: write to both stdout and a log file simultaneously
# ---------------------------------------------------------------------------


class _Tee:
    """Mirror writes to *stream* (e.g. sys.stdout) and to *file_obj*."""

    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj

    def write(self, data: str):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    # Forward any other attribute access to the underlying stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _results_dir(output_dir: str) -> str:
    path = os.path.abspath(output_dir)
    os.makedirs(path, exist_ok=True)
    return path


def _get_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    if requested == "cuda" or torch.cuda.is_available():
        return torch.device("cuda")
    print("[INFO] CUDA not available – running on CPU (results will be slow).\n")
    return torch.device("cpu")


def benchmark_model(
    model_name: str,
    *,
    pretrained_backbone: bool,
    device: torch.device,
    img_size: int = 256,
    warmup_runs: int = 1,
    timed_runs: int = 3,
    prior_method: str = "udcp",
) -> dict | None:
    """
    Build, warm-up and time a single model.

    Returns
    -------
    dict with result fields, or None if the model was skipped.
    """
    # ---- parse variant to know in_channels --------------------------------
    try:
        _, in_channels, physics_mode = parse_model_variant(model_name)
    except ValueError as exc:
        print(f"[SKIP] {model_name}: cannot parse variant – {exc}")
        return None

    # ---- build model -------------------------------------------------------
    try:
        model = build_model(model_name, pretrained_backbone=pretrained_backbone)
    except Exception:
        print(f"[SKIP] {model_name}: failed to instantiate –")
        traceback.print_exc()
        print()
        return None

    model = model.to(device).eval()
    rgb = torch.rand(1, 3, img_size, img_size)
    physics_extractor = None
    if physics_mode != "none":
        from uwir.cli.train import _resolve_physics_extractor

        physics_extractor = _resolve_physics_extractor(prior_method)

    def make_input() -> torch.Tensor:
        """Build model input, including legacy prior generation when required."""
        if physics_mode == "none":
            return rgb.to(device)
        from uwir.cli.train import _add_physics_channels

        augmented = _add_physics_channels(rgb[0], physics_mode, physics_extractor)
        return augmented.unsqueeze(0).to(device)

    dummy = make_input()

    # ---- warm-up -----------------------------------------------------------
    try:
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(make_input())
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as exc:
        print(f"[SKIP] {model_name}: forward pass failed during warm-up – {exc}\n")
        return None

    # ---- timed runs --------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(timed_runs):
            _ = model(make_input())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t_start) / timed_runs * 1e3
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 2**20
    else:
        # ru_maxrss is KiB on Linux and captures tensor allocations missed by tracemalloc.
        peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # ---- stats -------------------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters())
    params_m = n_params / 1e6

    flops_g = None
    flops_str = "N/A (thop not installed)"
    if _THOP_AVAILABLE:
        try:
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            flops_g = macs / 2**30
            flops_str = f"{flops_g:.3f} G"
        except Exception:
            flops_str = "N/A (profile error)"

    # ---- pretty print ------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  {model_name}  (physics_mode={physics_mode!r})")
    print(f"{'=' * 65}")
    print(f"  Device  : {device}")
    print(f"  Channels: {in_channels}")
    print(f"  Params  : {params_m:.3f} M")
    print(f"  FLOPs   : {flops_str}")
    print(f"  Peak mem: {peak_memory_mb:.2f} MiB")
    print(
        f"  Time    : {elapsed_ms:.2f} ms  "
        f"(avg over {timed_runs} run{'s' if timed_runs > 1 else ''}, no grad)"
    )

    return {
        "model": model_name,
        "physics_mode": physics_mode,
        "device": str(device),
        "in_channels": in_channels,
        "params_M": round(params_m, 3),
        "flops_G": round(flops_g, 3) if flops_g is not None else "N/A",
        "peak_memory_MiB": round(peak_memory_mb, 2),
        "time_ms": round(elapsed_ms, 2),
        "memory_scope": "CUDA allocated" if device.type == "cuda" else "process peak RSS",
        "latency_scope": f"{prior_method}+model" if physics_mode != "none" else "model",
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_csv(rows: list[dict], csv_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] CSV summary saved → {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark all (or selected) model architectures.")
    parser.add_argument(
        "filters",
        nargs="*",
        metavar="FILTER",
        help="Optional substrings to filter model names. "
        "A model is included if ANY filter matches its name. "
        "Leave empty to test all models.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Profiling device; select cpu and cuda in separate matched runs.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        default=True,
        help="Disable pretrained backbone loading (faster, no network access).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        metavar="N",
        help="Spatial resolution of the dummy input (default: 256).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of timed forward passes to average (default: 3).",
    )
    parser.add_argument(
        "--prior-method",
        choices=("udcp",),
        default="udcp",
        help="Prior included in end-to-end physics-variant latency.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/",
        help="Directory for benchmark logs and CSV reports (default: ./results/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered model names and exit.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)

    if args.list:
        print("Registered model names:")
        for name in ALL_MODEL_NAMES:
            print(f"  {name}")
        return

    # Select models to benchmark
    if args.filters:
        selected = [n for n in ALL_MODEL_NAMES if any(f in n for f in args.filters)]
        if not selected:
            print(f"[ERROR] No model names match filters: {args.filters}")
            print(f"Available: {ALL_MODEL_NAMES}")
            sys.exit(1)
    else:
        selected = list(ALL_MODEL_NAMES)

    device = _get_device(args.device)

    # ---- set up output files -----------------------------------------------
    rdir = _results_dir(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(rdir, f"net_benchmark_{stamp}.txt")
    csv_path = os.path.join(rdir, f"net_benchmark_{stamp}.csv")

    log_file = open(txt_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_file)

    try:
        print(
            f"\nBenchmarking {len(selected)} model(s) on {device} "
            f"| img_size={args.img_size} | runs={args.runs} "
            f"| pretrained={args.pretrained}\n"
        )
        print(f"[INFO] Full log  → {txt_path}")
        print(f"[INFO] CSV table → {csv_path}\n")

        rows = []
        skipped = 0

        for name in selected:
            result = benchmark_model(
                name,
                pretrained_backbone=args.pretrained,
                device=device,
                img_size=args.img_size,
                timed_runs=args.runs,
                prior_method=args.prior_method,
            )
            if result is not None:
                rows.append(result)
            else:
                skipped += 1

        # ---- summary table (console + log) ---------------------------------
        print(f"\n\n{'=' * 65}")
        print(f"  Summary: {len(rows)} passed, {skipped} skipped")
        print(f"{'=' * 65}")
        if rows:
            hdr = (
                f"{'Model':<22} {'Ch':>3}  {'Params(M)':>10}  {'FLOPs(G)':>10}  "
                f"{'Peak(MiB)':>10}  {'Time(ms)':>10}  {'Scope':>12}"
            )
            print(f"\n{hdr}")
            print("-" * len(hdr))
            for r in rows:
                flops_disp = (
                    f"{r['flops_G']:.3f}" if isinstance(r["flops_G"], float) else r["flops_G"]
                )
                print(
                    f"  {r['model']:<20} {r['in_channels']:>3}  "
                    f"{r['params_M']:>10.3f}  {flops_disp:>10}  "
                    f"{r['peak_memory_MiB']:>10.2f}  {r['time_ms']:>10.2f}  "
                    f"{r['latency_scope']:>12}"
                )
        print()

        # ---- CSV -----------------------------------------------------------
        _save_csv(rows, csv_path)

    finally:
        sys.stdout = orig_stdout
        log_file.close()

    print(f"\n[INFO] Done. Results saved to:\n       Log : {txt_path}\n       CSV : {csv_path}\n")


if __name__ == "__main__":
    main()
