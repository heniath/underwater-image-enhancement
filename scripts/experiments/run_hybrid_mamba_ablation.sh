#!/usr/bin/env bash
set -euo pipefail

# Five matched architecture screens on the fixed EUVP hold-out. This script
# intentionally performs no EUVP-515 or UIEB-90 test-set evaluation.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-./datasets/EUVP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./hybrid_mamba_ablation}"
THREADS="${THREADS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
CROP_SIZE="${CROP_SIZE:-256}"
EPOCHS="${EPOCHS:-30}"
ALLOW_SLOW_FALLBACK="${ALLOW_SLOW_FALLBACK:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RUN_ID="${RUN_ID:-hybrid_mamba_seed42_$(date +%Y%m%d_%H%M%S)}"

CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
LOG_ROOT="${OUTPUT_ROOT}/logs"
RESULT_ROOT="${OUTPUT_ROOT}/results"
mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${RESULT_ROOT}"

python - "${NUM_GPUS}" "${ALLOW_SLOW_FALLBACK}" <<'PY'
import sys

import torch

from uwir.models.mamba_unet import _MAMBA_CUDA

requested = int(sys.argv[1])
allow_fallback = sys.argv[2] == "1"
available = torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(available)]
print(f"CUDA devices ({available}): {names}")
print(f"Fused mamba_ssm selective scan: {_MAMBA_CUDA}")
if available < requested:
    raise SystemExit(f"This screen requires {requested} CUDA GPUs; found {available}.")
if not _MAMBA_CUDA and not allow_fallback:
    raise SystemExit(
        "Fused mamba_ssm selective scan is required. Install mamba_ssm, or explicitly "
        "set ALLOW_SLOW_FALLBACK=1 for the checkpointed PyTorch fallback."
    )
if any("T4" not in name for name in names[:requested]):
    print("WARNING: the matched reference configuration uses two NVIDIA T4 GPUs.")
PY

COMMON_ARGS=(
  --dataset euvp
  --data-train-euvp "${DATA_ROOT}"
  --euvp-subset all
  --crop-size "${CROP_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --grad-accumulation-steps "${ACCUMULATION_STEPS}"
  --threads "${THREADS}"
  --num-gpus "${NUM_GPUS}"
  --split-seed 42
  --prior-method udcp
  --guided-filter-radius 15
  --guided-filter-eps 1e-3
  --lr 1e-4
  --weight-decay 1e-5
  --l1-weight 1.0
  --perceptual-weight 0.1
  --ssim-weight 0.5
  --cos-restart true
  --start-warmup true
  --warmup-epochs 3
  --early-stop-patience "${EPOCHS}"
  --val-interval 1
  --amp true
  --pretrained-backbone false
  --seed 42
)

MODELS=(
  unet_3ch
  hybridmamba_core_3ch
  hybridmamba_local_3ch
  hybridmamba_attn_3ch
  hybridmambafusion_7ch_tv
)

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Running one-epoch fused-scan/AMP/checkpointing preflight."
  python -m uwir.cli.train \
    "${COMMON_ARGS[@]}" \
    --model hybridmamba_core_3ch \
    --epochs 1 \
    --snapshots 1 \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --log-dir "${LOG_ROOT}" \
    --val-folder "${RESULT_ROOT}" \
    --run-name "${RUN_ID}_preflight"
  exit 0
fi

for model in "${MODELS[@]}"; do
  python -m uwir.cli.train \
    "${COMMON_ARGS[@]}" \
    --model "${model}" \
    --epochs "${EPOCHS}" \
    --snapshots "${EPOCHS}" \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --log-dir "${LOG_ROOT}" \
    --val-folder "${RESULT_ROOT}" \
    --run-name "${RUN_ID}_${model}"
done

python - "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" "${RUN_ID}" "${EPOCHS}" \
  "$((BATCH_SIZE * ACCUMULATION_STEPS))" <<'PY'
import gc
import json
import pathlib
import sys

from uwir.models import build_model

checkpoint_root = pathlib.Path(sys.argv[1])
result_root = pathlib.Path(sys.argv[2])
run_id = sys.argv[3]
epochs = int(sys.argv[4])
effective_batch_size = int(sys.argv[5])
models = [
    "unet_3ch",
    "hybridmamba_core_3ch",
    "hybridmamba_local_3ch",
    "hybridmamba_attn_3ch",
    "hybridmambafusion_7ch_tv",
]

rows = []
previous = None
for model_name in models:
    matches = sorted(
        checkpoint_root.glob(f"{run_id}_{model_name}_*/training_history.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise RuntimeError(f"No training history found for {model_name}")
    history = json.loads(matches[-1].read_text())
    scores = list(zip(history["val_psnr"], history["val_ssim"], strict=True))
    best_index = max(range(len(scores)), key=lambda index: scores[index])
    model = build_model(model_name, pretrained_backbone=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()
    row = {
        "model": model_name,
        "best_epoch": best_index + 1,
        "best_validation_psnr": scores[best_index][0],
        "best_validation_ssim": scores[best_index][1],
        "parameters": parameters,
        "training_time_min": history["_meta"].get("training_time_min"),
        "peak_gpu_memory_mb": history["_meta"].get("peak_gpu_memory_mb"),
    }
    row["delta_from_previous"] = (
        None
        if previous is None
        else {
            "psnr": row["best_validation_psnr"] - previous["best_validation_psnr"],
            "ssim": row["best_validation_ssim"] - previous["best_validation_ssim"],
        }
    )
    rows.append(row)
    previous = row

core = next(row for row in rows if row["model"] == "hybridmamba_core_3ch")
full = next(row for row in rows if row["model"] == "hybridmambafusion_7ch_tv")
for row in rows:
    row["cumulative_delta_from_core"] = (
        None
        if row["model"] == "unet_3ch"
        else {
            "psnr": row["best_validation_psnr"] - core["best_validation_psnr"],
            "ssim": row["best_validation_ssim"] - core["best_validation_ssim"],
        }
    )
core_delta = {
    "psnr": full["best_validation_psnr"] - core["best_validation_psnr"],
    "ssim": full["best_validation_ssim"] - core["best_validation_ssim"],
}
report = {
    "screen": {
        "dataset": "EUVP fixed training hold-out only",
        "epochs": epochs,
        "seed": 42,
        "effective_batch_size": effective_batch_size,
        "models": rows,
    },
    "full_vs_core": core_delta,
    "confirmation_thresholds": {"minimum_psnr_gain": 0.20, "maximum_ssim_loss": 0.002},
    "confirmation_worthy": core_delta["psnr"] >= 0.20 and core_delta["ssim"] >= -0.002,
}
output = result_root / "hybrid_mamba_ablation_report.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"Report: {output}")
PY
