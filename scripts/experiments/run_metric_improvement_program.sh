#!/usr/bin/env bash
set -euo pipefail

# Ten matched runs: four 30-epoch screens followed by baseline/candidate
# confirmation at three paired seeds. Intended for two NVIDIA T4 GPUs.

DATA_ROOT="${DATA_ROOT:-./datasets/EUVP}"
UIEB_ROOT="${UIEB_ROOT:-./datasets/UIEB}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./metric_improvement_program}"
THREADS="${THREADS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CROP_SIZE="${CROP_SIZE:-256}"
SPLIT_SEED="${SPLIT_SEED:-42}"

SCREEN_ROOT="${OUTPUT_ROOT}/screen"
FULL_ROOT="${OUTPUT_ROOT}/full"
mkdir -p "${SCREEN_ROOT}/checkpoints" "${SCREEN_ROOT}/logs" "${SCREEN_ROOT}/results"
mkdir -p "${FULL_ROOT}/checkpoints" "${FULL_ROOT}/logs" "${FULL_ROOT}/results"

COMMON_ARGS=(
  --dataset euvp
  --data-train-euvp "${DATA_ROOT}"
  --euvp-subset all
  --crop-size "${CROP_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --grad-accumulation-steps 2
  --threads "${THREADS}"
  --num-gpus "${NUM_GPUS}"
  --split-seed "${SPLIT_SEED}"
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
  --early-stop-patience 10
  --val-interval 1
  --amp true
  --pretrained-backbone true
)

SCREEN_MODELS=(
  unet_3ch
  unet_5ch
  asppfusion_7ch_tv
  denseasppfusion_7ch_tv
)

for model in "${SCREEN_MODELS[@]}"; do
  python -m uwir.cli.train \
    "${COMMON_ARGS[@]}" \
    --model "${model}" \
    --epochs 30 \
    --seed 42 \
    --snapshots 30 \
    --checkpoint-dir "${SCREEN_ROOT}/checkpoints" \
    --log-dir "${SCREEN_ROOT}/logs" \
    --val-folder "${SCREEN_ROOT}/results" \
    --run-name "${model}_screen_seed42"
done

python - "${SCREEN_ROOT}/checkpoints" "${SCREEN_ROOT}/results" <<'PY'
import json
import pathlib
import sys

checkpoint_root = pathlib.Path(sys.argv[1])
result_root = pathlib.Path(sys.argv[2])
rows = []
for history_path in checkpoint_root.glob("*/training_history.json"):
    history = json.loads(history_path.read_text())
    model = history["_meta"]["model"]
    scores = list(zip(history["val_psnr"], history["val_ssim"]))
    best_epoch, (psnr, ssim) = max(enumerate(scores, 1), key=lambda item: item[1])
    rows.append({"model": model, "epoch": best_epoch, "psnr": psnr, "ssim": ssim})

rows.sort(key=lambda row: (row["psnr"], row["ssim"]), reverse=True)
candidates = [row for row in rows if row["model"] != "unet_3ch"]
winner = candidates[0]["model"]
summary = {"ranking": rows, "candidate_for_confirmation": winner}
(result_root / "screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(result_root / "winner.txt").write_text(winner + "\n")
print(json.dumps(summary, indent=2))
PY

WINNER="$(tr -d '[:space:]' < "${SCREEN_ROOT}/results/winner.txt")"
for seed in 42 123 3407; do
  for model in unet_3ch "${WINNER}"; do
    python -m uwir.cli.train \
      "${COMMON_ARGS[@]}" \
      --model "${model}" \
      --epochs 100 \
      --seed "${seed}" \
      --snapshots 100 \
      --checkpoint-dir "${FULL_ROOT}/checkpoints" \
      --log-dir "${FULL_ROOT}/logs" \
      --val-folder "${FULL_ROOT}/results" \
      --run-name "${model}_full_seed${seed}"
  done
done

python -m uwir.cli.evaluate \
  --checkpoint-dir "${FULL_ROOT}/checkpoints" \
  --val-folder "${FULL_ROOT}/results/euvp" \
  --data-train-euvp "${DATA_ROOT}" \
  --eval-benchmark euvp \
  --native-eval true \
  --tile-size 512 \
  --tile-overlap 64

python -m uwir.cli.evaluate \
  --checkpoint-dir "${FULL_ROOT}/checkpoints" \
  --val-folder "${FULL_ROOT}/results/uieb" \
  --data-train-uieb "${UIEB_ROOT}" \
  --eval-benchmark uieb \
  --native-eval true \
  --tile-size 512 \
  --tile-overlap 64

python scripts/experiments/summarize_metric_program.py "${FULL_ROOT}" "${WINNER}"
