#!/usr/bin/env bash
set -euo pipefail

# Four-run U-Net training-recipe screen for Kaggle/Linux CUDA.
# Override paths/settings as environment variables, for example:
#   DATA_ROOT=/kaggle/input/euvp-dataset/EUVP \
#   OUTPUT_ROOT=/kaggle/working/recipe_screen \
#   bash scripts/experiments/run_training_recipe_screen.sh

DATA_ROOT="${DATA_ROOT:-./datasets/EUVP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./training_recipe_screen}"
EUVP_SUBSET="${EUVP_SUBSET:-underwater_scenes}"
PRIOR_METHOD="${PRIOR_METHOD:-gupdm}"
CROP_SIZE="${CROP_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-12}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED="${SEED:-42}"
THREADS="${THREADS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
SCREEN_ID="${SCREEN_ID:-recipe_screen_$(date +%Y%m%d_%H%M%S)}"

CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
LOG_ROOT="${OUTPUT_ROOT}/logs"
RESULT_ROOT="${OUTPUT_ROOT}/results"
mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "${RESULT_ROOT}"

python - "${NUM_GPUS}" <<'PY'
import sys
import torch

requested = int(sys.argv[1])
available = torch.cuda.device_count()
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} (devices={available})")
if not torch.cuda.is_available():
    print("WARNING: this screen is intended for a Kaggle GPU accelerator.")
elif available < requested:
    print(f"WARNING: requested {requested} GPUs, but only {available} are available.")
PY

# name|L1|VGG perceptual|SSIM|scheduler
RECIPES=(
  "control|1.0|1.0|0.0|step"
  "lower_perceptual|1.0|0.1|0.0|step"
  "balanced_loss|1.0|0.1|0.5|step"
  "balanced_cosine|1.0|0.1|0.5|cosine"
)

echo "Screen ID: ${SCREEN_ID}"
echo "Matched setup: model=unet_5ch prior=${PRIOR_METHOD} subset=${EUVP_SUBSET} seed=${SEED}"
echo "Matched setup: epochs=${EPOCHS} batch=${BATCH_SIZE} crop=${CROP_SIZE} lr=${LEARNING_RATE}"

for spec in "${RECIPES[@]}"; do
  IFS='|' read -r recipe l1_weight perceptual_weight ssim_weight scheduler <<< "${spec}"
  run_name="${SCREEN_ID}_${recipe}_seed${SEED}"
  scheduler_args=(--cos-restart false)
  if [[ "${scheduler}" == "cosine" ]]; then
    scheduler_args=(--cos-restart true)
  fi

  echo "Running ${recipe}: L1=${l1_weight} VGG=${perceptual_weight} SSIM=${ssim_weight} scheduler=${scheduler}"
  python -m uwir.cli.train \
    --model unet_5ch \
    --dataset euvp \
    --data-train-euvp "${DATA_ROOT}" \
    --euvp-subset "${EUVP_SUBSET}" \
    --prior-method "${PRIOR_METHOD}" \
    --crop-size "${CROP_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --threads "${THREADS}" \
    --num-gpus "${NUM_GPUS}" \
    --l1-weight "${l1_weight}" \
    --perceptual-weight "${perceptual_weight}" \
    --ssim-weight "${ssim_weight}" \
    --pretrained-backbone false \
    --snapshots "${EPOCHS}" \
    --early-stop-patience "${EPOCHS}" \
    --scheduler-step 30 \
    --scheduler-gamma 0.5 \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --log-dir "${LOG_ROOT}" \
    --val-folder "${RESULT_ROOT}" \
    --run-name "${run_name}" \
    "${scheduler_args[@]}"
done

python - "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" "${SCREEN_ID}" \
  "${SEED}" "${EPOCHS}" "${LEARNING_RATE}" <<'PY'
import json
import math
import pathlib
import sys

checkpoint_root = pathlib.Path(sys.argv[1])
result_root = pathlib.Path(sys.argv[2])
screen_id = sys.argv[3]
seed = int(sys.argv[4])
epochs = int(sys.argv[5])
initial_lr = float(sys.argv[6])
recipes = {
    "control": (1.0, 1.0, 0.0, "step"),
    "lower_perceptual": (1.0, 0.1, 0.0, "step"),
    "balanced_loss": (1.0, 0.1, 0.5, "step"),
    "balanced_cosine": (1.0, 0.1, 0.5, "cosine_single_cycle"),
}

rows = []
sample_counts = set()
for recipe, expected_recipe in recipes.items():
    matches = list(checkpoint_root.glob(f"{screen_id}_{recipe}_seed{seed}_*/training_history.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one history for {recipe}, found {len(matches)}: {matches}")
    history_path = matches[0]
    run_dir = history_path.parent
    best_path = run_dir / "best_model.pth"
    if not best_path.is_file():
        raise RuntimeError(f"Best-PSNR checkpoint missing for {recipe}: {best_path}")

    history = json.loads(history_path.read_text())
    metadata = history.get("_meta", {})
    actual_recipe = (
        metadata.get("l1_weight"),
        metadata.get("perceptual_weight"),
        metadata.get("ssim_weight"),
        metadata.get("scheduler"),
    )
    if actual_recipe != expected_recipe:
        raise RuntimeError(
            f"{recipe}: recipe metadata differs: expected {expected_recipe}, got {actual_recipe}"
        )
    if metadata.get("seed") != seed:
        raise RuntimeError(f"{recipe}: expected seed {seed}, got {metadata.get('seed')}")
    counts = (metadata.get("train_samples"), metadata.get("validation_samples"))
    if not all(isinstance(count, int) and count > 0 for count in counts):
        raise RuntimeError(f"{recipe}: invalid train/validation counts: {counts}")
    sample_counts.add(counts)

    metric_keys = ("train_loss", "val_loss", "val_psnr", "val_ssim", "lr")
    for key in metric_keys:
        values = history.get(key, [])
        if len(values) != epochs:
            raise RuntimeError(f"{recipe}: expected {epochs} {key} values, got {len(values)}")
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{recipe}: non-finite value in {key}")

    best_index = max(
        range(epochs),
        key=lambda index: (history["val_psnr"][index], history["val_ssim"][index]),
    )

    if recipe == "balanced_cosine":
        lr_values = history["lr"]
        if any(right > left + 1e-12 for left, right in zip(lr_values, lr_values[1:])):
            raise RuntimeError("Cosine learning rate is not monotonically decreasing")
        if not lr_values[0] < initial_lr or not math.isclose(lr_values[-1], 1e-6, abs_tol=1e-10):
            raise RuntimeError(
                f"Unexpected cosine LR endpoints: first={lr_values[0]:.3g}, last={lr_values[-1]:.3g}"
            )

    rows.append(
        {
            "recipe": recipe,
            "best_epoch": best_index + 1,
            "psnr": history["val_psnr"][best_index],
            "ssim": history["val_ssim"][best_index],
            "checkpoint": str(best_path),
            "history": str(history_path),
        }
    )

if len(sample_counts) != 1:
    raise RuntimeError(f"Train/validation counts differ across recipes: {sorted(sample_counts)}")

rows.sort(key=lambda row: (row["psnr"], row["ssim"]), reverse=True)
control = next(row for row in rows if row["recipe"] == "control")
for row in rows:
    row["psnr_gain_vs_control"] = row["psnr"] - control["psnr"]
    row["ssim_change_vs_control"] = row["ssim"] - control["ssim"]
    row["passes_promotion_rule"] = (
        row["recipe"] != "control"
        and row["psnr_gain_vs_control"] >= 0.20
        and row["ssim_change_vs_control"] >= -0.002
    )

winner = next((row for row in rows if row["passes_promotion_rule"]), None)
selection = winner["recipe"] if winner else "existing_unet_recipe"
train_count, val_count = next(iter(sample_counts))
report = {
    "screen_id": screen_id,
    "seed": seed,
    "train_samples": train_count,
    "validation_samples": val_count,
    "ranking": rows,
    "selection": selection,
    "confirmation_seeds": [123, 3407] if winner else [],
    "held_out_test_evaluated": False,
}
report_path = result_root / f"{screen_id}_summary.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")

print("\nTraining-recipe ranking (best validation PSNR; SSIM tie-break):")
print(f"{'recipe':<20} {'epoch':>5} {'PSNR':>8} {'SSIM':>8} {'dPSNR':>8} {'dSSIM':>8} {'promote':>8}")
for row in rows:
    print(
        f"{row['recipe']:<20} {row['best_epoch']:>5} {row['psnr']:>8.3f} "
        f"{row['ssim']:>8.4f} {row['psnr_gain_vs_control']:>+8.3f} "
        f"{row['ssim_change_vs_control']:>+8.4f} "
        f"{('yes' if row['passes_promotion_rule'] else 'no'):>8}"
    )
print(f"\nMatched split confirmed: train={train_count}, val={val_count}, seed={seed}")
if winner:
    print(f"Selection: {selection}; confirm with seeds 123 and 3407 before full EUVP training.")
else:
    print("Selection: retain the existing U-Net recipe; no candidate passed the promotion rule.")
print(f"Summary: {report_path}")
PY
