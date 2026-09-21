#!/usr/bin/env bash
# =============================================================================
# run_ablation_loss.sh  –  Run the Loss Function Ablation Study on SGMA-Net in WSL
#
# Usage:
#   bash scripts/experiments/run_ablation_loss.sh
#   MODEL=sgmanet_3ch DATASET=euvp N_EPOCHS=20 bash scripts/experiments/run_ablation_loss.sh
#   MODEL=sgmanet_5ch DATASET=uieb N_EPOCHS=20 bash scripts/experiments/run_ablation_loss.sh
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-uwir}"
MODEL="${MODEL:-sgmanet_3ch}"
DATASET="${DATASET:-euvp}"
N_EPOCHS="${N_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CROP_SIZE="${CROP_SIZE:-256}"
SEED="${SEED:-0}"
EXPERIMENTS="${EXPERIMENTS:-}"

echo "============================================================================="
echo "  UWIR - Loss Functions Ablation Study"
echo "============================================================================="
echo "  Project Dir : ${PROJECT_DIR}"
echo "  Model       : ${MODEL}"
echo "  Dataset     : ${DATASET}"
echo "  Epochs/run  : ${N_EPOCHS}"
echo "  Batch Size  : ${BATCH_SIZE}"
echo "  Seed        : ${SEED}"
echo "============================================================================="

# Ensure conda env is activated if conda is available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" || true
elif [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" || true
fi

CMD=(python "${PROJECT_DIR}/scripts/experiments/ablation_loss.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --nEpochs "${N_EPOCHS}" \
    --batchSize "${BATCH_SIZE}" \
    --cropSize "${CROP_SIZE}" \
    --seed "${SEED}")

if [ -n "${EXPERIMENTS}" ]; then
    CMD+=(--experiments ${EXPERIMENTS})
fi

echo "Running command: ${CMD[*]}"
"${CMD[@]}"
