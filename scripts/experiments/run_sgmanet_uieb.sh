#!/usr/bin/env bash
# ==============================================================================
# run_sgmanet_uieb.sh
# --------------------
# Train and benchmark all 4 SGMA-Net input variants on the UIEB dataset inside WSL:
#   - sgmanet_3ch   (Standard RGB)
#   - sgmanet_4ch_t (RGB + Transmission prior t)
#   - sgmanet_4ch_b (RGB + Background light prior B)
#   - sgmanet_5ch   (RGB + Transmission t + Background B)
#
# Features:
#   - Utilizes hardware-accelerated fused CUDA Mamba kernel (selective_scan_fn)
#   - Cleans VRAM and memory caches between variants
#   - Automatically evaluates on UIEB test-90 benchmark upon completion
#
# Usage:
#   bash scripts/experiments/run_sgmanet_uieb.sh
#
# Custom overrides:
#   N_EPOCHS=100 BATCH_SIZE=16 bash scripts/experiments/run_sgmanet_uieb.sh
# ==============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

# ------------------------------------------------------------------------------
# 1. Environment & Python Setup
# ------------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-uwir}"
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
elif [ -f "/home/hungthinh1914/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/hungthinh1914/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

PYTHON="${PYTHON:-python}"

# ------------------------------------------------------------------------------
# 2. Configurable Hyperparameters
# ------------------------------------------------------------------------------
N_EPOCHS="${N_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CROP_SIZE="${CROP_SIZE:-256}"
LR="${LR:-1e-4}"
AMP="${AMP:-True}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
THREADS="${THREADS:-4}"
IN_MEMORY="${IN_MEMORY:-True}"
DATA_UIEB="${DATA_UIEB:-${PROJECT_DIR}/datasets/UIEB}"
COOLDOWN_SECS="${COOLDOWN_SECS:-5}"

# Loss configuration
L1_WEIGHT="${L1_WEIGHT:-1.0}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.1}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-1.0}"

VARIANTS=("sgmanet_3ch" "sgmanet_4ch_t" "sgmanet_4ch_b" "sgmanet_5ch")

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/results/eval_sgmanet_uieb"

# ------------------------------------------------------------------------------
# 3. Memory Cleanup Function
# ------------------------------------------------------------------------------
clean_environment() {
    local finished_model="$1"
    local next_model="$2"

    echo ""
    echo "================================================================="
    echo " [CLEANUP] Cleaning VRAM & System Cache after: ${finished_model}"
    echo "================================================================="

    "${PYTHON}" -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('>> PyTorch CUDA cache & IPC memory freed.')
" 2>/dev/null || true

    if command -v nvidia-smi &> /dev/null; then
        echo -n ">> Current GPU VRAM used: "
        nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv,noheader
    fi

    if [ -n "${next_model}" ]; then
        echo ">> Cooling down for ${COOLDOWN_SECS} seconds before training ${next_model}..."
        sleep "${COOLDOWN_SECS}"
    fi
    echo "================================================================="
    echo ""
}

# ------------------------------------------------------------------------------
# 4. Training Loop
# ------------------------------------------------------------------------------
START_TOTAL=$(date +%s)
TOTAL_RUNS=${#VARIANTS[@]}
RUN_INDEX=0

echo "================================================================="
echo " Starting SGMA-Net UIEB Training Suite (${TOTAL_RUNS} variants total)"
echo " Project Dir : ${PROJECT_DIR}"
echo " Python Path : $(which ${PYTHON})"
echo " Variants    : ${VARIANTS[*]}"
echo " Epochs      : ${N_EPOCHS}"
echo " Batch Size  : ${BATCH_SIZE}"
echo " AMP (FP16)  : ${AMP}"
echo " Dataset     : UIEB (${DATA_UIEB})"
echo " Start at    : $(date)"
echo "================================================================="

for model_name in "${VARIANTS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    TS=$(date +"%Y%m%d_%H%M%S")
    RUN_NAME="${model_name}_uieb"
    LOG_FILE="${PROJECT_DIR}/logs/${RUN_NAME}_${TS}.log"

    echo "-----------------------------------------------------------------"
    echo " [Variant ${RUN_INDEX}/${TOTAL_RUNS}] Training: ${model_name}"
    echo " Started at: $(date)"
    echo " Logging to: ${LOG_FILE}"
    echo "-----------------------------------------------------------------"

    RUN_CMD=(
        "${PYTHON}" -m uwir.cli.train
        --model "${model_name}"
        --dataset uieb
        --data_train_uieb "${DATA_UIEB}"
        --run_name "${RUN_NAME}"
        --nEpochs "${N_EPOCHS}"
        --batchSize "${BATCH_SIZE}"
        --cropSize "${CROP_SIZE}"
        --lr "${LR}"
        --amp "${AMP}"
        --amp_dtype "${AMP_DTYPE}"
        --threads "${THREADS}"
        --in_memory "${IN_MEMORY}"
        --prior_method udcp
        --L1_weight "${L1_WEIGHT}"
        --perceptual_weight "${PERCEPTUAL_WEIGHT}"
        --SSIM_weight "${SSIM_WEIGHT}"
        --checkpoint_dir "${PROJECT_DIR}/checkpoints"
    )

    "${RUN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo ">> [SUCCESS] Finished training: ${model_name}"
    else
        echo ">> [ERROR] Training failed for ${model_name} with exit code ${EXIT_CODE}"
    fi

    NEXT_MODEL=""
    if [ ${RUN_INDEX} -lt ${TOTAL_RUNS} ]; then
        NEXT_MODEL="${VARIANTS[${RUN_INDEX}]}"
    fi

    clean_environment "${model_name}" "${NEXT_MODEL}"
done

# ------------------------------------------------------------------------------
# 5. Automated Evaluation on UIEB Test Set (90 images)
# ------------------------------------------------------------------------------
echo "================================================================="
echo " Running evaluation on UIEB test set for all trained checkpoints"
echo "================================================================="
"${PYTHON}" -m uwir.cli.evaluate \
    --checkpoint_dir "${PROJECT_DIR}/checkpoints" \
    --run_filter uieb \
    --eval_benchmark uieb \
    --data_train_uieb "${DATA_UIEB}" \
    --val_folder "${PROJECT_DIR}/results/eval_sgmanet_uieb"

END_TOTAL=$(date +%s)
DURATION=$((END_TOTAL - START_TOTAL))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "================================================================="
echo " All SGMA-Net UIEB experiments completed successfully!"
echo " Total Elapsed Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo " Checkpoints directory: ${PROJECT_DIR}/checkpoints"
echo " Evaluation results  : ${PROJECT_DIR}/results/eval_sgmanet_uieb/test_results_all.json"
echo "================================================================="
