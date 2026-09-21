#!/usr/bin/env bash
# ==============================================================================
# run_sgmanet_all.sh
# ------------------
# Train SGMA-Net across EUVP and UIEB datasets for all channel variants:
#   - sgmanet_3ch   (RGB only)
#   - sgmanet_4ch_t (RGB + transmission prior t)
#   - sgmanet_4ch_b (RGB + background prior B)
#   - sgmanet_5ch   (RGB + t + B)
#
# Performs deep VRAM & system cache cleanup after each model before the next run.
#
# Usage:
#   bash scripts/experiments/run_sgmanet_all.sh
#
# Custom overrides:
#   N_EPOCHS=100 BATCH_SIZE=16 bash scripts/experiments/run_sgmanet_all.sh
# ==============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

# ------------------------------------------------------------------------------
# 1. Configurable Hyperparameters (override via env variables)
# ------------------------------------------------------------------------------
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CROP_SIZE="${CROP_SIZE:-256}"
LR="${LR:-1e-4}"
AMP="${AMP:-True}"
DATA_EUVP="${DATA_EUVP:-${PROJECT_DIR}/datasets/EUVP}"
DATA_UIEB="${DATA_UIEB:-${PROJECT_DIR}/datasets/UIEB}"
COOLDOWN_SECS="${COOLDOWN_SECS:-10}"

# Loss configuration (optimized for higher PSNR: Charbonnier + SSIM + Color + Wavelet)
USE_CHARBONNIER="${USE_CHARBONNIER:-True}"
L1_WEIGHT="${L1_WEIGHT:-1.0}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.3}"
COLOR_WEIGHT="${COLOR_WEIGHT:-0.2}"
WAVELET_WEIGHT="${WAVELET_WEIGHT:-0.1}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-0.05}"

# Datasets and model variants
DATASETS=("euvp" "uieb")
VARIANTS=("sgmanet_3ch" "sgmanet_4ch_t" "sgmanet_4ch_b" "sgmanet_5ch")

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"

# ------------------------------------------------------------------------------
# 2. Cleanup Function
# ------------------------------------------------------------------------------
clean_environment() {
    local finished_model="$1"
    local next_model="$2"

    echo ""
    echo "================================================================="
    echo " [CLEANUP] Cleaning VRAM & System Cache after: ${finished_model}"
    echo "================================================================="

    # 1. Clear PyTorch CUDA Cache and IPC
    python -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('>> PyTorch CUDA cache & IPC memory freed.')
" 2>/dev/null || true

    # 2. Sync file system buffers to disk
    sync

    # 3. Drop system memory caches (if root / accessible)
    if [ -w /proc/sys/vm/drop_caches ]; then
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
        echo ">> System pagecache dropped."
    fi

    # 4. Display GPU status
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
# 3. Main Sequential Execution Loop
# ------------------------------------------------------------------------------
START_TOTAL=$(date +%s)
TOTAL_RUNS=$((${#DATASETS[@]} * ${#VARIANTS[@]}))
RUN_INDEX=0

echo "================================================================="
echo " Starting SGMA-Net Full Suite Training (${TOTAL_RUNS} runs total)"
echo " Datasets : ${DATASETS[*]}"
echo " Variants : ${VARIANTS[*]}"
echo " Epochs   : ${N_EPOCHS}"
echo " Batch    : ${BATCH_SIZE}"
echo " AMP      : ${AMP}"
echo " Start at : $(date)"
echo "================================================================="

for ds in "${DATASETS[@]}"; do
    for model_name in "${VARIANTS[@]}"; do
        RUN_INDEX=$((RUN_INDEX + 1))
        TS=$(date +"%Y%m%d_%H%M%S")
        RUN_NAME="${model_name}_${ds}"
        LOG_FILE="${PROJECT_DIR}/logs/${RUN_NAME}_${TS}.log"

        echo "-----------------------------------------------------------------"
        echo " [Run ${RUN_INDEX}/${TOTAL_RUNS}] Dataset: ${ds} | Model: ${model_name}"
        echo " Started at: $(date)"
        echo " Logging to: ${LOG_FILE}"
        echo "-----------------------------------------------------------------"

        RUN_CMD=(
            python -m uwir.cli.train
            --model "${model_name}"
            --dataset "${ds}"
            --run_name "${RUN_NAME}"
            --nEpochs "${N_EPOCHS}"
            --batchSize "${BATCH_SIZE}"
            --cropSize "${CROP_SIZE}"
            --lr "${LR}"
            --amp "${AMP}"
        )

        if [ "${ds}" == "euvp" ]; then
            RUN_CMD+=(--data_train_euvp "${DATA_EUVP}")
        elif [ "${ds}" == "uieb" ]; then
            RUN_CMD+=(--data_train_uieb "${DATA_UIEB}")
        fi

        if [ "${USE_CHARBONNIER}" == "True" ]; then
            RUN_CMD+=(--use_charbonnier)
        fi
        RUN_CMD+=(
            --L1_weight "${L1_WEIGHT}"
            --SSIM_weight "${SSIM_WEIGHT}"
            --color_weight "${COLOR_WEIGHT}"
            --wavelet_weight "${WAVELET_WEIGHT}"
            --perceptual_weight "${PERCEPTUAL_WEIGHT}"
        )

        # Execute training and mirror output to log file
        "${RUN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo ">> [SUCCESS] Completed: ${RUN_NAME} at $(date)"
        else
            echo ">> [WARNING] Run failed with exit code ${EXIT_CODE}: ${RUN_NAME}"
        fi

        # Determine next model name for notification
        NEXT_MODEL=""
        if [ ${RUN_INDEX} -lt ${TOTAL_RUNS} ]; then
            NEXT_MODEL="next job"
        fi

        # Clean environment before the next run
        clean_environment "${RUN_NAME}" "${NEXT_MODEL}"
    done
done

END_TOTAL=$(date +%s)
DURATION=$((END_TOTAL - START_TOTAL))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "================================================================="
echo " All ${TOTAL_RUNS} training jobs finished!"
echo " Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo " Finished at   : $(date)"
echo "================================================================="
