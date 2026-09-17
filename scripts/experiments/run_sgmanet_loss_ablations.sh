#!/usr/bin/env bash
# ==============================================================================
# run_sgmanet_loss_ablations.sh
# -----------------------------
# Systematic Loss Ablation Suite for SGMA-Net on UIEB (800 train / 90 test) + EUVP:
#
#   Variant 1: (x1 T4) Base loss (L1 + L_perc 1:1) + TV loss
#   Variant 2: (x2 T4) Base loss (L1 + L_perc 1:1) + TV loss (Parallel Speed Benchmark)
#   Variant 3: (x2 T4) Base loss (L1 + L_perc 1:1) + Edge loss
#   Variant 4: (x2 T4) Base loss (L1 + L_perc 1:1) + Local Variance loss (MobileIE)
#   Variant 5: (x2 T4) Base loss (L1 + L_perc 1:1) + UIQM loss
#
# Usage:
#   bash scripts/experiments/run_sgmanet_loss_ablations.sh
#
# Config overrides:
#   N_EPOCHS=100 BATCH_SIZE=16 bash scripts/experiments/run_sgmanet_loss_ablations.sh
# ==============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

# ------------------------------------------------------------------------------
# 1. Configurable Parameters
# ------------------------------------------------------------------------------
MODEL_VARIANT="${MODEL_VARIANT:-sgmanet_5ch}"
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CROP_SIZE="${CROP_SIZE:-256}"
LR="${LR:-1e-4}"
AMP="${AMP:-True}"
DATA_UIEB="${DATA_UIEB:-${PROJECT_DIR}/datasets/UIEB}"
DATA_EUVP="${DATA_EUVP:-${PROJECT_DIR}/datasets/EUVP}"
UIEB_LIMIT="${UIEB_LIMIT:-800}"
COOLDOWN_SECS="${COOLDOWN_SECS:-5}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/results"

clean_gpu_memory() {
    python3 -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
" 2>/dev/null || true
    sync
    sleep "${COOLDOWN_SECS}"
}

# ------------------------------------------------------------------------------
# 2. Ablation Variant Definitions
# Format: Name | Num_GPUs | use_l1 | use_perc | use_ssim | use_tv | use_edge | use_lvw | use_uiqm | use_hvi
# ------------------------------------------------------------------------------
ABLATION_RUNS=(
    "sgmanet_5ch_uieb_base_only|2|1|1|0|0|0|0|0|0"
    "sgmanet_5ch_uieb_base_ssim_2gpu|2|1|1|1|0|0|0|0|0"
    "sgmanet_5ch_uieb_base_hvi_2gpu|2|1|1|0|0|0|0|0|1"
    "sgmanet_5ch_uieb_base_tv_1gpu|1|1|1|0|1|0|0|0|0"
    "sgmanet_5ch_uieb_base_tv_2gpu|2|1|1|0|1|0|0|0|0"
    "sgmanet_5ch_uieb_base_edge_2gpu|2|1|1|0|0|1|0|0|0"
    "sgmanet_5ch_uieb_base_lvw_2gpu|2|1|1|0|0|0|1|0|0"
    "sgmanet_5ch_uieb_base_uiqm_2gpu|2|1|1|0|0|0|0|1|0"
)

TOTAL_RUNS=${#ABLATION_RUNS[@]}
RUN_INDEX=0

echo "================================================================="
echo " Starting SGMA-Net Loss Ablation Suite (${TOTAL_RUNS} runs)"
echo " Model      : ${MODEL_VARIANT}"
echo " Dataset    : UIEB (train: ${UIEB_LIMIT} images, test: 90 images)"
echo " Epochs     : ${N_EPOCHS}"
echo " Batch Size : ${BATCH_SIZE}"
echo " Base Loss  : L1 (1.0) + L_perc (1.0)"
echo " Start at   : $(date)"
echo "================================================================="

for run_cfg in "${ABLATION_RUNS[@]}"; do
    IFS="|" read -r RUN_NAME NUM_GPUS USE_L1 USE_PERC USE_SSIM USE_TV USE_EDGE USE_LVW USE_UIQM USE_HVI <<< "${run_cfg}"
    RUN_INDEX=$((RUN_INDEX + 1))
    TS=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_DIR}/logs/${RUN_NAME}_${TS}.log"

    echo ""
    echo "-----------------------------------------------------------------"
    echo " [Run ${RUN_INDEX}/${TOTAL_RUNS}] ${RUN_NAME}"
    echo " GPUs: ${NUM_GPUS} | Toggles: L1=${USE_L1}, Perc=${USE_PERC}, SSIM=${USE_SSIM}, TV=${USE_TV}, Edge=${USE_EDGE}, LVW=${USE_LVW}, UIQM=${USE_UIQM}, HVI=${USE_HVI}"
    echo " Log : ${LOG_FILE}"
    echo "-----------------------------------------------------------------"

    START_T=$(date +%s)

    python3 -m uwir.cli.train \
        --model "${MODEL_VARIANT}" \
        --dataset "uieb" \
        --data_train_uieb "${DATA_UIEB}" \
        --uieb_limit "${UIEB_LIMIT}" \
        --run_name "${RUN_NAME}" \
        --nEpochs "${N_EPOCHS}" \
        --batchSize "${BATCH_SIZE}" \
        --cropSize "${CROP_SIZE}" \
        --lr "${LR}" \
        --amp "${AMP}" \
        --num_gpus "${NUM_GPUS}" \
        --use_l1 "${USE_L1}" \
        --use_perc "${USE_PERC}" \
        --use_ssim "${USE_SSIM}" \
        --use_tv "${USE_TV}" \
        --use_edge "${USE_EDGE}" \
        --use_lvw "${USE_LVW}" \
        --use_uiqm "${USE_UIQM}" \
        --use_hvi "${USE_HVI}" 2>&1 | tee "${LOG_FILE}"

    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))
    echo ">> Completed in ${ELAPSED}s"

    clean_gpu_memory
done

echo ""
echo "================================================================="
echo " Evaluating all ablation checkpoints on UIEB-90 and EUVP"
echo "================================================================="

python3 -m uwir.cli.evaluate \
    --checkpoint_dir ./checkpoints \
    --eval_benchmark uieb+euvp \
    --data_train_uieb "${DATA_UIEB}" \
    --data_train_euvp "${DATA_EUVP}" \
    --val_folder ./results/eval_loss_ablations

echo ""
echo " All ablation runs and evaluations finished successfully!"
