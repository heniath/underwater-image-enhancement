#!/usr/bin/env bash
# ==============================================================================
# run_sgmanet_edge_and_ssim_runs.sh
# ---------------------------------
# Runs SGMA-Net with:
#   1. Base + Edge Loss (weight = 1.0)
#   2. Base + Edge Loss (weight = 2.0)
#   3. Base + Edge Loss (weight = 10.0)
#   4. Base + TV Loss (weight = 1.0, user requested direct test)
#   5. Base + TV Loss (weight = 0.001, literature paper standard e.g. UNTV 2022)
#   6. Base + TV Loss (weight = 0.01, balanced regularizer)
#   7. Base + SSIM Loss (full 100 epochs, early_stop_patience=100)
#
# Suitable for both Local Machine (RTX 3060 / RTX 5060) and Server environments.
# To filter runs, pass e.g.: RUN_FILTER="tv" bash scripts/experiments/run_sgmanet_edge_and_ssim_runs.sh
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_VARIANT="${MODEL_VARIANT:-sgmanet_5ch}"
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CROP_SIZE="${CROP_SIZE:-256}"
LR="${LR:-1e-4}"
AMP="${AMP:-True}"
IN_MEMORY="${IN_MEMORY:-False}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"

if [ -z "${DATA_UIEB:-}" ]; then
    if [ -d "/root/Dataset/UIEB" ]; then
        DATA_UIEB="/root/Dataset/UIEB"
    elif [ -d "${PROJECT_DIR}/../Dataset/UIEB" ]; then
        DATA_UIEB="${PROJECT_DIR}/../Dataset/UIEB"
    else
        DATA_UIEB="${PROJECT_DIR}/datasets/UIEB"
    fi
fi

if [ -z "${DATA_EUVP:-}" ]; then
    if [ -d "/root/Dataset/EUVP" ]; then
        DATA_EUVP="/root/Dataset/EUVP"
    elif [ -d "${PROJECT_DIR}/../Dataset/EUVP" ]; then
        DATA_EUVP="${PROJECT_DIR}/../Dataset/EUVP"
    else
        DATA_EUVP="${PROJECT_DIR}/datasets/EUVP"
    fi
fi
UIEB_LIMIT="${UIEB_LIMIT:-800}"
NUM_GPUS="${NUM_GPUS:-1}"
RUN_FILTER="${RUN_FILTER:-}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/checkpoints"
mkdir -p "${PROJECT_DIR}/results"

if [ -f "/opt/venv-uwir/bin/python3" ]; then
    PYTHON_CMD="/opt/venv-uwir/bin/python3"
else
    PYTHON_CMD="${PYTHON_CMD:-python3}"
fi

clean_gpu_memory() {
    ${PYTHON_CMD} -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
" 2>/dev/null || true
    sync
    sleep 3
}

ALL_RUNS=(
    "sgmanet_5ch_uieb_edge_w1|--use_l1 1 --use_perc 1 --use_edge 1 --edge_weight 1.0"
    "sgmanet_5ch_uieb_edge_w2|--use_l1 1 --use_perc 1 --use_edge 1 --edge_weight 2.0"
    "sgmanet_5ch_uieb_edge_w10|--use_l1 1 --use_perc 1 --use_edge 1 --edge_weight 10.0"
    "sgmanet_5ch_uieb_tv_w1|--use_l1 1 --use_perc 1 --use_tv 1 --tv_weight 1.0"
    "sgmanet_5ch_uieb_tv_w0001|--use_l1 1 --use_perc 1 --use_tv 1 --tv_weight 0.001"
    "sgmanet_5ch_uieb_tv_w001|--use_l1 1 --use_perc 1 --use_tv 1 --tv_weight 0.01"
    "sgmanet_5ch_uieb_ssim_full|--use_l1 1 --use_perc 1 --use_ssim 1 --SSIM_weight 0.1 --early_stop_patience 100"
)

RUNS=()
for item in "${ALL_RUNS[@]}"; do
    if [[ -z "${RUN_FILTER}" || "${item}" == *"${RUN_FILTER}"* ]]; then
        RUNS+=("${item}")
    fi
done

TOTAL_RUNS=${#RUNS[@]}
RUN_IDX=0

echo "================================================================="
echo " Starting SGMA-Net Ablation Experiments (${TOTAL_RUNS} runs)"
echo " Filter       : ${RUN_FILTER:-ALL}"
echo " Model        : ${MODEL_VARIANT}"
echo " Epochs       : ${N_EPOCHS}"
echo " Batch Size   : ${BATCH_SIZE}"
echo " In-Memory    : ${IN_MEMORY}"
echo " Val Interval : ${VAL_INTERVAL}"
echo " Dataset UIEB : ${DATA_UIEB}"
echo " Start at     : $(date)"
echo "================================================================="

for run_item in "${RUNS[@]}"; do
    IFS="|" read -r RUN_NAME LOSS_FLAGS <<< "${run_item}"
    RUN_IDX=$((RUN_IDX + 1))
    TS=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${PROJECT_DIR}/logs/${RUN_NAME}_${TS}.log"

    echo ""
    echo "-----------------------------------------------------------------"
    echo " [Run ${RUN_IDX}/${TOTAL_RUNS}] ${RUN_NAME}"
    echo " Flags: ${LOSS_FLAGS}"
    echo " Log  : ${LOG_FILE}"
    echo "-----------------------------------------------------------------"

    START_T=$(date +%s)

    ${PYTHON_CMD} -m uwir.cli.train \
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
        --in_memory "${IN_MEMORY}" \
        --val_interval "${VAL_INTERVAL}" \
        --num_gpus "${NUM_GPUS}" \
        ${LOSS_FLAGS} 2>&1 | tee "${LOG_FILE}"

    END_T=$(date +%s)
    ELAPSED=$((END_T - START_T))
    echo ">> [Run ${RUN_IDX}] Completed in ${ELAPSED}s"

    clean_gpu_memory
done

echo ""
echo "================================================================="
echo " Evaluating new checkpoints on UIEB-90 and EUVP benchmarks"
echo "================================================================="

${PYTHON_CMD} -m uwir.cli.evaluate \
    --checkpoint_dir ./checkpoints \
    --eval_benchmark uieb+euvp \
    --data_train_uieb "${DATA_UIEB}" \
    --data_train_euvp "${DATA_EUVP}" \
    --val_folder ./results/eval_edge_ssim_runs

echo ""
echo " All runs and benchmark evaluations completed!"
