#!/usr/bin/env bash
# ==============================================================================
# eval_sgmanet_all.sh
# --------------------
# Evaluate all trained SGMA-Net checkpoints on EUVP and UIEB test sets.
#
# Usage:
#   bash scripts/experiments/eval_sgmanet_all.sh
# ==============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

echo "================================================================="
echo " [1/2] Evaluating SGMA-Net EUVP Models on EUVP Test Set (n=515)"
echo "================================================================="
python -m uwir.cli.evaluate \
    --checkpoint_dir ./checkpoints \
    --run_filter euvp \
    --eval_benchmark euvp \
    --data_train_euvp ./datasets/EUVP \
    --val_folder ./results/eval_sgmanet_euvp

echo ""
echo "================================================================="
echo " [2/2] Evaluating SGMA-Net UIEB Models on UIEB Test Set (n=90)"
echo "================================================================="
python -m uwir.cli.evaluate \
    --checkpoint_dir ./checkpoints \
    --run_filter uieb \
    --eval_benchmark uieb \
    --data_train_uieb ./datasets/UIEB \
    --val_folder ./results/eval_sgmanet_uieb

echo ""
echo "================================================================="
echo " All SGMA-Net evaluations completed!"
echo " Results saved to:"
echo "   - ./results/eval_sgmanet_euvp/test_results_all.json"
echo "   - ./results/eval_sgmanet_uieb/test_results_all.json"
echo "================================================================="
