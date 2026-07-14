#!/usr/bin/env bash
exec bash "$(dirname "$0")/scripts/experiments/run_hybrid_mamba_ablation.sh" "$@"
