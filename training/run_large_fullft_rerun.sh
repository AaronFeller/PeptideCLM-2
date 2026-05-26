#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-tmp/fullft_large_rerun_${RUN_STAMP}/runs_full_regression_v2}
TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT:-tmp/fullft_large_rerun_${RUN_STAMP}/logs/regression_full_v2}
MLM_GPU_INDEX=${MLM_GPU_INDEX:-0}
MTR_GPU_INDEX=${MTR_GPU_INDEX:-1}
HYBRID_GPU_INDEX=${HYBRID_GPU_INDEX:-2}

COMMON_ARGS=(
    --seed 101
    --output_root "$OUTPUT_ROOT"
    --log_root "$TRAIN_LOG_ROOT"
    --max_steps 2500
    --patience 3
    --learning_rate 5e-6
    --target_scaling none
    --force
)

CUDA_VISIBLE_DEVICES=$MLM_GPU_INDEX "$PYTHON_BIN" training/regression_finetune_ensemble_full_v2.py \
    --models aaronfeller/peptideclm-2-mlm-large "${COMMON_ARGS[@]}" --gpu_index 0 &
mlm_pid=$!

CUDA_VISIBLE_DEVICES=$MTR_GPU_INDEX "$PYTHON_BIN" training/regression_finetune_ensemble_full_v2.py \
    --models aaronfeller/peptideclm-2-mtr-large "${COMMON_ARGS[@]}" --gpu_index 0 &
mtr_pid=$!

CUDA_VISIBLE_DEVICES=$HYBRID_GPU_INDEX "$PYTHON_BIN" training/regression_finetune_ensemble_full_v2.py \
    --models aaronfeller/peptideclm-2-hybrid-large "${COMMON_ARGS[@]}" --gpu_index 0 &
hybrid_pid=$!

wait "$mlm_pid" "$mtr_pid" "$hybrid_pid"