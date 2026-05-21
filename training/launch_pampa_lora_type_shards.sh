#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
MLM_GPU_INDEX=${MLM_GPU_INDEX:-0}
MTR_GPU_INDEX=${MTR_GPU_INDEX:-1}
HYBRID_GPU_INDEX=${HYBRID_GPU_INDEX:-2}
LOG_DIR=${LOG_DIR:-tmp/logs/pampa_lora_type_shards}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
DRY_RUN=1

SEED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mlm-gpu-index)
            MLM_GPU_INDEX="$2"
            shift 2
            ;;
        --mtr-gpu-index)
            MTR_GPU_INDEX="$2"
            shift 2
            ;;
        --hybrid-gpu-index)
            HYBRID_GPU_INDEX="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --run-stamp)
            RUN_STAMP="$2"
            shift 2
            ;;
        --seed)
            SEED_ARGS=("--seed" "$2")
            shift 2
            ;;
        --seeds)
            shift
            SEED_ARGS=("--seeds")
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SEED_ARGS+=("$1")
                shift
            done
            ;;
        --execute)
            DRY_RUN=0
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if command -v stdbuf >/dev/null 2>&1; then
    PYTHON_STREAM_PREFIX="PYTHONUNBUFFERED=1 stdbuf -oL -eL"
else
    PYTHON_STREAM_PREFIX="PYTHONUNBUFFERED=1"
fi

run_logged_async() {
    local log_file="$1"
    shift
    local command="$*"
    echo "$command > $log_file 2>&1"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$(dirname "$ROOT_DIR/$log_file")"
        (cd "$ROOT_DIR" && eval "$command" > "$log_file" 2>&1) &
        LAST_BG_PID=$!
    fi
}

run_logged_sync() {
    local log_file="$1"
    shift
    local command="$*"
    echo "$command > $log_file 2>&1"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$(dirname "$ROOT_DIR/$log_file")"
        (cd "$ROOT_DIR" && eval "$command" > "$log_file" 2>&1)
    fi
}

PREP_LOG_FILE="$LOG_DIR/prepare_${RUN_STAMP}.log"
MLM_SMALL_LOG_FILE="$LOG_DIR/mlm_small_${RUN_STAMP}.log"
MLM_BASE_LOG_FILE="$LOG_DIR/mlm_base_${RUN_STAMP}.log"
MLM_LARGE_LOG_FILE="$LOG_DIR/mlm_large_${RUN_STAMP}.log"
MTR_SMALL_LOG_FILE="$LOG_DIR/mtr_small_${RUN_STAMP}.log"
MTR_BASE_LOG_FILE="$LOG_DIR/mtr_base_${RUN_STAMP}.log"
MTR_LARGE_LOG_FILE="$LOG_DIR/mtr_large_${RUN_STAMP}.log"
HYBRID_SMALL_LOG_FILE="$LOG_DIR/hybrid_small_${RUN_STAMP}.log"
HYBRID_BASE_LOG_FILE="$LOG_DIR/hybrid_base_${RUN_STAMP}.log"
HYBRID_LARGE_LOG_FILE="$LOG_DIR/hybrid_large_${RUN_STAMP}.log"

SEED_FLAGS=""
if [[ ${#SEED_ARGS[@]} -gt 0 ]]; then
    SEED_FLAGS="${SEED_ARGS[*]}"
fi

PREP_CMD="$PYTHON_BIN training/prepare_benchmark_data.py --task cycpeptmpdb_perm --seed 0"
MLM_SMALL_CMD="CUDA_VISIBLE_DEVICES=$MLM_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-mlm-small $SEED_FLAGS --gpu_index 0"
MLM_BASE_CMD="CUDA_VISIBLE_DEVICES=$MLM_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-mlm-base $SEED_FLAGS --gpu_index 0"
MLM_LARGE_CMD="CUDA_VISIBLE_DEVICES=$MLM_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-mlm-large $SEED_FLAGS --gpu_index 0"
MTR_SMALL_CMD="CUDA_VISIBLE_DEVICES=$MTR_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-mtr-small $SEED_FLAGS --gpu_index 0"
MTR_BASE_CMD="CUDA_VISIBLE_DEVICES=$MTR_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-mtr-base $SEED_FLAGS --gpu_index 0"
MTR_LARGE_CMD="CUDA_VISIBLE_DEVICES=$MTR_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-mtr-large $SEED_FLAGS --gpu_index 0"
HYBRID_SMALL_CMD="CUDA_VISIBLE_DEVICES=$HYBRID_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-hybrid-small $SEED_FLAGS --gpu_index 0"
HYBRID_BASE_CMD="CUDA_VISIBLE_DEVICES=$HYBRID_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-hybrid-base $SEED_FLAGS --gpu_index 0"
HYBRID_LARGE_CMD="CUDA_VISIBLE_DEVICES=$HYBRID_GPU_INDEX $PYTHON_STREAM_PREFIX $PYTHON_BIN training/regression_finetune_ensemble_v2.py --models aaronfeller/peptideclm-2-hybrid-large $SEED_FLAGS --gpu_index 0"

run_logged_sync "$PREP_LOG_FILE" "$PREP_CMD"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[phase 1] launching small and base jobs"
    run_logged_async "$MLM_SMALL_LOG_FILE" "$MLM_SMALL_CMD"
    run_logged_async "$MLM_BASE_LOG_FILE" "$MLM_BASE_CMD"
    run_logged_async "$MTR_SMALL_LOG_FILE" "$MTR_SMALL_CMD"
    run_logged_async "$MTR_BASE_LOG_FILE" "$MTR_BASE_CMD"
    run_logged_async "$HYBRID_SMALL_LOG_FILE" "$HYBRID_SMALL_CMD"
    run_logged_async "$HYBRID_BASE_LOG_FILE" "$HYBRID_BASE_CMD"
    echo "[phase 2] after small/base complete, launching large jobs"
    run_logged_async "$MLM_LARGE_LOG_FILE" "$MLM_LARGE_CMD"
    run_logged_async "$MTR_LARGE_LOG_FILE" "$MTR_LARGE_CMD"
    run_logged_async "$HYBRID_LARGE_LOG_FILE" "$HYBRID_LARGE_CMD"
else
    run_logged_async "$MLM_SMALL_LOG_FILE" "$MLM_SMALL_CMD"
    mlm_small_pid="$LAST_BG_PID"
    run_logged_async "$MLM_BASE_LOG_FILE" "$MLM_BASE_CMD"
    mlm_base_pid="$LAST_BG_PID"
    run_logged_async "$MTR_SMALL_LOG_FILE" "$MTR_SMALL_CMD"
    mtr_small_pid="$LAST_BG_PID"
    run_logged_async "$MTR_BASE_LOG_FILE" "$MTR_BASE_CMD"
    mtr_base_pid="$LAST_BG_PID"
    run_logged_async "$HYBRID_SMALL_LOG_FILE" "$HYBRID_SMALL_CMD"
    hybrid_small_pid="$LAST_BG_PID"
    run_logged_async "$HYBRID_BASE_LOG_FILE" "$HYBRID_BASE_CMD"
    hybrid_base_pid="$LAST_BG_PID"
    echo "MLM small log: $ROOT_DIR/$MLM_SMALL_LOG_FILE"
    echo "MLM base log: $ROOT_DIR/$MLM_BASE_LOG_FILE"
    echo "MLM large log: $ROOT_DIR/$MLM_LARGE_LOG_FILE"
    echo "MTR small log: $ROOT_DIR/$MTR_SMALL_LOG_FILE"
    echo "MTR base log: $ROOT_DIR/$MTR_BASE_LOG_FILE"
    echo "MTR large log: $ROOT_DIR/$MTR_LARGE_LOG_FILE"
    echo "Hybrid small log: $ROOT_DIR/$HYBRID_SMALL_LOG_FILE"
    echo "Hybrid base log: $ROOT_DIR/$HYBRID_BASE_LOG_FILE"
    echo "Hybrid large log: $ROOT_DIR/$HYBRID_LARGE_LOG_FILE"

    wait "$mlm_small_pid" "$mlm_base_pid" "$mtr_small_pid" "$mtr_base_pid" "$hybrid_small_pid" "$hybrid_base_pid"

    run_logged_async "$MLM_LARGE_LOG_FILE" "$MLM_LARGE_CMD"
    mlm_large_pid="$LAST_BG_PID"
    run_logged_async "$MTR_LARGE_LOG_FILE" "$MTR_LARGE_CMD"
    mtr_large_pid="$LAST_BG_PID"
    run_logged_async "$HYBRID_LARGE_LOG_FILE" "$HYBRID_LARGE_CMD"
    hybrid_large_pid="$LAST_BG_PID"

    wait "$mlm_large_pid" "$mtr_large_pid" "$hybrid_large_pid"
fi