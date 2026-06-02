#!/usr/bin/env bash
set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

cleanup_background_jobs() {
    local pid
    for pid in "$@"; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    for pid in "$@"; do
        [[ -z "$pid" ]] && continue
        wait "$pid" >/dev/null 2>&1 || true
    done
}

wait_and_report() {
    local pid=$1
    local label=$2
    if wait "$pid"; then
        log "$label finished successfully (pid=$pid)"
    else
        local exit_code=$?
        log "$label failed (pid=$pid exit_code=$exit_code)"
        return "$exit_code"
    fi
}

launch_model_job() {
    local label=$1
    local gpu_index=$2
    local model_name=$3
    local log_path=$4

    log "Launching $label on GPU $gpu_index with one model per holdout"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu_index "$PYTHON_BIN" training/regression_finetune_holdout_random_v1.py \
        --models "$model_name" "${COMMON_ARGS[@]}" --gpu_index 0 \
        >"$log_path" 2>&1 &
    LAST_BG_PID=$!
    log "$label pid=$LAST_BG_PID log=$log_path"
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-tmp/holdout_random_medium_${RUN_STAMP}/runs_holdout_random_v1}
TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT:-tmp/holdout_random_medium_${RUN_STAMP}/logs/regression_holdout_random_v1}
LAUNCH_LOG_DIR=${LAUNCH_LOG_DIR:-tmp/holdout_random_medium_${RUN_STAMP}/launcher_logs}
MLM_GPU_INDEX=${MLM_GPU_INDEX:-1}
MTR_GPU_INDEX=${MTR_GPU_INDEX:-2}
HYBRID_GPU_INDEX=${HYBRID_GPU_INDEX:-3}

mkdir -p "$LAUNCH_LOG_DIR"

COMMON_ARGS=(
    --seed 101
    --output_root "$OUTPUT_ROOT"
    --log_root "$TRAIN_LOG_ROOT"
    --batch_size 8
    --eval_batch_size 64
    --accumulate_grad_batches 2
    --max_epochs 10
    --patience 4
    --val_check_interval 0.2
    --val_fraction 0.2
    --val_stratify_bins 5
    --oversample_train_bins 5
    --head_dropout 0.20
    --force
)

if [[ "${RANDOM_INIT_BACKBONE:-0}" == "1" ]]; then
    COMMON_ARGS+=(--random_init_backbone)
fi

MLM_STDOUT_LOG="$LAUNCH_LOG_DIR/mlm_medium_${RUN_STAMP}.log"
MTR_STDOUT_LOG="$LAUNCH_LOG_DIR/mtr_medium_${RUN_STAMP}.log"
HYBRID_STDOUT_LOG="$LAUNCH_LOG_DIR/hybrid_medium_${RUN_STAMP}.log"

log "Starting medium holdout-random v1 run"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "TRAIN_LOG_ROOT=$TRAIN_LOG_ROOT"
log "Batching: batch_size=8 eval_batch_size=64 accumulate_grad_batches=2 effective_batch_size=16"
log "Random backbone init: RANDOM_INIT_BACKBONE=${RANDOM_INIT_BACKBONE:-0}"
log "GPU assignment: MLM=$MLM_GPU_INDEX MTR=$MTR_GPU_INDEX HYBRID=$HYBRID_GPU_INDEX"

launch_model_job "MLM medium" "$MLM_GPU_INDEX" "aaronfeller/peptideclm-2-mlm-base" "$MLM_STDOUT_LOG"
mlm_pid=$LAST_BG_PID
launch_model_job "MTR medium" "$MTR_GPU_INDEX" "aaronfeller/peptideclm-2-mtr-base" "$MTR_STDOUT_LOG"
mtr_pid=$LAST_BG_PID
launch_model_job "Hybrid medium" "$HYBRID_GPU_INDEX" "aaronfeller/peptideclm-2-hybrid-base" "$HYBRID_STDOUT_LOG"
hybrid_pid=$LAST_BG_PID

trap 'cleanup_background_jobs "$mlm_pid" "$mtr_pid" "$hybrid_pid"' EXIT

wait_and_report "$mlm_pid" "MLM medium"
wait_and_report "$mtr_pid" "MTR medium"
wait_and_report "$hybrid_pid" "Hybrid medium"
trap - EXIT
log "Medium holdout-random v1 run finished"