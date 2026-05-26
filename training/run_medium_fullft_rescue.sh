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

check_cuda_runtime() {
    local gpu_index=$1
    local role=$2
    if ! CUDA_VISIBLE_DEVICES="$gpu_index" "$PYTHON_BIN" -c "import torch; device = torch.device('cuda:0'); torch.cuda.get_device_properties(device); print('CUDA_OK', device)" >/dev/null 2>&1; then
        cat >&2 <<EOF
CUDA preflight failed for $role on GPU $gpu_index.
The selected Python environment cannot initialize CUDA on this host.

Quick check:
    $PYTHON_BIN -c "import torch; print(torch.__version__, torch.version.cuda)"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
EOF
        return 1
    fi
    log "CUDA preflight passed for $role on GPU $gpu_index"
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

    log "Launching $label on GPU $gpu_index with 5 validation-fold workers"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu_index "$PYTHON_BIN" training/regression_finetune_ensemble_full_v2.py \
        --models "$model_name" "${COMMON_ARGS[@]}" --gpu_index 0 \
        >"$log_path" 2>&1 &
    LAST_BG_PID=$!
    log "$label pid=$LAST_BG_PID log=$log_path"
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-tmp/fullft_base_rescue_${RUN_STAMP}/runs_full_regression_v2}
TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT:-tmp/fullft_base_rescue_${RUN_STAMP}/logs/regression_full_v2}
LAUNCH_LOG_DIR=${LAUNCH_LOG_DIR:-tmp/fullft_base_rescue_${RUN_STAMP}/launcher_logs}
MLM_GPU_INDEX=${MLM_GPU_INDEX:-1}
MTR_GPU_INDEX=${MTR_GPU_INDEX:-2}
HYBRID_GPU_INDEX=${HYBRID_GPU_INDEX:-3}

mkdir -p "$LAUNCH_LOG_DIR"

MLM_STDOUT_LOG="$LAUNCH_LOG_DIR/mlm_base_${RUN_STAMP}.log"
MTR_STDOUT_LOG="$LAUNCH_LOG_DIR/mtr_base_${RUN_STAMP}.log"
HYBRID_STDOUT_LOG="$LAUNCH_LOG_DIR/hybrid_base_${RUN_STAMP}.log"

log "Starting medium full-finetune rescue run"
log "ROOT_DIR=$ROOT_DIR"
log "PYTHON_BIN=$PYTHON_BIN"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "TRAIN_LOG_ROOT=$TRAIN_LOG_ROOT"
log "LAUNCH_LOG_DIR=$LAUNCH_LOG_DIR"
log "GPU assignment: MLM=$MLM_GPU_INDEX MTR=$MTR_GPU_INDEX HYBRID=$HYBRID_GPU_INDEX"
log "Backbone launch mode=parallel total concurrent training jobs=15 (5 per GPU)"

log "Preparing benchmark data"
"$PYTHON_BIN" training/prepare_benchmark_data.py --task cycpeptmpdb_perm --seed 0
log "Prepared benchmark data"

check_cuda_runtime "$MLM_GPU_INDEX" "MLM base"
check_cuda_runtime "$MTR_GPU_INDEX" "MTR base"
check_cuda_runtime "$HYBRID_GPU_INDEX" "Hybrid base"

COMMON_ARGS=(
    --seed 101
    --output_root "$OUTPUT_ROOT"
    --log_root "$TRAIN_LOG_ROOT"
    --max_steps 2500
    --patience 3
    --learning_rate 1e-5
    --head_dropout 0.10
    --weight_decay 1e-4
    --target_scaling none
    --parallel_val_folds 5
    --force
)

launch_model_job "MLM base" "$MLM_GPU_INDEX" "aaronfeller/peptideclm-2-mlm-base" "$MLM_STDOUT_LOG"
mlm_pid=$LAST_BG_PID
launch_model_job "MTR base" "$MTR_GPU_INDEX" "aaronfeller/peptideclm-2-mtr-base" "$MTR_STDOUT_LOG"
mtr_pid=$LAST_BG_PID
launch_model_job "Hybrid base" "$HYBRID_GPU_INDEX" "aaronfeller/peptideclm-2-hybrid-base" "$HYBRID_STDOUT_LOG"
hybrid_pid=$LAST_BG_PID

trap 'cleanup_background_jobs "$mlm_pid" "$mtr_pid" "$hybrid_pid"' EXIT

log "All backbone processes launched; waiting for completion"
wait_and_report "$mlm_pid" "MLM base"
wait_and_report "$mtr_pid" "MTR base"
wait_and_report "$hybrid_pid" "Hybrid base"
trap - EXIT
log "Medium full-finetune rescue run finished"