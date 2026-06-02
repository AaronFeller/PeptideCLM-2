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

check_cuda_runtime() {
    local gpu_index=$1
    local role=$2
    if ! nvidia-smi -i "$gpu_index" --query-gpu=index,name,uuid --format=csv,noheader >/dev/null 2>&1; then
        cat >&2 <<EOF
CUDA quick preflight failed for $role on GPU $gpu_index.
Quick check:
    nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
EOF
        return 1
    fi
    log "CUDA preflight passed for $role on GPU $gpu_index"
}

launch_model_job() {
    local label=$1
    local gpu_index=$2
    local model_name=$3
    local output_root=$4
    local log_root=$5
    local launch_log_path=$6
    local config_name=$7
    local max_steps=$8
    local patience=$9
    local learning_rate=${10}
    local head_dropout=${11}
    local weight_decay=${12}
    local warmup_fraction=${13}
    local parallel_val_folds=${14}
    shift 14
    local seed_values=("$@")

    local command=(
        "$PYTHON_BIN" training/regression_finetune_ensemble_full_v2.py
        --models "$model_name"
        --config_name "$config_name"
        --seeds "${seed_values[@]}"
        --output_root "$output_root"
        --log_root "$log_root"
        --max_steps "$max_steps"
        --patience "$patience"
        --learning_rate "$learning_rate"
        --head_dropout "$head_dropout"
        --weight_decay "$weight_decay"
        --warmup_fraction "$warmup_fraction"
        --target_scaling "$TARGET_SCALING"
        --parallel_val_folds "$parallel_val_folds"
        --parallel_startup_delay_seconds "$PARALLEL_STARTUP_DELAY_SECONDS"
        --gpu_index 0
        --force
    )

    if [[ ${#ACTIVE_TEST_FOLDS[@]} -gt 0 ]]; then
        command+=(--test_folds "${ACTIVE_TEST_FOLDS[@]}")
    fi

    log "Launching $label on GPU $gpu_index config=$config_name seeds=${seed_values[*]} holdouts=${ACTIVE_TEST_FOLDS[*]:-all} parallel_val_folds=$parallel_val_folds"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu_index "${command[@]}" >"$launch_log_path" 2>&1 &
    LAST_BG_PID=$!
    log "$label pid=$LAST_BG_PID log=$launch_log_path"
}

resolve_parallel_val_folds_for_scale() {
    local scale=$1
    case "$scale" in
        small)
            echo "$SMALL_PARALLEL_VAL_FOLDS"
            ;;
        base)
            echo "$BASE_PARALLEL_VAL_FOLDS"
            ;;
        large)
            echo "$LARGE_PARALLEL_VAL_FOLDS"
            ;;
        *)
            echo "$BASE_PARALLEL_VAL_FOLDS"
            ;;
    esac
}

run_scale_triplet() {
    local stage_name=$1
    local scale=$2
    local config_name=$3
    local output_root=$4
    local log_root=$5
    local launch_log_dir=$6
    local max_steps=$7
    local patience=$8
    local learning_rate=$9
    local head_dropout=${10}
    local weight_decay=${11}
    local warmup_fraction=${12}
    local parallel_val_folds=${13}
    shift 13
    local seed_values=("$@")

    mkdir -p "$launch_log_dir"

    launch_model_job "MLM $scale $stage_name" "$MLM_GPU_INDEX" "aaronfeller/peptideclm-2-mlm-$scale" "$output_root" "$log_root" "$launch_log_dir/mlm_${scale}_${config_name}.log" "$config_name" "$max_steps" "$patience" "$learning_rate" "$head_dropout" "$weight_decay" "$warmup_fraction" "$parallel_val_folds" "${seed_values[@]}"
    local mlm_pid=$LAST_BG_PID
    launch_model_job "MTR $scale $stage_name" "$MTR_GPU_INDEX" "aaronfeller/peptideclm-2-mtr-$scale" "$output_root" "$log_root" "$launch_log_dir/mtr_${scale}_${config_name}.log" "$config_name" "$max_steps" "$patience" "$learning_rate" "$head_dropout" "$weight_decay" "$warmup_fraction" "$parallel_val_folds" "${seed_values[@]}"
    local mtr_pid=$LAST_BG_PID
    launch_model_job "Hybrid $scale $stage_name" "$HYBRID_GPU_INDEX" "aaronfeller/peptideclm-2-hybrid-$scale" "$output_root" "$log_root" "$launch_log_dir/hybrid_${scale}_${config_name}.log" "$config_name" "$max_steps" "$patience" "$learning_rate" "$head_dropout" "$weight_decay" "$warmup_fraction" "$parallel_val_folds" "${seed_values[@]}"
    local hybrid_pid=$LAST_BG_PID

    trap 'cleanup_background_jobs "$mlm_pid" "$mtr_pid" "$hybrid_pid"' EXIT
    wait_and_report "$mlm_pid" "MLM $scale $stage_name"
    wait_and_report "$mtr_pid" "MTR $scale $stage_name"
    wait_and_report "$hybrid_pid" "Hybrid $scale $stage_name"
    trap - EXIT
}

run_scale_proxy_sweep() {
    local scale=$1
    shift
    local configs=("$@")
    local spec
    for spec in "${configs[@]}"; do
        IFS='|' read -r config_name max_steps patience learning_rate head_dropout weight_decay warmup_fraction <<< "$spec"
        local output_root="$SWEEP_ROOT/proxy/$scale/$config_name/runs_full_regression_v2"
        local log_root="$SWEEP_ROOT/proxy/$scale/$config_name/logs/regression_full_v2"
        local launch_log_dir="$SWEEP_ROOT/proxy/$scale/$config_name/launcher_logs"
        local parallel_val_folds
        parallel_val_folds=$(resolve_parallel_val_folds_for_scale "$scale")
        ACTIVE_TEST_FOLDS=("${PROXY_TEST_FOLDS[@]}")
        log "[proxy-start] scale=$scale config=$config_name output_root=$output_root"
        run_scale_triplet "proxy" "$scale" "$config_name" "$output_root" "$log_root" "$launch_log_dir" "$max_steps" "$patience" "$learning_rate" "$head_dropout" "$weight_decay" "$warmup_fraction" "$parallel_val_folds" "${TUNING_SEEDS[@]}"
        log "[proxy-done] scale=$scale config=$config_name"
    done
}

run_final_reruns() {
    local best_config_tsv=$1
    while IFS=$'\t' read -r model_scale config_name learning_rate weight_decay head_dropout warmup_fraction max_steps patience batch_size eval_batch_size accumulate_grad_batches parallel_val_folds test_folds rmse r2 mae mse spearman model_count seed_count; do
        [[ "$model_scale" == "model_scale" ]] && continue
        [[ -z "$model_scale" ]] && continue
        local output_root="$SWEEP_ROOT/final/$model_scale/$config_name/runs_full_regression_v2"
        local log_root="$SWEEP_ROOT/final/$model_scale/$config_name/logs/regression_full_v2"
        local launch_log_dir="$SWEEP_ROOT/final/$model_scale/$config_name/launcher_logs"
        local parallel_val_folds
        parallel_val_folds=$(resolve_parallel_val_folds_for_scale "$model_scale")
        ACTIVE_TEST_FOLDS=()
        log "[final-start] scale=$model_scale config=$config_name output_root=$output_root"
        run_scale_triplet "final" "$model_scale" "$config_name" "$output_root" "$log_root" "$launch_log_dir" "$max_steps" "$patience" "$learning_rate" "$head_dropout" "$weight_decay" "$warmup_fraction" "$parallel_val_folds" "${FINAL_SEEDS[@]}"
        log "[final-done] scale=$model_scale config=$config_name"
    done < "$best_config_tsv"
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
SWEEP_ROOT=${SWEEP_ROOT:-tmp/fullft_grid_search_${RUN_STAMP}}
REPORT_DIR=${REPORT_DIR:-$SWEEP_ROOT/report}
SELECTION_METRIC=${SELECTION_METRIC:-rmse}
TARGET_SCALING=${TARGET_SCALING:-none}
SMALL_PARALLEL_VAL_FOLDS=${SMALL_PARALLEL_VAL_FOLDS:-5}
BASE_PARALLEL_VAL_FOLDS=${BASE_PARALLEL_VAL_FOLDS:-5}
LARGE_PARALLEL_VAL_FOLDS=${LARGE_PARALLEL_VAL_FOLDS:-2}
PARALLEL_STARTUP_DELAY_SECONDS=${PARALLEL_STARTUP_DELAY_SECONDS:-20}
MLM_GPU_INDEX=${MLM_GPU_INDEX:-1}
MTR_GPU_INDEX=${MTR_GPU_INDEX:-2}
HYBRID_GPU_INDEX=${HYBRID_GPU_INDEX:-3}
RUN_PROXY_SWEEP=${RUN_PROXY_SWEEP:-1}
RUN_FINAL_RERUN=${RUN_FINAL_RERUN:-1}

read -r -a PROXY_TEST_FOLDS <<< "${PROXY_TEST_FOLDS:-0 3}"
read -r -a TUNING_SEEDS <<< "${TUNING_SEEDS:-101}"
read -r -a FINAL_SEEDS <<< "${FINAL_SEEDS:-101}"
ACTIVE_TEST_FOLDS=()

small_configs=(
    "small_default|8000|8|1e-5|0.15|1e-3|0.10"
    "small_shorter|6000|6|1e-5|0.15|1e-3|0.10"
    "small_longer|12000|12|1e-5|0.15|1e-3|0.10"
    "small_lower_lr|10000|10|7.5e-6|0.15|1e-3|0.10"
    "small_higher_lr|8000|8|1.25e-5|0.15|1e-3|0.10"
    "small_low_dropout|10000|10|1e-5|0.10|1e-3|0.10"
    "small_high_dropout|10000|10|1e-5|0.25|1e-3|0.10"
    "small_low_wd|10000|10|1e-5|0.15|5e-4|0.10"
    "small_more_reg|10000|10|1e-5|0.20|2e-3|0.10"
    "small_heavy_reg|10000|10|7.5e-6|0.25|3e-3|0.10"
    "small_short_warmup|10000|10|1e-5|0.15|1e-3|0.05"
    "small_long_warmup|10000|10|1e-5|0.15|1e-3|0.15"
    "small_low_lr_longwarm|12000|12|7.5e-6|0.15|1e-3|0.20"
)

base_configs=(
    "base_rescue|10000|8|5e-6|0.10|1e-3|0.10"
    "base_shorter|8000|6|5e-6|0.10|1e-3|0.10"
    "base_longer|14000|12|5e-6|0.10|1e-3|0.10"
    "base_lower_lr|12000|10|3e-6|0.10|1e-3|0.10"
    "base_higher_lr|10000|8|7.5e-6|0.10|1e-3|0.10"
    "base_low_dropout|10000|8|5e-6|0.05|1e-3|0.10"
    "base_higher_dropout|10000|10|5e-6|0.15|1e-3|0.10"
    "base_low_wd|10000|8|5e-6|0.10|5e-4|0.10"
    "base_more_reg|10000|10|5e-6|0.15|2e-3|0.10"
    "base_high_reg|12000|12|3e-6|0.15|3e-3|0.10"
    "base_short_warmup|10000|8|5e-6|0.10|1e-3|0.05"
    "base_long_warmup|12000|10|5e-6|0.10|1e-3|0.15"
    "base_low_lr_longwarm|14000|12|3e-6|0.10|1e-3|0.20"
)

large_configs=(
    "large_default|6000|6|5e-6|0.15|1e-3|0.10"
    "large_shorter|5000|5|5e-6|0.15|1e-3|0.10"
    "large_longer|10000|10|5e-6|0.15|1e-3|0.10"
    "large_lower_lr|8000|8|3e-6|0.15|1e-3|0.10"
    "large_higher_lr|6000|6|7.5e-6|0.15|1e-3|0.10"
    "large_low_dropout|8000|8|5e-6|0.10|1e-3|0.10"
    "large_high_dropout|8000|8|5e-6|0.20|1e-3|0.10"
    "large_low_wd|8000|8|5e-6|0.15|5e-4|0.10"
    "large_more_reg|8000|8|5e-6|0.20|2e-3|0.10"
    "large_heavy_reg|10000|10|3e-6|0.20|3e-3|0.10"
    "large_short_warmup|8000|8|5e-6|0.15|1e-3|0.05"
    "large_long_warmup|8000|8|5e-6|0.15|1e-3|0.15"
    "large_low_lr_longwarm|10000|10|3e-6|0.15|1e-3|0.20"
)

mkdir -p "$REPORT_DIR"

log "Starting full-ft grid search"
log "ROOT_DIR=$ROOT_DIR"
log "PYTHON_BIN=$PYTHON_BIN"
log "SWEEP_ROOT=$SWEEP_ROOT"
log "REPORT_DIR=$REPORT_DIR"
log "SELECTION_METRIC=$SELECTION_METRIC"
log "PROXY_TEST_FOLDS=${PROXY_TEST_FOLDS[*]}"
log "TUNING_SEEDS=${TUNING_SEEDS[*]} FINAL_SEEDS=${FINAL_SEEDS[*]}"
log "Parallel val folds: small=$SMALL_PARALLEL_VAL_FOLDS base=$BASE_PARALLEL_VAL_FOLDS large=$LARGE_PARALLEL_VAL_FOLDS"
log "Grid sizes: small=${#small_configs[@]} base=${#base_configs[@]} large=${#large_configs[@]}"
log "GPU assignment: MLM=$MLM_GPU_INDEX MTR=$MTR_GPU_INDEX HYBRID=$HYBRID_GPU_INDEX"

check_cuda_runtime "$MLM_GPU_INDEX" "MLM sweep worker"
check_cuda_runtime "$MTR_GPU_INDEX" "MTR sweep worker"
check_cuda_runtime "$HYBRID_GPU_INDEX" "Hybrid sweep worker"

if [[ "$RUN_PROXY_SWEEP" == "1" ]]; then
    run_scale_proxy_sweep small "${small_configs[@]}"
    run_scale_proxy_sweep base "${base_configs[@]}"
    run_scale_proxy_sweep large "${large_configs[@]}"
fi

"$PYTHON_BIN" training/report_fullft_grid_search.py \
    --sweep_root "$SWEEP_ROOT" \
    --selection_metric "$SELECTION_METRIC" \
    --report_dir "$REPORT_DIR"

BEST_CONFIG_TSV="$REPORT_DIR/best_configs.tsv"
if [[ "$RUN_FINAL_RERUN" == "1" ]]; then
    if [[ ! -s "$BEST_CONFIG_TSV" ]]; then
        echo "Expected best-config report at $BEST_CONFIG_TSV" >&2
        exit 1
    fi
    run_final_reruns "$BEST_CONFIG_TSV"
    "$PYTHON_BIN" training/report_fullft_grid_search.py \
        --sweep_root "$SWEEP_ROOT" \
        --selection_metric "$SELECTION_METRIC" \
        --report_dir "$REPORT_DIR"
fi

log "Full-ft grid search complete. Report bundle: $REPORT_DIR"