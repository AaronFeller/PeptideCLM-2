#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$ROOT_DIR/.venv311/bin/python"}
SCRIPT_PATH=${SCRIPT_PATH:-"$ROOT_DIR/finetune_pampa_holdouts_hf.py"}
MLM_GPU_INDEX=${MLM_GPU_INDEX:-1}
MTR_GPU_INDEX=${MTR_GPU_INDEX:-2}
HYBRID_GPU_INDEX=${HYBRID_GPU_INDEX:-3}
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/tmp/logs/pampa_holdouts_hf"}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Training script not found: $SCRIPT_PATH" >&2
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] $*"
}

aggregate_summary() {
    local output_root=${1:-"$ROOT_DIR/tmp/runs_hf_pampa_holdouts"}
    "$PYTHON_BIN" - "$output_root" <<'PY'
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import r2_score


output_root = Path(sys.argv[1])
rows = []
for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
    metrics_path = model_dir / "holdout_metrics.csv"
    prediction_path = model_dir / "holdout_predictions.csv"
    if not metrics_path.exists():
        continue
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        continue
    summary_r2 = float("nan")
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        valid = predictions[["value", "prediction"]].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(valid) >= 2:
            summary_r2 = float(r2_score(valid["value"], valid["prediction"]))
    summary = {
        "model_name": str(metrics["model_name"].iloc[0]),
        "r2": summary_r2,
        "mean_mse": float(metrics["mse"].mean()),
        "mean_rmse": float(metrics["rmse"].mean()),
        "mean_mae": float(metrics["mae"].mean()),
        "output_dir": str(model_dir),
    }
    name = summary["model_name"].lower()
    if "-small" in name:
        summary["learning_rate"] = 3e-4
    elif "-large" in name:
        summary["learning_rate"] = 5e-5
    else:
        summary["learning_rate"] = 1e-4
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path, nrows=1)
        if not predictions.empty:
            summary["batch_size"] = 16
            summary["eval_batch_size"] = 128
    rows.append(summary)

summary_frame = pd.DataFrame(rows)
if not summary_frame.empty:
    summary_frame = summary_frame.sort_values("model_name").reset_index(drop=True)
summary_frame.to_csv(output_root / "summary_metrics.csv", index=False)
PY
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

launch_type_job() {
    local model_type=$1
    local gpu_index=$2
    local log_path=$3

    log "Launching model_type=$model_type on GPU $gpu_index"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu_index" "$PYTHON_BIN" "$SCRIPT_PATH" --model_type "$model_type" "$@" >"$log_path" 2>&1 &
}

EXTRA_ARGS=("$@")
MLM_LOG_PATH="$LOG_DIR/mlm_${RUN_STAMP}.log"
MTR_LOG_PATH="$LOG_DIR/mtr_${RUN_STAMP}.log"
HYBRID_LOG_PATH="$LOG_DIR/hybrid_${RUN_STAMP}.log"

log "Launching HF PAMPA holdout finetuning"
echo "Python: $PYTHON_BIN"
echo "Script: $SCRIPT_PATH"
echo "Logs: $LOG_DIR"
echo "GPU assignment: mlm=$MLM_GPU_INDEX mtr=$MTR_GPU_INDEX hybrid=$HYBRID_GPU_INDEX"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$MLM_GPU_INDEX" "$PYTHON_BIN" "$SCRIPT_PATH" --model_type mlm "${EXTRA_ARGS[@]}" >"$MLM_LOG_PATH" 2>&1 &
mlm_pid=$!
log "mlm pid=$mlm_pid log=$MLM_LOG_PATH"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$MTR_GPU_INDEX" "$PYTHON_BIN" "$SCRIPT_PATH" --model_type mtr "${EXTRA_ARGS[@]}" >"$MTR_LOG_PATH" 2>&1 &
mtr_pid=$!
log "mtr pid=$mtr_pid log=$MTR_LOG_PATH"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$HYBRID_GPU_INDEX" "$PYTHON_BIN" "$SCRIPT_PATH" --model_type hybrid "${EXTRA_ARGS[@]}" >"$HYBRID_LOG_PATH" 2>&1 &
hybrid_pid=$!
log "hybrid pid=$hybrid_pid log=$HYBRID_LOG_PATH"

trap 'cleanup_background_jobs "$mlm_pid" "$mtr_pid" "$hybrid_pid"' EXIT INT TERM

wait_and_report "$mlm_pid" "mlm"
wait_and_report "$mtr_pid" "mtr"
wait_and_report "$hybrid_pid" "hybrid"
trap - EXIT INT TERM
aggregate_summary
log "Rebuilt summary_metrics.csv from per-model outputs"
log "All model-type jobs finished"