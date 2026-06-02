#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODELS=(
  "aaronfeller/peptideclm-2-mlm-large"
  "aaronfeller/peptideclm-2-mtr-large"
  "aaronfeller/peptideclm-2-hybrid-large"
)

GPUS=(1 2 3)
EXP_ROOT="results_pepmsnd_just_BERT3-redo"
LOG_ROOT="main/logs/launcher"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

get_gpu_free_mem() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((1 + $1))p"
}

find_free_gpus() {
  local threshold=48000
  local free
  local candidates=()

  for gpu in "${GPUS[@]}"; do
    free=$(get_gpu_free_mem "$gpu")
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free > threshold )); then
      candidates+=("$gpu")
    fi
  done

  echo "${candidates[@]}"
}

# Build job list (3 models x 10 folds)
declare -a JOB_MODELS
declare -a JOB_FOLDS

for MODEL in "${MODELS[@]}"; do
  for FOLD in {1..10}; do
    JOB_MODELS+=("$MODEL")
    JOB_FOLDS+=("$FOLD")
  done
done

TOTAL_JOBS=${#JOB_MODELS[@]}
JOB_ID=0

echo "[INFO] Total jobs to run: $TOTAL_JOBS"
echo "[INFO] Entering scheduling loop..."

while (( JOB_ID < TOTAL_JOBS )); do
  FREE_GPUS=( $(find_free_gpus) )

  if (( ${#FREE_GPUS[@]} == 0 )); then
    echo "[INFO] No GPUs free. Sleeping 120s..."
    sleep 120
    continue
  fi

  echo "[INFO] Free GPUs: ${FREE_GPUS[*]}"

  for gpu in "${FREE_GPUS[@]}"; do
    if (( JOB_ID >= TOTAL_JOBS )); then
      break
    fi

    MODEL=${JOB_MODELS[$JOB_ID]}
    FOLD=${JOB_FOLDS[$JOB_ID]}

    SAVE_PATH="${EXP_ROOT}/${MODEL//\//_}/fold${FOLD}"
    LOG_FILE="${LOG_ROOT}/pepmsnd_species_env_${MODEL//\//_}_fold${FOLD}.log"

    mkdir -p "$SAVE_PATH" "$(dirname "$LOG_FILE")"

    echo "[LAUNCH] Job $JOB_ID: fold $FOLD | model $MODEL | GPU $gpu"
    echo "  Log: $LOG_FILE"
    echo "  Out: $SAVE_PATH"

    CUDA_VISIBLE_DEVICES=$gpu \
      .venv/bin/python main/scripts/train_pepmsnd_species_env.py \
        --model_name "$MODEL" \
        --fold "$FOLD" \
        --save_path "$SAVE_PATH" \
        --seed 103 \
        > "$LOG_FILE" 2>&1 &

    JOB_ID=$((JOB_ID + 1))
    sleep 5
  done

  echo "[INFO] Sleeping 120s before next GPU scan..."
  sleep 120
done

wait

echo "========================================="
echo "All PepMSND Species/Environment jobs complete."
echo "========================================="
