#!/usr/bin/env bash
set -euo pipefail # Added -u for unbound variable safety

###############################################
# USER CONFIGURATION
###############################################

DATASETS=("AmpHGT" "THPep" "CellPPD") # Fixed casing mismatch for 'AmpHGT'

MODELS=(
    "aaronfeller/peptideclm-2-mlm-large"
    "aaronfeller/peptideclm-2-mtr-large"
    "aaronfeller/peptideclm-2-hybrid-large"
    "chemberta77m|DeepChem/ChemBERTa-77M-MTR" # Added baseline mapping compat if needed
)

GPUS=(0 1 2 3 4 5 6 7) 
TRAIN_SCRIPT="classification_finetuning.py"
LOG_DIR="logs/launcher"
mkdir -p "$LOG_DIR"

###############################################
# BUILD JOB QUEUE
###############################################

declare -a JOB_DATASETS
declare -a JOB_MODELS

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        JOB_DATASETS+=("$DATASET")
        JOB_MODELS+=("$MODEL")
    done
done

TOTAL_JOBS=${#JOB_DATASETS[@]}
echo "[INFO] Total jobs queued: $TOTAL_JOBS"

###############################################
# ACTIVE SLOT SCHEDULER
###############################################

# Associative array tracking which PID is running on which GPU slot
declare -A SLOT_PIDS

# Initialize slots as empty (PID 0 means free)
for gpu in "${GPUS[@]}"; do
    SLOT_PIDS[$gpu]=0
done

JOB_ID=0

while (( JOB_ID < TOTAL_JOBS )); do
    for gpu in "${GPUS[@]}"; do
        # Double check we haven't run out of jobs mid-loop
        if (( JOB_ID >= TOTAL_JOBS )); then
            break
        fi

        CURRENT_PID=${SLOT_PIDS[$gpu]}

        # If a slot has a PID, verify if it is still running
        if [[ "$CURRENT_PID" -ne 0 ]]; then
            if ! kill -0 "$CURRENT_PID" 2>/dev/null; then
                echo "[INFO] GPU $gpu finished active process $CURRENT_PID"
                SLOT_PIDS[$gpu]=0
                CURRENT_PID=0
            fi
        fi

        # If slot is free, deploy the next experiment immediately
        if [[ "$CURRENT_PID" -eq 0 ]]; then
            DATASET=${JOB_DATASETS[$JOB_ID]}
            MODEL=${JOB_MODELS[$JOB_ID]}

            # Strip model string prefixes for clean log filenames
            CLEAN_MODEL_NAME=${MODEL//\//_}
            LOG_FILE="${LOG_DIR}/job_${JOB_ID}_${DATASET}_${CLEAN_MODEL_NAME}.log"
            mkdir -p "$(dirname "$LOG_FILE")"

            echo "[LAUNCH] Job $JOB_ID/$TOTAL_JOBS: $DATASET | $MODEL onto GPU $gpu"

            # Execute via masked visibility topology
            CUDA_VISIBLE_DEVICES=$gpu \
            python "$TRAIN_SCRIPT" \
                --dataset "$DATASET" \
                --gpu 0 \
                --gpu_index 0 \
                --model_name "$MODEL" \
                --batch_size 32 \
                --save_path "results_bs32" \
                > "$LOG_FILE" 2>&1 &

            # Lock the GPU slot to this background process ID ($!)
            SLOT_PIDS[$gpu]=$!
            JOB_ID=$((JOB_ID + 1))
        fi
    done

    # Quick polling interval to check for completed processes
    sleep 1
done

###############################################
# SAFETY WAIT GUARD
###############################################
echo "[INFO] All jobs dispatched. Waiting for final active tasks to close..."

# Extract all running PIDs from our slot map and block until they complete
RUNNING_PIDS=()
for gpu in "${GPUS[@]}"; do
    PID=${SLOT_PIDS[$gpu]}
    if [[ "$PID" -ne 0 ]]; then
        RUNNING_PIDS+=("$PID")
    fi
done

if (( ${#RUNNING_PIDS[@]} > 0 )); then
    wait "${RUNNING_PIDS[@]}"
fi

echo "========================================="
echo "All parallel experiment queues completed."
echo "========================================="