#!/usr/bin/env bash
set -e

###############################################
# USER CONFIGURATION
###############################################

DATASETS=("AmpHGT" "MHC" "THPep" "CellPPD")

MODELS=(
    "aaronfeller/PeptideMTR_sm"
    "aaronfeller/PeptideMTR_base"
    "aaronfeller/PeptideMTR_lg"
    "aaronfeller/PeptideMLM_sm"
    "aaronfeller/PeptideMLM_base"
    "aaronfeller/PeptideMLM_lg"
    "aaronfeller/PeptideMLM-MTR_sm"
    "aaronfeller/PeptideMLM-MTR_base"
    "aaronfeller/PeptideMLM-MTR_lg"
)

GPUS=(4 5 6)

TRAIN_SCRIPT="scripts/train_model.py"

LOG_DIR="logs/launcher"
mkdir -p "$LOG_DIR"

###############################################
# GPU MEMORY CHECK
###############################################

# Free memory (MB) for GPU index
get_gpu_free_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | sed -n "$((1 + $1))p"
}

# Return list of GPUs with >80GB free
find_free_gpus() {
    local threshold=80000
    local free
    local candidates=()

    for gpu in "${GPUS[@]}"; do
        free=$(get_gpu_free_mem $gpu)
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free > threshold )); then
            candidates+=("$gpu")
        fi
    done

    echo "${candidates[@]}"
}

###############################################
# BUILD JOB QUEUE (WITH SKIPS)
###############################################

declare -a JOB_DATASETS
declare -a JOB_MODELS

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        # Skip already-completed AmpHGT runs
        if [[ "$DATASET" == "AmpHGT" && ("$MODEL" == *"_sm"* || "$MODEL" == *"_base"*) ]]; then
            echo "[SKIP] Already completed: $DATASET $MODEL"
            continue
        fi

        if [[ "$DATASET" == "CellPPD" && "$MODEL" == *"MLM-MTR_sm"* ]]; then
            echo "[SKIP] Already completed: $DATASET $MODEL"
            continue
        fi

        JOB_DATASETS+=("$DATASET")
        JOB_MODELS+=("$MODEL")
    done
done

TOTAL_JOBS=${#JOB_DATASETS[@]}
echo "[INFO] Total jobs to run: $TOTAL_JOBS"

###############################################
# MAIN LOOP — FILL ALL FREE GPUS, WAIT 60s, REPEAT
###############################################

JOB_ID=0

echo "[INFO] Entering scheduling loop..."

while (( JOB_ID < TOTAL_JOBS )); do

    # Get all currently free GPUs
    FREE_GPUS=( $(find_free_gpus) )

    if (( ${#FREE_GPUS[@]} == 0 )); then
        echo "[INFO] No GPUs with >80GB free — sleeping 120s..."
        sleep 120
        continue
    fi

    echo "[INFO] Free GPUs: ${FREE_GPUS[*]}"

    # Launch one job per free GPU (or until we run out of jobs)
    for gpu in "${FREE_GPUS[@]}"; do
        if (( JOB_ID >= TOTAL_JOBS )); then
            break
        fi

        DATASET=${JOB_DATASETS[$JOB_ID]}
        MODEL=${JOB_MODELS[$JOB_ID]}

        LOG_FILE="${LOG_DIR}/job_${JOB_ID}_${DATASET}/${MODEL//\//_}.log"
        mkdir -p "$(dirname "$LOG_FILE")"

        echo "[LAUNCH] Job $JOB_ID: $DATASET $MODEL on GPU $gpu"
        echo "  Log: $LOG_FILE"

        CUDA_VISIBLE_DEVICES=$gpu \
        python "$TRAIN_SCRIPT" \
            --dataset "$DATASET" \
            --gpu 0 \
            --model_name "$MODEL" \
            --batch_size 32 \
            --save_path "results_bs32" \
            > "$LOG_FILE" 2>&1 &

        JOB_ID=$((JOB_ID + 1))
        sleep 2
    done

    echo "[INFO] Sleeping 120s before next GPU scan..."
    sleep 120
done

echo "========================================="
echo "All jobs launched."
echo "========================================="