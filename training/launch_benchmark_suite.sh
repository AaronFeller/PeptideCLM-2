#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_CSV=${GPU_CSV:-0,1}
SEED_CSV=${SEED_CSV:-0,1,2}
DRY_RUN=1
PREPARE_ONLY=0
TASK_FILTER=${TASK_FILTER:-all}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_CSV="$2"
            shift 2
            ;;
        --seeds)
            SEED_CSV="$2"
            shift 2
            ;;
        --task-filter)
            TASK_FILTER="$2"
            shift 2
            ;;
        --execute)
            DRY_RUN=0
            shift
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

MODELS=(
    # "peptideclm|aaronfeller/peptideclm-2-mlm-small"
    # "peptideclm|aaronfeller/peptideclm-2-mtr-small"
    # "peptideclm|aaronfeller/peptideclm-2-hybrid-small"
    # "peptideclm|aaronfeller/peptideclm-2-mlm-base"
    # "peptideclm|aaronfeller/peptideclm-2-mtr-base"
    # "peptideclm|aaronfeller/peptideclm-2-hybrid-base"
    "peptideclm|aaronfeller/peptideclm-2-mtr-large"
    # "peptideclm|aaronfeller/peptideclm-2-mlm-large"
    # "peptideclm|aaronfeller/peptideclm-2-hybrid-large"
    # "chemberta77m|DeepChem/ChemBERTa-77M-MTR"
    # "chemeleon|CheMeleon"
    # "xgboost_rdkit|xgboost-rdkit"
    # "xgboost_morgan|xgboost-morgan"
)

PEPMSND_MODELS=(
    # "pepmsnd_kan|aaronfeller/peptideclm-2-mlm-large"
    # "pepmsnd_kan|aaronfeller/peptideclm-2-mtr-large"
    # "pepmsnd_kan|aaronfeller/peptideclm-2-hybrid-large"
    # "pepmsnd_species_env|aaronfeller/peptideclm-2-mlm-large"
    # "pepmsnd_species_env|aaronfeller/peptideclm-2-mtr-large"
    # "pepmsnd_species_env|aaronfeller/peptideclm-2-hybrid-large"
)

CLASSIFICATION_TASKS=(amp_hgt cellppd thpep)
# REGRESSION_TASKS=(cycpeptmpdb_perm)
# PEPMSND_TASKS=(pepmsnd)

# Global array to track parallel process IDs
pids=()

# Handles parallel execution of training jobs
run_cmd() {
    echo "$*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Localize and randomize port selection to prevent distributed network overlap
        (
            export MASTER_PORT=$(shuf -i 29000-37000 -n 1)
            cd "$ROOT_DIR" 
            eval "$*"
        ) &
        pids+=($!) # Track the background process ID
        
        # Limit concurrent background jobs to match the number of allocated GPUs
        while [[ ${#pids[@]} -ge ${#GPUS[@]} ]]; do
            still_running=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    still_running+=("$pid")
                fi
            done
            pids=("${still_running[@]}")
            
            if [[ ${#pids[@]} -ge ${#GPUS[@]} ]]; then
                sleep 1
            fi
        done
    fi
}

# Keeps data prep strictly sequential to prevent write collisions
run_prep() {
    echo "$*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        (cd "$ROOT_DIR" && eval "$*")
    fi
}

should_run_task() {
    local task="$1"
    if [[ "$TASK_FILTER" == "all" ]]; then
        return 0
    fi
    [[ ",$TASK_FILTER," == *",$task,"* ]]
}

# Execute data preparation sequentially
run_prep "$PYTHON_BIN training/prepare_benchmark_data.py --task cycpeptmpdb_perm --seed 0 $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run)"
run_prep "$PYTHON_BIN training/prepare_benchmark_data.py --task pepmsnd --seed 0 $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run)"
for seed in "${SEEDS[@]}"; do
    run_prep "$PYTHON_BIN training/prepare_benchmark_data.py --task thpep --seed $seed $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run)"
done

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    exit 0
fi
    
# Execute experimental training runs in parallel
gpu_index=0
for seed in "${SEEDS[@]}"; do
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r family model_name <<< "$entry"
        for task in "${CLASSIFICATION_TASKS[@]}" "${REGRESSION_TASKS[@]}"; do
            should_run_task "$task" || continue
            gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
            run_cmd "$PYTHON_BIN training/run_experiment.py --task $task --model_family $family --model_name '$model_name' --seed $seed --gpu_index $gpu $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run || echo --execute)"
            gpu_index=$((gpu_index + 1))
        done
    done

    for entry in "${PEPMSND_MODELS[@]}"; do
        IFS='|' read -r family model_name <<< "$entry"
        for fold in $(seq 1 10); do
            should_run_task pepmsnd || continue
            gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
            run_cmd "$PYTHON_BIN training/run_experiment.py --task pepmsnd --model_family $family --model_name '$model_name' --seed $seed --fold $fold --gpu_index $gpu $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run || echo --execute)"
            gpu_index=$((gpu_index + 1))
        done
    done
done

# Block the script from terminating until the final active processes complete
if [[ ${#pids[@]} -gt 0 ]]; then
    wait "${pids[@]}"
fi
echo "All experiments completed."