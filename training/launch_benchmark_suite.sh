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
    "peptideclm|aaronfeller/peptideclm-2-mlm-small"
    "peptideclm|aaronfeller/peptideclm-2-mlm-base"
    "peptideclm|aaronfeller/peptideclm-2-mlm-large"
    "peptideclm|aaronfeller/peptideclm-2-mtr-small"
    "peptideclm|aaronfeller/peptideclm-2-mtr-base"
    "peptideclm|aaronfeller/peptideclm-2-mtr-large"
    "peptideclm|aaronfeller/peptideclm-2-hybrid-small"
    "peptideclm|aaronfeller/peptideclm-2-hybrid-base"
    "peptideclm|aaronfeller/peptideclm-2-hybrid-large"
    "chemberta77m|DeepChem/ChemBERTa-77M-MTR"
    "chemeleon|CheMeleon"
    "xgboost_rdkit|xgboost-rdkit"
    "xgboost_morgan|xgboost-morgan"
)

PEPMSND_MODELS=(
    "pepmsnd_kan|aaronfeller/peptideclm-2-hybrid-small"
    "pepmsnd_kan|aaronfeller/peptideclm-2-hybrid-base"
    "pepmsnd_kan|aaronfeller/peptideclm-2-hybrid-large"
    "pepmsnd_species_env|aaronfeller/peptideclm-2-hybrid-small"
    "pepmsnd_species_env|aaronfeller/peptideclm-2-hybrid-base"
    "pepmsnd_species_env|aaronfeller/peptideclm-2-hybrid-large"
)

CLASSIFICATION_TASKS=(amp_hgt cellppd thpep)
REGRESSION_TASKS=(cycpeptmpdb_perm)
PEPMSND_TASKS=(pepmsnd)

run_cmd() {
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

run_cmd "$PYTHON_BIN training/prepare_benchmark_data.py --task cycpeptmpdb_perm --seed 0 $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run)"
run_cmd "$PYTHON_BIN training/prepare_benchmark_data.py --task pepmsnd --seed 0 $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run)"
for seed in "${SEEDS[@]}"; do
    run_cmd "$PYTHON_BIN training/prepare_benchmark_data.py --task thpep --seed $seed $([[ "$DRY_RUN" -eq 1 ]] && echo --dry-run)"
done

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    exit 0
fi

gpu_index=0
for seed in "${SEEDS[@]}"; do
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r family model_name <<< "$entry"
        for task in "${CLASSIFICATION_TASKS[@]}" "${REGRESSION_TASKS[@]}"; do
            should_run_task "$task" || continue
            gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
            run_cmd "$PYTHON_BIN training/run_experiment.py --task $task --model_family $family --model_name '$model_name' --seed $seed --gpu_index $gpu $([[ "$DRY_RUN" -eq 1 ]] && echo --dry_run || echo --execute)"
            gpu_index=$((gpu_index + 1))
        done
    done

    for entry in "${PEPMSND_MODELS[@]}"; do
        IFS='|' read -r family model_name <<< "$entry"
        for fold in $(seq 1 10); do
            should_run_task pepmsnd || continue
            gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
            run_cmd "$PYTHON_BIN training/run_experiment.py --task pepmsnd --model_family $family --model_name '$model_name' --seed $seed --fold $fold --gpu_index $gpu $([[ "$DRY_RUN" -eq 1 ]] && echo --dry_run || echo --execute)"
            gpu_index=$((gpu_index + 1))
        done
    done
done