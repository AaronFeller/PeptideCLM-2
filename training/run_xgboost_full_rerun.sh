#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-tmp/xgboost_full_rerun_${RUN_STAMP}}
CACHE_ROOT=${CACHE_ROOT:-tmp/embeddings_regression}
N_JOBS=${N_JOBS:-16}

"$PYTHON_BIN" training/prepare_benchmark_data.py --task cycpeptmpdb_perm --seed 0

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" training/run_xgboost_transformer_regression_pipeline.py \
    --all_models \
    --seeds 101 202 303 \
    --output_root "$OUTPUT_ROOT" \
    --cache_root "$CACHE_ROOT" \
    --n_jobs "$N_JOBS"