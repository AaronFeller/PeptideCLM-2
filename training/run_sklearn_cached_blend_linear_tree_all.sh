#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-.venv311/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-tmp/runs_sklearn_cached_blend_linear_tree_${RUN_STAMP}}
CACHE_ROOT=${CACHE_ROOT:-tmp/embeddings_regression}
N_JOBS=${N_JOBS:-32}

"$PYTHON_BIN" training/prepare_benchmark_data.py --task cycpeptmpdb_perm --seed 0

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" training/run_sklearn_cached_regression_search.py \
    --all_models \
    --candidate_names blend_linear_tree \
    --seeds 101 202 303 \
    --output_root "$OUTPUT_ROOT" \
    --cache_root "$CACHE_ROOT" \
    --rebalance_train_bins \
    --rebalance_bin_count 5 \
    --n_jobs "$N_JOBS"