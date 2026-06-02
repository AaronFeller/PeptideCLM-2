from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from training.adapters.common import load_task_frames
from training.experiment.manifest import REPO_ROOT


def make_candidate(
    name: str,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    reg_lambda: float,
    reg_alpha: float,
    early_stopping_rounds: int,
) -> tuple[str, dict[str, float | int]]:
    return (
        name,
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "early_stopping_rounds": early_stopping_rounds,
        },
    )


CANDIDATE_CONFIGS: list[tuple[str, dict[str, float | int]]] = [
    make_candidate("large_default", n_estimators=2500, max_depth=4, learning_rate=0.025, subsample=0.75, colsample_bytree=0.35, min_child_weight=6.0, reg_lambda=3.0, reg_alpha=0.25, early_stopping_rounds=80),
    make_candidate("base_like", n_estimators=2000, max_depth=4, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=4.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=60),
    make_candidate("depth5_anchor", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_more_cols", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.60, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_max_cols", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.70, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_more_rows", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_rows_cols", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.60, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_low_child", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=2.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_high_child", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=4.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_low_lambda", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=1.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_mid_lambda", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=1.5, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth5_l1", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.05, early_stopping_rounds=80),
    make_candidate("depth5_l1_stronger", n_estimators=2500, max_depth=5, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.10, early_stopping_rounds=80),
    make_candidate("depth5_slower", n_estimators=3200, max_depth=5, learning_rate=0.025, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=100),
    make_candidate("depth5_faster", n_estimators=2200, max_depth=5, learning_rate=0.035, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth6_anchor", n_estimators=2500, max_depth=6, learning_rate=0.03, subsample=0.80, colsample_bytree=0.50, min_child_weight=3.0, reg_lambda=2.0, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth6_guarded", n_estimators=2500, max_depth=6, learning_rate=0.028, subsample=0.80, colsample_bytree=0.45, min_child_weight=4.0, reg_lambda=2.0, reg_alpha=0.05, early_stopping_rounds=100),
    make_candidate("depth6_wide", n_estimators=2500, max_depth=6, learning_rate=0.03, subsample=0.85, colsample_bytree=0.60, min_child_weight=2.0, reg_lambda=1.5, reg_alpha=0.0, early_stopping_rounds=80),
    make_candidate("depth4_wide", n_estimators=2800, max_depth=4, learning_rate=0.03, subsample=0.85, colsample_bytree=0.70, min_child_weight=2.0, reg_lambda=1.0, reg_alpha=0.0, early_stopping_rounds=100),
    make_candidate("depth5_balanced", n_estimators=2800, max_depth=5, learning_rate=0.028, subsample=0.85, colsample_bytree=0.55, min_child_weight=3.0, reg_lambda=1.5, reg_alpha=0.0, early_stopping_rounds=100),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached large-model XGBoost sweep for cycpeptmpdb_perm.")
    parser.add_argument("--task", default="cycpeptmpdb_perm")
    parser.add_argument("--model", default="aaronfeller/peptideclm-2-mtr-large")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--cache_root", type=Path, default=REPO_ROOT / "tmp" / "embeddings_regression")
    parser.add_argument("--prepared_data_root", type=Path, default=REPO_ROOT / "tmp" / "prepared_data")
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "xgboost_large_cached_sweep")
    parser.add_argument("--n_jobs", type=int, default=8)
    return parser.parse_args()


def compute_regression_metrics_with_mse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        spearman = spearmanr(y_true, y_pred)
    spearman_value = getattr(spearman, "statistic", np.nan)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearman_value if np.isfinite(spearman_value) else np.nan),
    }


def fit_regression_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    n_jobs: int,
    xgboost_config: dict[str, float | int],
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_estimators=int(xgboost_config["n_estimators"]),
        max_depth=int(xgboost_config["max_depth"]),
        learning_rate=float(xgboost_config["learning_rate"]),
        subsample=float(xgboost_config["subsample"]),
        colsample_bytree=float(xgboost_config["colsample_bytree"]),
        min_child_weight=float(xgboost_config["min_child_weight"]),
        reg_lambda=float(xgboost_config["reg_lambda"]),
        reg_alpha=float(xgboost_config["reg_alpha"]),
        tree_method="hist",
        device="cpu",
        n_jobs=n_jobs,
        early_stopping_rounds=int(xgboost_config["early_stopping_rounds"]),
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def round_robin_ensemble_predictions(
    features: np.ndarray,
    data_frame: pd.DataFrame,
    seed: int,
    n_jobs: int,
    xgboost_config: dict[str, float | int],
) -> np.ndarray:
    fold_ids = sorted(data_frame["fold"].unique().tolist())
    predictions = np.zeros(len(data_frame), dtype=np.float32)

    for fold_id in fold_ids:
        test_mask = data_frame["fold"] == fold_id
        test_x = features[test_mask.to_numpy()]
        fold_predictions = []

        for val_fold in fold_ids:
            if val_fold == fold_id:
                continue

            train_mask = (~test_mask) & (data_frame["fold"] != val_fold)
            val_mask = data_frame["fold"] == val_fold

            train_x = features[train_mask.to_numpy()]
            val_x = features[val_mask.to_numpy()]
            train_y = np.asarray(data_frame.loc[train_mask, "value"], dtype=np.float32)
            val_y = np.asarray(data_frame.loc[val_mask, "value"], dtype=np.float32)

            model = fit_regression_model(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                seed=seed,
                n_jobs=n_jobs,
                xgboost_config=xgboost_config,
            )
            fold_predictions.append(model.predict(test_x).astype(np.float32))

        predictions[test_mask.to_numpy()] = np.mean(np.vstack(fold_predictions), axis=0)

    return predictions


def main() -> int:
    args = parse_args()
    if args.task != "cycpeptmpdb_perm":
        raise ValueError("This sweep currently supports only cycpeptmpdb_perm.")

    data_frame = load_task_frames(args.task, seed=0, prepared_data_root=args.prepared_data_root)["full"]
    data_frame = data_frame.copy().sort_values("fold").reset_index(drop=True)

    model_variant = args.model.split("/")[-1].replace("-", "_")
    cache_path = args.cache_root / model_variant / args.task / "full.npy"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached embeddings not found: {cache_path}")

    features = np.load(cache_path)
    true_targets = data_frame["value"].to_numpy(dtype=np.float32)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for candidate_name, xgboost_config in CANDIDATE_CONFIGS:
        print(f"[candidate-start] name={candidate_name} config={xgboost_config}", flush=True)
        for seed in args.seeds:
            predictions = round_robin_ensemble_predictions(
                features=features,
                data_frame=data_frame,
                seed=seed,
                n_jobs=args.n_jobs,
                xgboost_config=xgboost_config,
            )
            metrics = compute_regression_metrics_with_mse(true_targets, predictions)
            row = {
                "candidate": candidate_name,
                "model_name": args.model,
                "seed": seed,
                **xgboost_config,
                **metrics,
            }
            summary_rows.append(row)
            print(
                f"[candidate-done] name={candidate_name} seed={seed} r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f}",
                flush=True,
            )

    summary_frame = pd.DataFrame(summary_rows).sort_values(["r2", "spearman"], ascending=[False, False]).reset_index(drop=True)
    summary_path = args.output_root / "summary_metrics.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(f"[sweep-done] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())