from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    load_task_frames,
    morgan_feature_matrix,
    rdkit_feature_matrix,
    write_baseline_outputs,
)
from training.experiment.manifest import REPO_ROOT


DEFAULT_MODELS = [
    "aaronfeller/peptideclm-2-mlm-small",
    "aaronfeller/peptideclm-2-mtr-small",
    "aaronfeller/peptideclm-2-hybrid-small",
    "aaronfeller/peptideclm-2-mlm-base",
    "aaronfeller/peptideclm-2-mtr-base",
    "aaronfeller/peptideclm-2-hybrid-base",
    "aaronfeller/peptideclm-2-mlm-large",
    "aaronfeller/peptideclm-2-mtr-large",
    "aaronfeller/peptideclm-2-hybrid-large",
]
DEFAULT_SEEDS = [101, 202, 303]
DESCRIPTOR_FEATURE_SETS = ["rdkit", "morgan"]


def make_candidate(
    name: str,
    estimator_type: str,
    *,
    params: dict[str, object],
    scale_features: bool = False,
    blend_with: tuple[str, str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "estimator_type": estimator_type,
        "params": params,
        "scale_features": scale_features,
        "blend_with": blend_with,
    }


CANDIDATE_CONFIGS: list[dict[str, object]] = [
    make_candidate("ridge_anchor", "ridge", params={"alpha": 4.0}, scale_features=True),
    make_candidate("elasticnet_balanced", "elasticnet", params={"alpha": 0.0015, "l1_ratio": 0.25, "max_iter": 5000}, scale_features=True),
    make_candidate("elasticnet_sparse", "elasticnet", params={"alpha": 0.0040, "l1_ratio": 0.60, "max_iter": 5000}, scale_features=True),
    make_candidate("huber_anchor", "huber", params={"epsilon": 1.35, "alpha": 0.0005, "max_iter": 1000}, scale_features=True),
    make_candidate("extra_trees_balanced", "extra_trees", params={"n_estimators": 700, "max_features": "sqrt", "min_samples_leaf": 2, "min_samples_split": 4, "bootstrap": False}),
    make_candidate("extra_trees_wide", "extra_trees", params={"n_estimators": 900, "max_features": 0.35, "min_samples_leaf": 1, "min_samples_split": 2, "bootstrap": False}),
    make_candidate("extra_trees_shrunk", "extra_trees", params={"n_estimators": 700, "max_features": 0.20, "min_samples_leaf": 4, "min_samples_split": 6, "bootstrap": False}),
    make_candidate("hist_gbr_balanced", "hist_gbr", params={"learning_rate": 0.04, "max_depth": 6, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.10, "loss": "squared_error"}),
    make_candidate("hist_gbr_robust", "hist_gbr", params={"learning_rate": 0.03, "max_depth": 5, "max_leaf_nodes": 63, "min_samples_leaf": 16, "l2_regularization": 0.25, "loss": "absolute_error"}),
    make_candidate("blend_linear_tree", "blend", params={}, blend_with=("elasticnet_balanced", "extra_trees_balanced")),
    make_candidate("blend_huber_hist", "blend", params={}, blend_with=("huber_anchor", "hist_gbr_balanced")),
]


warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search non-XGBoost regressors on cached transformer embeddings and descriptor features for cycpeptmpdb_perm.")
    parser.add_argument("--task", default="cycpeptmpdb_perm")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--candidate_names", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs_sklearn_cached_regression_search")
    parser.add_argument("--cache_root", type=Path, default=REPO_ROOT / "tmp" / "embeddings_regression")
    parser.add_argument("--prepared_data_root", type=Path, default=REPO_ROOT / "tmp" / "prepared_data")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--rebalance_train_bins", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebalance_bin_count", type=int, default=5)
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        return list(DEFAULT_MODELS)
    if args.models:
        return list(args.models)
    if args.model:
        return [args.model]
    return list(DEFAULT_MODELS)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    if args.seed is not None:
        return [int(args.seed)]
    return list(DEFAULT_SEEDS)


def resolve_candidates(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.candidate_names:
        return list(CANDIDATE_CONFIGS)
    name_set = set(args.candidate_names)
    selected = [candidate for candidate in CANDIDATE_CONFIGS if str(candidate["name"]) in name_set]
    if not selected:
        raise ValueError(f"No candidates matched: {sorted(name_set)}")
    return selected


def descriptor_matrix(feature_set: str, smiles_list: list[str]) -> np.ndarray:
    if feature_set == "rdkit":
        return rdkit_feature_matrix(smiles_list)
    if feature_set == "morgan":
        return morgan_feature_matrix(smiles_list)
    raise ValueError(f"Unsupported descriptor feature set: {feature_set}")


def get_cached_descriptor_features(feature_set: str, smiles_list: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"[cache-hit] descriptor features: {feature_set} -> {cache_path}", flush=True)
        return np.load(cache_path)
    print(f"[descriptor-start] computing {feature_set} features for {len(smiles_list)} molecules", flush=True)
    features = descriptor_matrix(feature_set, smiles_list)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    print(f"[descriptor-done] {feature_set} features -> {cache_path}", flush=True)
    return features


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        spearman = spearmanr(y_true, y_pred)
    spearman_value = getattr(spearman, "statistic", np.nan)
    calibration_model = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    error = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearman_value if np.isfinite(spearman_value) else np.nan),
        "bias": float(np.mean(error)),
        "calibration_slope": float(calibration_model.coef_[0]),
        "calibration_intercept": float(calibration_model.intercept_),
    }


def upsample_target_bins(train_x: np.ndarray, train_y: np.ndarray, bin_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(train_y) < 2 or bin_count <= 1:
        return train_x, train_y

    target_frame = pd.DataFrame({"target": np.asarray(train_y, dtype=np.float32)})
    target_bins = pd.cut(target_frame["target"], bins=bin_count, duplicates="drop")
    if target_bins.isna().all():
        return train_x, train_y

    bin_codes = target_bins.cat.codes.to_numpy()
    valid_codes = bin_codes[bin_codes >= 0]
    if len(valid_codes) <= 1:
        return train_x, train_y

    unique_codes, counts = np.unique(valid_codes, return_counts=True)
    if len(unique_codes) <= 1:
        return train_x, train_y

    rng = np.random.default_rng(seed)
    max_count = int(np.max(counts))
    sampled_indices: list[np.ndarray] = []
    for code in unique_codes:
        group_indices = np.where(bin_codes == code)[0]
        sampled_group = rng.choice(group_indices, size=max_count, replace=len(group_indices) < max_count)
        sampled_indices.append(sampled_group.astype(np.int64))

    balanced_indices = np.concatenate(sampled_indices)
    rng.shuffle(balanced_indices)
    return train_x[balanced_indices], train_y[balanced_indices]


def make_estimator(candidate: dict[str, object], seed: int, n_jobs: int):
    estimator_type = str(candidate["estimator_type"])
    params = dict(candidate["params"])
    if estimator_type == "ridge":
        estimator = Ridge(**params)
    elif estimator_type == "elasticnet":
        estimator = ElasticNet(random_state=seed, **params)
    elif estimator_type == "huber":
        estimator = HuberRegressor(**params)
    elif estimator_type == "extra_trees":
        estimator = ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **params)
    elif estimator_type == "hist_gbr":
        estimator = HistGradientBoostingRegressor(random_state=seed, **params)
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")

    if bool(candidate.get("scale_features", False)):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
    return estimator


def fit_predict_candidate(
    candidate: dict[str, object],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if str(candidate["estimator_type"]) == "blend":
        blend_names = tuple(candidate["blend_with"])
        val_prediction_columns = []
        test_prediction_columns = []
        for blend_name in blend_names:
            sub_candidate = next(config for config in CANDIDATE_CONFIGS if str(config["name"]) == blend_name)
            val_predictions, test_predictions = fit_predict_candidate(
                sub_candidate,
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                seed,
                n_jobs,
            )
            val_prediction_columns.append(val_predictions)
            test_prediction_columns.append(test_predictions)

        val_prediction_matrix = np.column_stack(val_prediction_columns)
        test_prediction_matrix = np.column_stack(test_prediction_columns)
        # Learn non-negative inner-fold blend weights instead of forcing a 50/50 average.
        blend_model = LinearRegression(positive=True)
        blend_model.fit(val_prediction_matrix, val_y)
        blended_val_predictions = blend_model.predict(val_prediction_matrix).astype(np.float32)
        blended_test_predictions = blend_model.predict(test_prediction_matrix).astype(np.float32)
        return blended_val_predictions, blended_test_predictions

    estimator = make_estimator(candidate, seed=seed, n_jobs=n_jobs)
    estimator.fit(train_x, train_y)
    val_predictions = np.asarray(estimator.predict(val_x), dtype=np.float32)
    test_predictions = np.asarray(estimator.predict(test_x), dtype=np.float32)
    return val_predictions, test_predictions


def round_robin_ensemble_predictions(
    features: np.ndarray,
    data_frame: pd.DataFrame,
    candidate: dict[str, object],
    seed: int,
    n_jobs: int,
    rebalance_train_bins: bool,
    rebalance_bin_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    fold_ids = sorted(data_frame["fold"].unique().tolist())
    predictions = np.zeros(len(data_frame), dtype=np.float32)
    prediction_std = np.zeros(len(data_frame), dtype=np.float32)
    prediction_columns = {
        f"prediction_{val_fold}": np.full(len(data_frame), np.nan, dtype=np.float32)
        for val_fold in fold_ids
    }

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
            if rebalance_train_bins:
                train_x, train_y = upsample_target_bins(train_x, train_y, rebalance_bin_count, seed + int(fold_id) + int(val_fold))

            _, current_predictions = fit_predict_candidate(
                candidate=candidate,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                seed=seed + int(fold_id) + int(val_fold),
                n_jobs=n_jobs,
            )
            fold_predictions.append(current_predictions)
            prediction_columns[f"prediction_{val_fold}"][test_mask.to_numpy()] = current_predictions

        stacked = np.vstack(fold_predictions)
        predictions[test_mask.to_numpy()] = stacked.mean(axis=0)
        prediction_std[test_mask.to_numpy()] = stacked.std(axis=0)

    return predictions, prediction_std, prediction_columns, len(fold_ids)


def build_detailed_prediction_frame(
    data_frame: pd.DataFrame,
    prediction_columns: dict[str, np.ndarray],
    mean_prediction: np.ndarray,
    std_prediction: np.ndarray,
) -> pd.DataFrame:
    detailed_frame = data_frame.copy()
    for column_name, column_values in sorted(prediction_columns.items()):
        detailed_frame[column_name] = column_values
    detailed_frame["mean_prediction"] = mean_prediction
    detailed_frame["std_prediction"] = std_prediction
    detailed_frame["prediction"] = mean_prediction
    detailed_frame["prediction_std"] = std_prediction
    return detailed_frame


def has_completed_run(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.csv"
    return metrics_path.exists() and metrics_path.stat().st_size > 0


def load_summary_row(run_dir: Path, model_name: str, seed: int, candidate_name: str) -> dict[str, object]:
    metrics_frame = pd.read_csv(run_dir / "metrics.csv")
    summary = {"model_name": model_name, "seed": seed, "candidate": candidate_name, "run_dir": str(run_dir)}
    for _, row in metrics_frame.iterrows():
        summary[str(row["metric_name"])] = float(row["metric_value"])
    return summary


def evaluate_feature_source(
    *,
    model_name: str,
    model_variant: str,
    model_family: str,
    features: np.ndarray,
    data_frame: pd.DataFrame,
    candidates: list[dict[str, object]],
    seeds: list[int],
    output_root: Path,
    rebalance_train_bins: bool,
    rebalance_bin_count: int,
    n_jobs: int,
    summary_rows: list[dict[str, object]],
) -> None:
    print(f"[feature-start] family={model_family} name={model_name} dim={features.shape[1]}", flush=True)
    for candidate in candidates:
        candidate_name = str(candidate["name"])
        print(f"[candidate-start] family={model_family} name={model_name} candidate={candidate_name}", flush=True)
        for seed in seeds:
            run_dir = output_root / data_frame.attrs.get("task", "cycpeptmpdb_perm") / model_family / candidate_name / model_variant / f"seed_{seed}"
            if has_completed_run(run_dir):
                summary_rows.append(load_summary_row(run_dir, model_name=model_name, seed=seed, candidate_name=candidate_name))
                print(f"[skip-seed] family={model_family} name={model_name} candidate={candidate_name} seed={seed}", flush=True)
                continue

            predictions, prediction_std, prediction_columns, fold_count = round_robin_ensemble_predictions(
                features=features,
                data_frame=data_frame,
                candidate=candidate,
                seed=seed,
                n_jobs=n_jobs,
                rebalance_train_bins=rebalance_train_bins,
                rebalance_bin_count=rebalance_bin_count,
            )
            metrics = compute_regression_metrics(data_frame["value"].to_numpy(dtype=np.float32), predictions)

            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "task": data_frame.attrs.get("task", "cycpeptmpdb_perm"),
                "model_name": model_name,
                "seed": seed,
                "status": "ready",
                "feature_source": model_family,
                "strategy": "provided_fold_round_robin_ensemble",
                "fold_count": fold_count,
                "candidate": candidate,
                "rebalance_train_bins": bool(rebalance_train_bins),
                "rebalance_bin_count": int(rebalance_bin_count),
            }
            (run_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            prediction_frame = build_prediction_frame(
                task_id=data_frame.attrs.get("task", "cycpeptmpdb_perm"),
                model_family=model_family,
                model_variant=f"{candidate_name}-{model_variant}",
                seed=seed,
                split_id="cv_test",
                sample_ids=data_frame["SMILES"],
                input_values=data_frame["SMILES"],
                true_targets=data_frame["value"],
                predictions=predictions,
                prediction_type="regression",
                threshold=None,
            )
            metric_frame = build_metric_frame(
                task_id=data_frame.attrs.get("task", "cycpeptmpdb_perm"),
                model_family=model_family,
                model_variant=f"{candidate_name}-{model_variant}",
                seed=seed,
                split_id="cv_test",
                metrics=metrics,
                primary_metric_names={"r2", "rmse", "mae", "spearman"},
            )
            write_baseline_outputs(
                output_dir=run_dir,
                prediction_frame=prediction_frame,
                metric_frame=metric_frame,
                adapter_metadata=payload,
            )

            detailed_frame = build_detailed_prediction_frame(
                data_frame=data_frame,
                prediction_columns=prediction_columns,
                mean_prediction=predictions,
                std_prediction=prediction_std,
            )
            detailed_frame.to_csv(run_dir / "cv_predictions_detailed.csv", index=False)

            summary = {"model_name": model_name, "seed": seed, "candidate": candidate_name, "run_dir": str(run_dir)}
            summary.update(metrics)
            summary_rows.append(summary)
            print(
                f"[seed-done] family={model_family} name={model_name} candidate={candidate_name} seed={seed} r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} slope={metrics['calibration_slope']:.3f}",
                flush=True,
            )


def main() -> int:
    args = parse_args()
    if args.task != "cycpeptmpdb_perm":
        raise ValueError("This search currently supports only cycpeptmpdb_perm.")

    data_frame = load_task_frames(args.task, seed=0, prepared_data_root=args.prepared_data_root)["full"]
    data_frame = data_frame.copy().sort_values("fold").reset_index(drop=True)
    data_frame.attrs["task"] = args.task
    models = resolve_models(args)
    seeds = resolve_seeds(args)
    candidates = resolve_candidates(args)
    smiles_values = data_frame["SMILES"].tolist()
    effective_n_jobs = os.cpu_count() if args.n_jobs == -1 else args.n_jobs

    print(
        f"[run-start] task={args.task} transformer_models={len(models)} descriptor_sets={len(DESCRIPTOR_FEATURE_SETS)} seeds={seeds} candidates={len(candidates)} n_jobs={effective_n_jobs} rebalance_train_bins={args.rebalance_train_bins} rebalance_bin_count={args.rebalance_bin_count}",
        flush=True,
    )

    summary_rows: list[dict[str, object]] = []
    for model_name in models:
        model_variant = model_name.split("/")[-1].replace("-", "_")
        cache_path = args.cache_root / model_variant / args.task / "full.npy"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached embeddings not found: {cache_path}")
        features = np.load(cache_path)
        evaluate_feature_source(
            model_name=model_name,
            model_variant=model_variant,
            model_family="sklearn_transformer",
            features=features,
            data_frame=data_frame,
            candidates=candidates,
            seeds=seeds,
            output_root=args.output_root,
            rebalance_train_bins=args.rebalance_train_bins,
            rebalance_bin_count=args.rebalance_bin_count,
            n_jobs=effective_n_jobs,
            summary_rows=summary_rows,
        )

    for feature_set in DESCRIPTOR_FEATURE_SETS:
        descriptor_variant = f"sklearn-{feature_set}"
        cache_path = args.cache_root / f"xgboost-{feature_set}" / args.task / "full.npy"
        features = get_cached_descriptor_features(feature_set, smiles_values, cache_path)
        evaluate_feature_source(
            model_name=feature_set,
            model_variant=descriptor_variant,
            model_family=f"sklearn_{feature_set}",
            features=features,
            data_frame=data_frame,
            candidates=candidates,
            seeds=seeds,
            output_root=args.output_root,
            rebalance_train_bins=args.rebalance_train_bins,
            rebalance_bin_count=args.rebalance_bin_count,
            n_jobs=effective_n_jobs,
            summary_rows=summary_rows,
        )

    summary_root = args.output_root / args.task / "sklearn_search"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values(["r2", "spearman"], ascending=[False, False]).reset_index(drop=True)
    summary_frame.to_csv(summary_root / "summary_metrics.csv", index=False)
    print(f"[run-done] wrote summary -> {summary_root / 'summary_metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())