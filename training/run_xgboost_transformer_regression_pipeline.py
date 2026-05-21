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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tokenizers
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBRegressor

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    load_task_frames,
    morgan_feature_matrix,
    rdkit_feature_matrix,
    write_baseline_outputs,
)
from training.experiment.manifest import REPO_ROOT


torch.serialization.add_safe_globals([
    transformers.tokenization_utils_tokenizers.TokenizersBackend,
    tokenizers.Tokenizer,
])

warnings.filterwarnings("ignore")

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


def extract_mean_pool(outputs) -> torch.Tensor:
    if hasattr(outputs, "mean_pool"):
        return outputs.mean_pool
    if isinstance(outputs, dict) and "mean_pool" in outputs:
        return outputs["mean_pool"]
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if isinstance(outputs, dict) and "pooler_output" in outputs and outputs["pooler_output"] is not None:
        return outputs["pooler_output"]
    last_hidden_state = None
    if hasattr(outputs, "last_hidden_state"):
        last_hidden_state = outputs.last_hidden_state
    elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
        last_hidden_state = outputs["last_hidden_state"]
    if last_hidden_state is None:
        raise ValueError("Model output does not contain mean_pool or last_hidden_state.")
    return last_hidden_state.mean(dim=1)


def forward_backbone(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        return model(input_ids, mask=attention_mask)


@torch.no_grad()
def compute_embeddings(model, tokenizer, device: torch.device, smiles_list: list[str], batch_size: int, max_length: int) -> np.ndarray:
    model.eval()
    embeddings = []
    for start in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[start : start + batch_size]
        batch_inputs = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
        outputs = forward_backbone(model, input_ids=batch_inputs["input_ids"], attention_mask=batch_inputs["attention_mask"])
        embeddings.append(extract_mean_pool(outputs).detach().cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)


def get_cached_embeddings(model_name: str, tokenizer, device: torch.device, smiles_list: list[str], cache_path: Path, batch_size: int, max_length: int) -> np.ndarray:
    if cache_path.exists():
        print(f"[cache-hit] transformer embeddings: {model_name} -> {cache_path}", flush=True)
        return np.load(cache_path)
    print(f"[embed-start] transformer embeddings: {model_name} on {device} ({len(smiles_list)} molecules)", flush=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    embeddings = compute_embeddings(model, tokenizer, device, smiles_list, batch_size=batch_size, max_length=max_length)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"[embed-done] transformer embeddings: {model_name} -> {cache_path}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


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


def compute_regression_metrics_with_mse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        spearman = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearman.statistic if np.isfinite(spearman.statistic) else np.nan),
    }


def fit_regression_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    n_jobs: int,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_estimators=512,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device="cpu",
        n_jobs=n_jobs,
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression-only transformer embedding plus XGBoost pipeline for the CycPeptMPDB PAMPA benchmark.")
    parser.add_argument("--task", default="cycpeptmpdb_perm")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs_xgboost_transformer_regression")
    parser.add_argument("--cache_root", type=Path, default=REPO_ROOT / "tmp" / "embeddings_regression")
    parser.add_argument("--prepared_data_root", type=Path, default=REPO_ROOT / "tmp" / "prepared_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--n_jobs", type=int, default=-1)
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


def round_robin_ensemble_predictions(
    features: np.ndarray,
    data_frame: pd.DataFrame,
    seed: int,
    n_jobs: int,
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
        ensemble_df = data_frame.loc[~test_mask].copy().reset_index(drop=True)
        test_x = features[test_mask.to_numpy()]
        fold_predictions = []

        for val_fold in sorted(ensemble_df["fold"].unique().tolist()):
            train_mask = (~test_mask) & (data_frame["fold"] != val_fold)
            val_mask = data_frame["fold"] == val_fold

            train_x = features[train_mask.to_numpy()]
            val_x = features[val_mask.to_numpy()]
            train_y = data_frame.loc[train_mask, "value"].to_numpy(dtype=np.float32)
            val_y = data_frame.loc[val_mask, "value"].to_numpy(dtype=np.float32)

            regressor = fit_regression_model(
                train_x,
                train_y,
                val_x,
                val_y,
                seed + int(fold_id) + int(val_fold),
                n_jobs=n_jobs,
            )
            current_predictions = regressor.predict(test_x).astype(np.float32)
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


def load_summary_row(run_dir: Path, model_name: str, seed: int) -> dict[str, object]:
    metrics_frame = pd.read_csv(run_dir / "metrics.csv")
    summary = {"model_name": model_name, "seed": seed, "run_dir": str(run_dir)}
    for _, row in metrics_frame.iterrows():
        summary[str(row["metric_name"])] = float(row["metric_value"])
    return summary


def main() -> int:
    args = parse_args()
    if args.task != "cycpeptmpdb_perm":
        raise ValueError("This pipeline is regression-only and currently supports only cycpeptmpdb_perm.")

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    data_frame = load_task_frames(args.task, seed=0, prepared_data_root=args.prepared_data_root)["full"]
    data_frame = data_frame.copy().sort_values("fold").reset_index(drop=True)
    models = resolve_models(args)
    seeds = resolve_seeds(args)
    smiles_values = data_frame["SMILES"].tolist()
    effective_n_jobs = os.cpu_count() if args.n_jobs == -1 else args.n_jobs

    print(
        f"[run-start] task={args.task} device={device} transformer_models={len(models)} descriptor_sets={len(DESCRIPTOR_FEATURE_SETS)} seeds={seeds} xgboost_n_jobs={effective_n_jobs}",
        flush=True,
    )

    if args.cache_only:
        print("[cache-only] generating feature caches and exiting before XGBoost fitting", flush=True)

    summary_rows = []
    for model_name in models:
        model_variant = model_name.split("/")[-1].replace("-", "_")
        print(f"[model-start] transformer model={model_name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        cache_path = args.cache_root / model_variant / args.task / "full.npy"
        features = get_cached_embeddings(
            model_name=model_name,
            tokenizer=tokenizer,
            device=device,
            smiles_list=data_frame["SMILES"].tolist(),
            cache_path=cache_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        if args.cache_only:
            print(f"[model-cached] transformer model={model_name}", flush=True)
            continue

        for seed in seeds:
            run_dir = args.output_root / args.task / "xgboost_transformer" / model_variant / f"seed_{seed}"
            if has_completed_run(run_dir):
                summary_rows.append(load_summary_row(run_dir, model_name=model_name, seed=seed))
                print(f"[skip-seed] transformer model={model_name} seed={seed} found completed output", flush=True)
                continue

            print(f"[seed-start] transformer model={model_name} seed={seed}", flush=True)
            predictions, prediction_std, prediction_columns, fold_count = round_robin_ensemble_predictions(
                features,
                data_frame,
                seed,
                n_jobs=args.n_jobs,
            )

            metrics = compute_regression_metrics_with_mse(
                data_frame["value"].to_numpy(dtype=np.float32),
                predictions,
            )

            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "task": args.task,
                "model_name": model_name,
                "seed": seed,
                "status": "ready",
                "feature_source": "transformer_embedding",
                "strategy": "provided_fold_round_robin_ensemble",
                "fold_count": fold_count,
            }
            (run_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            prediction_frame = build_prediction_frame(
                task_id=args.task,
                model_family="xgboost_transformer",
                model_variant=f"xgboost-{model_variant}",
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
                task_id=args.task,
                model_family="xgboost_transformer",
                model_variant=f"xgboost-{model_variant}",
                seed=seed,
                split_id="cv_test",
                metrics=metrics,
                primary_metric_names={"r2", "rmse", "mae"},
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

            summary = {"model_name": model_name, "seed": seed, "run_dir": str(run_dir)}
            summary.update(metrics)
            summary_rows.append(summary)
            print(
                f"[seed-done] transformer model={model_name} seed={seed} r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f}",
                flush=True,
            )

    for feature_set in DESCRIPTOR_FEATURE_SETS:
        descriptor_variant = f"xgboost-{feature_set}"
        cache_path = args.cache_root / descriptor_variant / args.task / "full.npy"
        print(f"[model-start] descriptor feature_set={feature_set}", flush=True)
        features = get_cached_descriptor_features(feature_set, smiles_values, cache_path)

        if args.cache_only:
            print(f"[model-cached] descriptor feature_set={feature_set}", flush=True)
            continue

        for seed in seeds:
            run_dir = args.output_root / args.task / f"xgboost_{feature_set}" / descriptor_variant / f"seed_{seed}"
            if has_completed_run(run_dir):
                summary_rows.append(load_summary_row(run_dir, model_name=feature_set, seed=seed))
                print(f"[skip-seed] descriptor feature_set={feature_set} seed={seed} found completed output", flush=True)
                continue

            print(f"[seed-start] descriptor feature_set={feature_set} seed={seed}", flush=True)
            predictions, prediction_std, prediction_columns, fold_count = round_robin_ensemble_predictions(
                features,
                data_frame,
                seed,
                n_jobs=args.n_jobs,
            )
            metrics = compute_regression_metrics_with_mse(
                data_frame["value"].to_numpy(dtype=np.float32),
                predictions,
            )

            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "task": args.task,
                "feature_set": feature_set,
                "seed": seed,
                "status": "ready",
                "feature_source": "descriptor",
                "strategy": "provided_fold_round_robin_ensemble",
                "fold_count": fold_count,
            }
            (run_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            prediction_frame = build_prediction_frame(
                task_id=args.task,
                model_family=f"xgboost_{feature_set}",
                model_variant=descriptor_variant,
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
                task_id=args.task,
                model_family=f"xgboost_{feature_set}",
                model_variant=descriptor_variant,
                seed=seed,
                split_id="cv_test",
                metrics=metrics,
                primary_metric_names={"r2", "rmse", "mae"},
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

            summary = {"model_name": feature_set, "seed": seed, "run_dir": str(run_dir)}
            summary.update(metrics)
            summary_rows.append(summary)
        print(
        f"[seed-done] descriptor feature_set={feature_set} seed={seed} r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f}",
        flush=True,
        )

    if args.cache_only:
        print(f"[cache-only-done] caches available under {args.cache_root / args.task}", flush=True)
        return 0

    summary_root = args.output_root / args.task / "xgboost_transformer"
    summary_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_root / "summary_metrics.csv", index=False)
    print(f"[run-done] wrote summary -> {summary_root / 'summary_metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())