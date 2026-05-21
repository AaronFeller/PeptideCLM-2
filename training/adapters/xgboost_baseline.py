from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    compute_classification_metrics,
    compute_regression_metrics,
    get_task_spec,
    load_task_frames,
    morgan_feature_matrix,
    rdkit_feature_matrix,
    write_baseline_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost boosted-tree baseline over RDKit descriptors or Morgan fingerprints.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--feature_set", required=True, choices=["rdkit", "morgan"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prepared_data_root", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def feature_matrix(feature_set: str, smiles_values: list[str]) -> np.ndarray:
    if feature_set == "rdkit":
        return rdkit_feature_matrix(smiles_values)
    if feature_set == "morgan":
        return morgan_feature_matrix(smiles_values)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def resolve_smiles_column(frame) -> str:
    if "smiles" in frame.columns:
        return "smiles"
    if "SMILES" in frame.columns:
        return "SMILES"
    raise ValueError("Expected a smiles or SMILES column in the input frame.")


def fit_classification_with_val(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, seed: int) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=512,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def fit_regression_model(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, seed: int) -> XGBRegressor:
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
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def run_fixed_classification(args: argparse.Namespace, model_family: str, model_variant: str) -> tuple[np.ndarray, dict, dict]:
    frames = load_task_frames(args.task, seed=args.seed, prepared_data_root=args.prepared_data_root)
    train_df = frames["train"]
    test_df = frames["test"]
    smiles_col = resolve_smiles_column(train_df)
    test_smiles_col = resolve_smiles_column(test_df)

    if "val" in frames:
        val_df = frames["val"]
        val_smiles_col = resolve_smiles_column(val_df)
        train_x = feature_matrix(args.feature_set, train_df[smiles_col].tolist())
        train_y = train_df["label"].to_numpy(dtype=np.int32)
        val_x = feature_matrix(args.feature_set, val_df[val_smiles_col].tolist())
        val_y = val_df["label"].to_numpy(dtype=np.int32)
        test_x = feature_matrix(args.feature_set, test_df[test_smiles_col].tolist())
        model = fit_classification_with_val(train_x, train_y, val_x, val_y, args.seed)
        test_pred = model.predict_proba(test_x)[:, 1]
        metadata = {"strategy": "fixed_train_val_test", "fold_count": 1}
        return test_pred, {"test": test_df}, metadata

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_x = feature_matrix(args.feature_set, train_df[smiles_col].tolist())
    train_y = train_df["label"].to_numpy(dtype=np.int32)
    test_x = feature_matrix(args.feature_set, test_df[test_smiles_col].tolist())
    fold_predictions = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(train_x, train_y), start=1):
        model = fit_classification_with_val(train_x[fit_idx], train_y[fit_idx], train_x[val_idx], train_y[val_idx], args.seed + fold_index)
        fold_predictions.append(model.predict_proba(test_x)[:, 1])
    test_pred = np.mean(np.vstack(fold_predictions), axis=0)
    metadata = {"strategy": "five_fold_test_ensemble", "fold_count": 5}
    return test_pred, {"test": test_df}, metadata


def run_regression_cv(args: argparse.Namespace, model_family: str, model_variant: str) -> tuple[np.ndarray, dict, dict]:
    full_df = load_task_frames(args.task, seed=args.seed, prepared_data_root=args.prepared_data_root)["full"]
    full_df = full_df.copy().sort_values("fold").reset_index(drop=True)
    smiles_col = resolve_smiles_column(full_df)
    predictions = np.zeros(len(full_df), dtype=np.float32)
    prediction_std = np.zeros(len(full_df), dtype=np.float32)
    fold_ids = sorted(full_df["fold"].unique().tolist())

    for fold_id in fold_ids:
        test_mask = full_df["fold"] == fold_id
        val_fold = fold_ids[(fold_ids.index(fold_id) - 1) % len(fold_ids)]
        val_mask = full_df["fold"] == val_fold
        train_mask = ~(test_mask | val_mask)

        train_df = full_df.loc[train_mask]
        val_df = full_df.loc[val_mask]
        test_df = full_df.loc[test_mask]

        train_x = feature_matrix(args.feature_set, train_df["smiles"].tolist())
        val_x = feature_matrix(args.feature_set, val_df["smiles"].tolist())
        test_x = feature_matrix(args.feature_set, test_df["smiles"].tolist())
        train_y = train_df["value"].to_numpy(dtype=np.float32)
        val_y = val_df["value"].to_numpy(dtype=np.float32)

        model = fit_regression_model(train_x, train_y, val_x, val_y, args.seed + int(fold_id))
        predictions[test_mask.to_numpy()] = model.predict(test_x)

    metadata = {"strategy": "provided_fold_cv", "fold_count": len(fold_ids)}
    return predictions, {"test": full_df}, metadata


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_family = f"xgboost_{args.feature_set}"
    model_variant = f"xgboost-{args.feature_set}"

    payload = {
        "task": args.task,
        "feature_set": args.feature_set,
        "seed": args.seed,
        "status": "ready" if not args.dry_run else "dry_run",
    }
    (args.output_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if args.dry_run:
        return 0

    task_spec = get_task_spec(args.task)
    if task_spec.task_type == "classification":
        predictions, split_frames, metadata = run_fixed_classification(args, model_family, model_variant)
        test_df = split_frames["test"]
        smiles_col = resolve_smiles_column(test_df)
        metrics = compute_classification_metrics(test_df["label"].to_numpy(dtype=np.int32), predictions)
        prediction_frame = build_prediction_frame(
            task_id=args.task,
            model_family=model_family,
            model_variant=model_variant,
            seed=args.seed,
            split_id="test",
            sample_ids=test_df[task_spec.sample_id_column],
            input_values=test_df[smiles_col],
            true_targets=test_df["label"],
            predictions=predictions,
            prediction_type="probability",
            threshold=0.5,
        )
    else:
        predictions, split_frames, metadata = run_regression_cv(args, model_family, model_variant)
        test_df = split_frames["test"]
        smiles_col = resolve_smiles_column(test_df)
        metrics = compute_regression_metrics(test_df["value"].to_numpy(dtype=np.float32), predictions)
        prediction_frame = build_prediction_frame(
            task_id=args.task,
            model_family=model_family,
            model_variant=model_variant,
            seed=args.seed,
            split_id="cv_test",
            sample_ids=test_df[task_spec.sample_id_column],
            input_values=test_df[smiles_col],
            true_targets=test_df["value"],
            predictions=predictions,
            prediction_type="regression",
            threshold=None,
        )

    metric_frame = build_metric_frame(
        task_id=args.task,
        model_family=model_family,
        model_variant=model_variant,
        seed=args.seed,
        split_id=prediction_frame["split_id"].iloc[0],
        metrics=metrics,
        primary_metric_names={"mcc", "auroc", "f1", "r2", "rmse", "mae"},
    )
    write_baseline_outputs(
        output_dir=args.output_dir,
        prediction_frame=prediction_frame,
        metric_frame=metric_frame,
        adapter_metadata={**payload, **metadata},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())