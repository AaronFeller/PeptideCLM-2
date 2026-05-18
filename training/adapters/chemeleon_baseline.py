from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    compute_classification_metrics,
    compute_regression_metrics,
    get_task_spec,
    load_task_frames,
    write_baseline_outputs,
)
from training.experiment.manifest import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CheMeleon baseline via the ChemProp CLI.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prepared_data_root", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def chemprop_binary() -> str:
    binary = shutil.which("chemprop")
    if binary is None:
        raise FileNotFoundError("chemprop CLI is not available in PATH for the active environment.")
    return binary


def export_training_csv(df: pd.DataFrame, sample_col: str, input_col: str, label_col: str, path: Path) -> None:
    export_df = pd.DataFrame(
        {
            "sample_id": df[sample_col].astype(str),
            "smiles": df[input_col].astype(str),
            "target": df[label_col],
        }
    )
    export_df.to_csv(path, index=False)


def export_prediction_csv(df: pd.DataFrame, sample_col: str, input_col: str, path: Path) -> None:
    export_df = pd.DataFrame(
        {
            "sample_id": df[sample_col].astype(str),
            "smiles": df[input_col].astype(str),
        }
    )
    export_df.to_csv(path, index=False)


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def predict_column(prediction_input_csv: Path, prediction_output_csv: Path) -> pd.Series:
    input_columns = set(pd.read_csv(prediction_input_csv, nrows=1).columns)
    output_df = pd.read_csv(prediction_output_csv)
    prediction_columns = [column for column in output_df.columns if column not in input_columns]
    if not prediction_columns:
        raise ValueError(f"Could not identify prediction columns in {prediction_output_csv}")
    return output_df[prediction_columns[0]]


def train_and_predict_once(
    *,
    task_type: str,
    seed: int,
    gpu_index: int,
    batch_size: int,
    epochs: int,
    patience: int,
    working_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_col: str,
    input_col: str,
    label_col: str,
) -> np.ndarray:
    chemprop = chemprop_binary()
    working_dir.mkdir(parents=True, exist_ok=True)

    train_csv = working_dir / "train.csv"
    val_csv = working_dir / "val.csv"
    test_csv = working_dir / "test.csv"
    predict_input_csv = working_dir / "predict_input.csv"
    predict_output_csv = working_dir / "predict_output.csv"
    model_dir = working_dir / "chemprop_model"

    export_training_csv(train_df, sample_col, input_col, label_col, train_csv)
    export_training_csv(val_df, sample_col, input_col, label_col, val_csv)
    export_training_csv(test_df, sample_col, input_col, label_col, test_csv)
    export_prediction_csv(test_df, sample_col, input_col, predict_input_csv)

    metrics = ["roc", "f1", "accuracy"] if task_type == "classification" else ["rmse", "r2", "mae"]
    tracking_metric = "roc" if task_type == "classification" else "rmse"

    train_command = [
        chemprop,
        "train",
        "--data-path",
        str(train_csv),
        str(val_csv),
        str(test_csv),
        "--output-dir",
        str(model_dir),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "target",
        "--task-type",
        task_type,
        "--metrics",
        *metrics,
        "--tracking-metric",
        tracking_metric,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "0",
        "--pytorch-seed",
        str(seed),
        "--data-seed",
        str(seed),
        "--from-foundation",
        "CheMeleon",
        "--remove-checkpoints",
    ]
    if task_type == "classification":
        train_command.append("--class-balance")
    if shutil.which("nvidia-smi") is not None:
        train_command.extend(["--accelerator", "gpu", "--devices", str(gpu_index)])
    run_command(train_command)

    predict_command = [
        chemprop,
        "predict",
        "--test-path",
        str(predict_input_csv),
        "--model-paths",
        str(model_dir),
        "--output",
        str(predict_output_csv),
        "--smiles-columns",
        "smiles",
        "--num-workers",
        "0",
        "--batch-size",
        str(batch_size),
    ]
    if shutil.which("nvidia-smi") is not None:
        predict_command.extend(["--accelerator", "gpu", "--devices", str(gpu_index)])
    run_command(predict_command)
    return predict_column(predict_input_csv, predict_output_csv).to_numpy(dtype=np.float32)


def classification_predictions(args: argparse.Namespace, task_spec) -> tuple[np.ndarray, pd.DataFrame, dict]:
    frames = load_task_frames(args.task, args.seed, args.prepared_data_root)
    train_df = frames["train"]
    test_df = frames["test"]

    if "val" in frames:
        predictions = train_and_predict_once(
            task_type=task_spec.task_type,
            seed=args.seed,
            gpu_index=args.gpu_index,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            working_dir=args.output_dir / "chemeleon_work" / "fold_0",
            train_df=train_df,
            val_df=frames["val"],
            test_df=test_df,
            sample_col=task_spec.sample_id_column,
            input_col=task_spec.input_column,
            label_col=task_spec.label_column,
        )
        return predictions, test_df, {"strategy": "fixed_train_val_test", "fold_count": 1}

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_labels = train_df[task_spec.label_column].to_numpy(dtype=np.int32)
    fold_predictions = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(train_df, train_labels), start=1):
        fold_predictions.append(
            train_and_predict_once(
                task_type=task_spec.task_type,
                seed=args.seed + fold_index,
                gpu_index=args.gpu_index,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                working_dir=args.output_dir / "chemeleon_work" / f"fold_{fold_index}",
                train_df=train_df.iloc[fit_idx],
                val_df=train_df.iloc[val_idx],
                test_df=test_df,
                sample_col=task_spec.sample_id_column,
                input_col=task_spec.input_column,
                label_col=task_spec.label_column,
            )
        )
    predictions = np.mean(np.vstack(fold_predictions), axis=0)
    return predictions, test_df, {"strategy": "five_fold_test_ensemble", "fold_count": 5}


def regression_predictions(args: argparse.Namespace, task_spec) -> tuple[np.ndarray, pd.DataFrame, dict]:
    full_df = load_task_frames(args.task, args.seed, args.prepared_data_root)["full"].copy().sort_values("fold").reset_index(drop=True)
    fold_ids = sorted(full_df["fold"].unique().tolist())
    predictions = np.zeros(len(full_df), dtype=np.float32)
    for fold_id in fold_ids:
        test_mask = full_df["fold"] == fold_id
        val_fold = fold_ids[(fold_ids.index(fold_id) - 1) % len(fold_ids)]
        val_mask = full_df["fold"] == val_fold
        train_mask = ~(test_mask | val_mask)
        predictions[test_mask.to_numpy()] = train_and_predict_once(
            task_type=task_spec.task_type,
            seed=args.seed + int(fold_id),
            gpu_index=args.gpu_index,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            working_dir=args.output_dir / "chemeleon_work" / f"fold_{int(fold_id)}",
            train_df=full_df.loc[train_mask],
            val_df=full_df.loc[val_mask],
            test_df=full_df.loc[test_mask],
            sample_col=task_spec.sample_id_column,
            input_col=task_spec.input_column,
            label_col=task_spec.label_column,
        )
    return predictions, full_df, {"strategy": "provided_fold_cv", "fold_count": len(fold_ids)}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args.task,
        "model_name": args.model_name,
        "seed": args.seed,
        "status": "ready" if not args.dry_run else "dry_run",
        "foundation_flag": "--from-foundation CheMeleon",
        "release_hint": "v1.0.0",
    }
    (args.output_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if args.dry_run:
        return 0

    task_spec = get_task_spec(args.task)
    model_family = "chemeleon"
    model_variant = args.model_name.split("/")[-1].replace(".", "_")

    if task_spec.task_type == "classification":
        predictions, test_df, metadata = classification_predictions(args, task_spec)
        metrics = compute_classification_metrics(test_df[task_spec.label_column].to_numpy(dtype=np.int32), predictions)
        prediction_frame = build_prediction_frame(
            task_id=args.task,
            model_family=model_family,
            model_variant=model_variant,
            seed=args.seed,
            split_id="test",
            sample_ids=test_df[task_spec.sample_id_column],
            input_values=test_df[task_spec.input_column],
            true_targets=test_df[task_spec.label_column],
            predictions=predictions,
            prediction_type="probability",
            threshold=0.5,
        )
    else:
        predictions, test_df, metadata = regression_predictions(args, task_spec)
        metrics = compute_regression_metrics(test_df[task_spec.label_column].to_numpy(dtype=np.float32), predictions)
        prediction_frame = build_prediction_frame(
            task_id=args.task,
            model_family=model_family,
            model_variant=model_variant,
            seed=args.seed,
            split_id="cv_test",
            sample_ids=test_df[task_spec.sample_id_column],
            input_values=test_df[task_spec.input_column],
            true_targets=test_df[task_spec.label_column],
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
        adapter_metadata={
            **payload,
            **metadata,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())