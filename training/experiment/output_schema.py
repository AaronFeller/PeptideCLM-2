from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PREDICTION_COLUMNS = [
    "task_id",
    "model_family",
    "model_variant",
    "seed",
    "split_id",
    "sample_id",
    "input_value",
    "true_target",
    "prediction",
    "prediction_type",
    "threshold",
]

METRIC_COLUMNS = [
    "task_id",
    "model_family",
    "model_variant",
    "seed",
    "split_id",
    "metric_name",
    "metric_value",
    "metric_role",
]


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    run_dir: Path
    prediction_path: Path
    metrics_path: Path
    metadata_path: Path
    log_dir: Path


def build_run_id(task_id: str, model_family: str, model_variant: str, seed: int, suffix: str | None = None) -> str:
    base = f"{task_id}__{model_family}__{model_variant}__seed{seed}"
    return f"{base}__{suffix}" if suffix else base


def build_run_layout(
    task_id: str,
    model_family: str,
    model_variant: str,
    seed: int,
    run_root: Path,
    log_root: Path,
    suffix: str | None = None,
) -> RunLayout:
    run_id = build_run_id(task_id, model_family, model_variant, seed, suffix=suffix)
    run_dir = run_root / task_id / model_family / model_variant / f"seed_{seed}"
    log_dir = log_root / task_id / model_family / model_variant / f"seed_{seed}"
    return RunLayout(
        run_id=run_id,
        run_dir=run_dir,
        prediction_path=run_dir / "predictions.csv",
        metrics_path=run_dir / "metrics.csv",
        metadata_path=run_dir / "run_metadata.json",
        log_dir=log_dir,
    )