from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from training.experiment.manifest import REPO_ROOT, load_manifest
from training.experiment.output_schema import METRIC_COLUMNS, PREDICTION_COLUMNS


RDKIT_DESCRIPTOR_NAMES = [
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "ExactMolWt",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "MaxAbsPartialCharge",
    "MaxPartialCharge",
    "MinAbsPartialCharge",
    "MinPartialCharge",
    "MolLogP",
    "MolMR",
    "MolWt",
    "RingCount",
    "HeavyAtomCount",
    "HeavyAtomMolWt",
    "FractionCSP3",
    "HallKierAlpha",
    "LabuteASA",
    "TPSA",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
]

_DESCRIPTOR_FUNCS = {name: func for name, func in Descriptors._descList if name in RDKIT_DESCRIPTOR_NAMES}
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    input_column: str
    label_column: str
    sample_id_column: str
    split_semantics: str
    split_files: dict[str, str]


def get_task_spec(task_id: str) -> TaskSpec:
    manifest = load_manifest()
    for task in manifest["tasks"]:
        if task["task_id"] == task_id:
            return TaskSpec(
                task_id=task["task_id"],
                task_type=task["task_type"],
                input_column=task["input_column"],
                label_column=task["label_column"],
                sample_id_column=task["sample_id_column"],
                split_semantics=task["split_semantics"],
                split_files=task["split_files"],
            )
    raise ValueError(f"Unknown task: {task_id}")


def load_task_frames(task_id: str, seed: int, prepared_data_root: Path | None = None) -> dict[str, pd.DataFrame]:
    prepared_root = prepared_data_root or (REPO_ROOT / "tmp" / "prepared_data")
    spec = get_task_spec(task_id)

    if task_id in {"amp_hgt", "cellppd"}:
        return {split: pd.read_csv(REPO_ROOT / rel_path) for split, rel_path in spec.split_files.items()}

    if task_id == "thpep":
        base = prepared_root / "thpep" / f"seed_{seed}"
        return {
            "train": pd.read_csv(base / "THPep_train.csv"),
            "val": pd.read_csv(base / "THPep_val.csv"),
            "test": pd.read_csv(base / "THPep_test.csv"),
        }

    if task_id == "cycpeptmpdb_perm":
        base = prepared_root / "cycpeptmpdb_perm"
        return {"full": pd.read_csv(base / "perm_external.csv")}

    if task_id == "pepmsnd":
        base = prepared_root / "pepmsnd"
        outputs = {}
        for fold_index in range(1, 11):
            outputs[f"train_{fold_index}"] = pd.read_csv(base / f"X_train{fold_index}.csv")
            outputs[f"test_{fold_index}"] = pd.read_csv(base / f"X_test{fold_index}.csv")
        return outputs

    raise ValueError(f"Task {task_id} is not supported by these baseline adapters.")


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def rdkit_feature_matrix(smiles_values: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for smiles in smiles_values:
        mol = smiles_to_mol(smiles)
        if mol is None:
            rows.append([0.0] * len(RDKIT_DESCRIPTOR_NAMES))
            continue
        values: list[float] = []
        for descriptor_name in RDKIT_DESCRIPTOR_NAMES:
            try:
                raw_value = _DESCRIPTOR_FUNCS[descriptor_name](mol)
                value = float(raw_value)
                if not np.isfinite(value):
                    value = 0.0
            except Exception:
                value = 0.0
            values.append(value)
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def morgan_feature_matrix(smiles_values: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for smiles in smiles_values:
        mol = smiles_to_mol(smiles)
        if mol is None:
            rows.append(np.zeros((1024,), dtype=np.float32))
            continue
        fp = _MORGAN_GENERATOR.GetFingerprint(mol)
        rows.append(np.asarray(fp, dtype=np.float32))
    return np.vstack(rows)


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    spearman = spearmanr(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": float(spearman.statistic if np.isfinite(spearman.statistic) else np.nan),
    }


def build_prediction_frame(
    *,
    task_id: str,
    model_family: str,
    model_variant: str,
    seed: int,
    split_id: str,
    sample_ids: pd.Series,
    input_values: pd.Series,
    true_targets: pd.Series,
    predictions: np.ndarray,
    prediction_type: str,
    threshold: float | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "task_id": task_id,
            "model_family": model_family,
            "model_variant": model_variant,
            "seed": seed,
            "split_id": split_id,
            "sample_id": sample_ids.astype(str).tolist(),
            "input_value": input_values.astype(str).tolist(),
            "true_target": true_targets.tolist(),
            "prediction": predictions.tolist(),
            "prediction_type": prediction_type,
            "threshold": threshold,
        }
    )
    return frame[PREDICTION_COLUMNS]


def build_metric_frame(
    *,
    task_id: str,
    model_family: str,
    model_variant: str,
    seed: int,
    split_id: str,
    metrics: dict[str, float],
    primary_metric_names: set[str],
) -> pd.DataFrame:
    rows = []
    for metric_name, metric_value in metrics.items():
        rows.append(
            {
                "task_id": task_id,
                "model_family": model_family,
                "model_variant": model_variant,
                "seed": seed,
                "split_id": split_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_role": "primary" if metric_name in primary_metric_names else "secondary",
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def write_baseline_outputs(
    *,
    output_dir: Path,
    prediction_frame: pd.DataFrame,
    metric_frame: pd.DataFrame,
    adapter_metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    metric_frame.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "adapter_metadata.json").write_text(json.dumps(adapter_metadata, indent=2), encoding="utf-8")