from __future__ import annotations

from pathlib import Path
import math
import re

import pandas as pd
from scipy.stats import t


RESULTS_ROOT = Path(__file__).resolve().parent
METRICS_OUTPUT_PATH = RESULTS_ROOT / "results_metrics_summary.csv"
OUTPUT_PATH = RESULTS_ROOT / "results_category_summary.csv"


def compute_mean_ci95(values: pd.Series) -> tuple[float, float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    mean_value = float(numeric.mean())
    if len(numeric) < 2:
        return mean_value, None, None

    sample_std = float(numeric.std(ddof=1))
    if sample_std == 0.0:
        return mean_value, mean_value, mean_value

    margin = float(t.ppf(0.975, df=len(numeric) - 1) * sample_std / math.sqrt(len(numeric)))
    return mean_value, mean_value - margin, mean_value + margin


def compute_run_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    valid = frame.dropna(subset=["value", "mean_prediction"]).copy()
    y_true = pd.to_numeric(valid["value"], errors="coerce")
    y_pred = pd.to_numeric(valid["mean_prediction"], errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true.loc[mask]
    y_pred = y_pred.loc[mask]
    dropped_row_count = int(len(frame) - len(y_true))

    if y_true.empty:
        raise ValueError("No valid rows after coercing value and mean_prediction to numeric.")

    residual = y_true - y_pred
    mse = float((residual**2).mean())
    rmse = float(math.sqrt(mse))
    mae = float(residual.abs().mean())
    total_var = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1.0 - ((residual**2).sum() / total_var)) if total_var != 0.0 else float("nan")
    pearson = float(y_true.corr(y_pred, method="pearson"))
    spearman = float(y_true.corr(y_pred, method="spearman"))
    return {
        "row_count": int(len(frame)),
        "valid_row_count": int(len(y_true)),
        "dropped_row_count": dropped_row_count,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
    }


def build_metrics_summary(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(results_root.glob("*/*.csv")):
        if csv_path.name.startswith("."):
            continue

        relative_path = csv_path.relative_to(results_root).as_posix()
        row: dict[str, object] = {
            "relative_path": relative_path,
            "parent_dir": csv_path.parent.relative_to(results_root).as_posix(),
            "file_name": csv_path.name,
        }

        try:
            frame = pd.read_csv(csv_path)
            required_columns = {"value", "mean_prediction"}
            missing_columns = sorted(required_columns.difference(frame.columns))
            if missing_columns:
                row.update(
                    {
                        "status": "missing_columns",
                        "row_count": int(len(frame)),
                        "valid_row_count": None,
                        "dropped_row_count": None,
                        "r2": None,
                        "mse": None,
                        "rmse": None,
                        "mae": None,
                        "pearson": None,
                        "spearman": None,
                        "error_message": ", ".join(missing_columns),
                    }
                )
            else:
                metrics = compute_run_metrics(frame)
                row.update(metrics)
                row.update({"status": "ok", "error_message": None})
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "row_count": None,
                    "valid_row_count": None,
                    "dropped_row_count": None,
                    "r2": None,
                    "mse": None,
                    "rmse": None,
                    "mae": None,
                    "pearson": None,
                    "spearman": None,
                    "error_message": str(exc),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def parse_category_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    path_parts = working["relative_path"].astype(str).str.split("/")
    working["result_family"] = path_parts.str[0]
    working["experiment_name"] = working["file_name"].astype(str).str.replace(r"\.csv$", "", regex=True)
    working["model_bucket"] = working["experiment_name"].map(extract_model_bucket)
    working["parent_dir"] = working["relative_path"].astype(str).str.rsplit("/", n=1).str[0]
    working["model_size"] = working["model_bucket"].astype(str).str.extract(r"(small|medium|large|base)", expand=False)
    working["experiment_group"] = working["experiment_name"].map(normalize_experiment_group)
    return working


def extract_model_bucket(experiment_name: str) -> str:
    normalized = str(experiment_name)
    normalized = re.sub(r"_(study_\d+|round\d+)$", "", normalized)
    return normalized


def normalize_experiment_group(experiment_name: str) -> str:
    normalized = str(experiment_name)
    normalized = re.sub(r"_study_\d+$", "_study", normalized)
    normalized = re.sub(r"_round\d+$", "", normalized)
    return normalized


def build_category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame.loc[frame["status"] == "ok"].copy()
    ok = parse_category_columns(ok)

    rows: list[dict[str, object]] = []
    group_columns = ["result_family", "model_bucket", "model_size", "parent_dir", "experiment_group"]
    for group_key, group_frame in ok.groupby(group_columns, dropna=False, sort=True):
        best_rmse_row = group_frame.loc[group_frame["rmse"].idxmin()]
        best_r2_row = group_frame.loc[group_frame["r2"].idxmax()]
        mean_rmse, rmse_ci95_low, rmse_ci95_high = compute_mean_ci95(group_frame["rmse"])
        mean_r2, r2_ci95_low, r2_ci95_high = compute_mean_ci95(group_frame["r2"])
        mean_mae, mae_ci95_low, mae_ci95_high = compute_mean_ci95(group_frame["mae"])
        mean_spearman, spearman_ci95_low, spearman_ci95_high = compute_mean_ci95(group_frame["spearman"])
        rows.append(
            {
                "result_family": group_key[0],
                "model_bucket": group_key[1],
                "model_size": group_key[2],
                "parent_dir": group_key[3],
                "experiment_group": group_key[4],
                "run_count": int(len(group_frame)),
                "mean_rmse": mean_rmse,
                "rmse_ci95_low": rmse_ci95_low,
                "rmse_ci95_high": rmse_ci95_high,
                "median_rmse": float(group_frame["rmse"].median()),
                "mean_r2": mean_r2,
                "r2_ci95_low": r2_ci95_low,
                "r2_ci95_high": r2_ci95_high,
                "median_r2": float(group_frame["r2"].median()),
                "mean_mae": mean_mae,
                "mae_ci95_low": mae_ci95_low,
                "mae_ci95_high": mae_ci95_high,
                "mean_spearman": mean_spearman,
                "spearman_ci95_low": spearman_ci95_low,
                "spearman_ci95_high": spearman_ci95_high,
                "best_rmse": float(best_rmse_row["rmse"]),
                "best_rmse_file": best_rmse_row["relative_path"],
                "best_rmse_experiment": best_rmse_row["experiment_name"],
                "best_r2": float(best_r2_row["r2"]),
                "best_r2_file": best_r2_row["relative_path"],
                "best_r2_experiment": best_r2_row["experiment_name"],
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["result_family", "model_bucket", "experiment_group", "best_rmse"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return summary


def main() -> int:
    frame = build_metrics_summary(RESULTS_ROOT)
    frame.to_csv(METRICS_OUTPUT_PATH, index=False)
    summary = build_category_summary(frame)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {METRICS_OUTPUT_PATH} with {len(frame)} run rows.")
    print(f"Wrote {OUTPUT_PATH} with {len(summary)} grouped rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())