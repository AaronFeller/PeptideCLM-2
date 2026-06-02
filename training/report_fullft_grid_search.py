from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MINIMIZE_METRICS = {"rmse", "mae", "mse"}
MAXIMIZE_METRICS = {"r2", "spearman"}
METRIC_COLUMNS = ["rmse", "r2", "mae", "mse", "spearman"]
BEST_CONFIG_COLUMNS = [
    "model_scale",
    "config_name",
    "learning_rate",
    "weight_decay",
    "head_dropout",
    "warmup_fraction",
    "max_steps",
    "patience",
    "batch_size",
    "eval_batch_size",
    "accumulate_grad_batches",
    "parallel_val_folds",
    "test_folds",
    "rmse",
    "r2",
    "mae",
    "mse",
    "spearman",
    "model_count",
    "seed_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate full-finetune grid-search results and emit best-config reports.")
    parser.add_argument("--sweep_root", type=Path, required=True)
    parser.add_argument("--selection_metric", choices=sorted(MINIMIZE_METRICS | MAXIMIZE_METRICS), default="rmse")
    parser.add_argument("--report_dir", type=Path, default=None)
    return parser.parse_args()


def load_stage_rows(stage_root: Path, stage_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not stage_root.exists():
        return pd.DataFrame()

    for summary_path in sorted(stage_root.rglob("summary_metrics.csv")):
        try:
            frame = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame = frame.copy()
        frame["stage"] = stage_name
        frame["summary_path"] = str(summary_path)
        rows.append(frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    if "config_name" not in working.columns:
        working["config_name"] = "default"
    working["config_name"] = working["config_name"].fillna("default")

    metric_aggs = {metric_name: (metric_name, "mean") for metric_name in METRIC_COLUMNS if metric_name in working.columns}
    metadata_aggs = {
        "max_steps": ("max_steps", "first"),
        "patience": ("patience", "first"),
        "learning_rate": "first",
        "weight_decay": "first",
        "head_dropout": "first",
        "warmup_fraction": "first",
        "batch_size": "first",
        "eval_batch_size": "first",
        "accumulate_grad_batches": "first",
        "parallel_val_folds": "first",
        "test_folds": "first",
    }
    available_metadata = {}
    for key, value in metadata_aggs.items():
        if key not in working.columns:
            continue
        if isinstance(value, tuple):
            available_metadata[key] = value
        else:
            available_metadata[key] = (key, value)
    grouped = (
        working.groupby(["model_scale", "config_name"], dropna=False)
        .agg(
            model_count=("model_name", "nunique"),
            seed_count=("seed", "nunique"),
            **metric_aggs,
            **available_metadata,
        )
        .reset_index()
    )
    return grouped


def rank_candidates(frame: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    ascending = selection_metric in MINIMIZE_METRICS
    ranked_groups = []
    for _, scale_frame in frame.groupby("model_scale", sort=True):
        ranked = scale_frame.sort_values(selection_metric, ascending=ascending).reset_index(drop=True)
        ranked.insert(len(ranked.columns), "rank", range(1, len(ranked) + 1))
        ranked_groups.append(ranked)
    return pd.concat(ranked_groups, ignore_index=True)


def select_best_configs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=BEST_CONFIG_COLUMNS + ["rank"])
    best = frame.loc[frame.groupby("model_scale")["rank"].idxmin()].copy()
    return best.sort_values("model_scale").reset_index(drop=True)


def select_report_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].copy()


def build_markdown_report(
    *,
    selection_metric: str,
    proxy_ranked: pd.DataFrame,
    best_configs: pd.DataFrame,
    final_rows: pd.DataFrame,
    final_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Full-FT Grid Search Report",
        "",
        f"Selection metric: {selection_metric}",
        "",
        "## Best Configs",
        "",
    ]
    if best_configs.empty:
        lines.append("No proxy sweep results were found.")
    else:
        lines.extend([
            "| Scale | Config | RMSE | R2 | MAE | Spearman | LR | WD | Dropout | Warmup | Max Steps | Patience |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in best_configs.to_dict(orient="records"):
            lines.append(
                "| {model_scale} | {config_name} | {rmse:.4f} | {r2:.4f} | {mae:.4f} | {spearman:.4f} | {learning_rate:.2e} | {weight_decay:.2e} | {head_dropout:.2f} | {warmup_fraction:.2f} | {max_steps} | {patience} |".format(
                    **row
                )
            )

    lines.extend(["", "## Proxy Sweep Rankings", ""])
    if proxy_ranked.empty:
        lines.append("No proxy sweep rows were found.")
    else:
        lines.extend([
            "| Scale | Rank | Config | RMSE | R2 | MAE | Spearman | Models | Seeds | Proxy Holdouts |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in proxy_ranked.to_dict(orient="records"):
            lines.append(
                "| {model_scale} | {rank} | {config_name} | {rmse:.4f} | {r2:.4f} | {mae:.4f} | {spearman:.4f} | {model_count} | {seed_count} | {test_folds} |".format(
                    **row
                )
            )

    lines.extend(["", "## Final All-Holdout Results", ""])
    if final_rows.empty:
        lines.append("Final all-holdout rerun results were not found yet.")
    else:
        lines.extend([
            "| Scale | Config | Model | Seed | RMSE | R2 | MAE | Spearman |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in final_rows.sort_values(["model_scale", "model_name", "seed"]).to_dict(orient="records"):
            lines.append(
                "| {model_scale} | {config_name} | {model_name} | {seed} | {rmse:.4f} | {r2:.4f} | {mae:.4f} | {spearman:.4f} |".format(
                    **row
                )
            )
        if not final_summary.empty:
            lines.extend(["", "### Final Scale Summary", ""])
            lines.extend([
                "| Scale | Config | RMSE | R2 | MAE | Spearman | Models | Seeds |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for row in final_summary.to_dict(orient="records"):
                lines.append(
                    "| {model_scale} | {config_name} | {rmse:.4f} | {r2:.4f} | {mae:.4f} | {spearman:.4f} | {model_count} | {seed_count} |".format(
                        **row
                    )
                )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir or args.sweep_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    proxy_rows = load_stage_rows(args.sweep_root / "proxy", "proxy")
    final_rows = load_stage_rows(args.sweep_root / "final", "final")

    proxy_candidates = aggregate_candidates(proxy_rows)
    proxy_ranked = rank_candidates(proxy_candidates, args.selection_metric)
    best_configs = select_best_configs(proxy_ranked)

    final_summary = aggregate_candidates(final_rows)
    if not final_summary.empty:
        final_summary = final_summary.sort_values(["model_scale", "config_name"]).reset_index(drop=True)

    proxy_rows.to_csv(report_dir / "proxy_raw_rows.csv", index=False)
    proxy_ranked.to_csv(report_dir / "proxy_sweep_results.csv", index=False)
    select_report_columns(best_configs, BEST_CONFIG_COLUMNS).to_csv(report_dir / "best_configs.csv", index=False)
    select_report_columns(best_configs, BEST_CONFIG_COLUMNS).to_csv(report_dir / "best_configs.tsv", index=False, sep="\t")
    (report_dir / "best_configs.json").write_text(
        json.dumps(best_configs.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    final_rows.to_csv(report_dir / "final_all_holdout_results.csv", index=False)
    final_summary.to_csv(report_dir / "final_scale_summary.csv", index=False)

    markdown_report = build_markdown_report(
        selection_metric=args.selection_metric,
        proxy_ranked=proxy_ranked,
        best_configs=best_configs,
        final_rows=final_rows,
        final_summary=final_summary,
    )
    (report_dir / "fullft_grid_search_report.md").write_text(markdown_report, encoding="utf-8")
    print(f"[report] wrote grid-search report bundle to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())