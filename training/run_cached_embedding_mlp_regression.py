from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from training.adapters.common import build_metric_frame, build_prediction_frame, load_task_frames, write_baseline_outputs
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

MLP_SIZE_DEFAULTS: dict[str, dict[str, float | int]] = {
    "small": {
        "hidden_dim": 128,
        "bottleneck_dim": 32,
        "learning_rate": 3e-4,
        "weight_decay": 5e-4,
        "dropout": 0.10,
        "ranking_weight": 0.0,
    },
    "base": {
        "hidden_dim": 256,
        "bottleneck_dim": 64,
        "learning_rate": 2e-4,
        "weight_decay": 7.5e-4,
        "dropout": 0.15,
        "ranking_weight": 0.0,
    },
    "large": {
        "hidden_dim": 384,
        "bottleneck_dim": 96,
        "learning_rate": 1.5e-4,
        "weight_decay": 1e-3,
        "dropout": 0.20,
        "ranking_weight": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen cached-embedding MLP regression with optional ranking loss.")
    parser.add_argument("--task", default="cycpeptmpdb_perm")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--cache_root", type=Path, default=REPO_ROOT / "tmp" / "embeddings_regression")
    parser.add_argument("--prepared_data_root", type=Path, default=REPO_ROOT / "tmp" / "prepared_data")
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs_cached_embedding_mlp")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--bottleneck_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.5)
    parser.add_argument("--ranking_weight", type=float, default=None)
    parser.add_argument("--rank_pair_count", type=int, default=256)
    parser.add_argument("--rank_target_margin", type=float, default=0.10)
    parser.add_argument("--rank_prediction_margin", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def fit_minmax_scaler(values: pd.Series | np.ndarray) -> tuple[float, float]:
    value_array = np.asarray(values, dtype=np.float32)
    value_min = float(np.min(value_array))
    value_max = float(np.max(value_array))
    return value_min, value_max


def transform_targets(values: pd.Series | np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float32)
    scale = value_max - value_min
    if scale <= 1e-8:
        return np.zeros_like(value_array, dtype=np.float32)
    return ((value_array - value_min) / scale).astype(np.float32)


def inverse_transform_targets(values: np.ndarray, value_min: float, value_max: float) -> np.ndarray:
    value_array = np.asarray(values, dtype=np.float32)
    scale = value_max - value_min
    if scale <= 1e-8:
        return np.full_like(value_array, fill_value=value_min, dtype=np.float32)
    return (value_array * scale + value_min).astype(np.float32)


def resolve_model_scale(model_name: str, embedding_dim: int) -> str:
    normalized_name = model_name.lower()
    if "-small" in normalized_name:
        return "small"
    if "-large" in normalized_name:
        return "large"
    if "-base" in normalized_name:
        return "base"
    if embedding_dim >= 1024:
        return "large"
    if embedding_dim >= 768:
        return "base"
    return "small"


def resolve_mlp_config(model_name: str, embedding_dim: int, args: argparse.Namespace) -> dict[str, float | int | str]:
    scale = resolve_model_scale(model_name, embedding_dim)
    config = dict(MLP_SIZE_DEFAULTS[scale])
    overrides = {
        "hidden_dim": args.hidden_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "ranking_weight": args.ranking_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["scale"] = scale
    return config


def fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def build_rebalanced_sampler(targets: np.ndarray, bin_count: int, seed: int) -> WeightedRandomSampler | None:
    if len(targets) < 2 or bin_count <= 1:
        return None

    target_frame = pd.DataFrame({"target": np.asarray(targets, dtype=np.float32)})
    target_bins = pd.cut(target_frame["target"], bins=bin_count, duplicates="drop")
    if target_bins.isna().all():
        return None

    bin_codes = target_bins.cat.codes.to_numpy()
    valid_mask = bin_codes >= 0
    if valid_mask.sum() <= 1:
        return None

    unique_codes, counts = np.unique(bin_codes[valid_mask], return_counts=True)
    if len(unique_codes) <= 1:
        return None

    count_map = {int(code): int(count) for code, count in zip(unique_codes, counts)}
    sample_weights = np.ones(len(targets), dtype=np.float32)
    for index, code in enumerate(bin_codes):
        if code >= 0:
            sample_weights[index] = 1.0 / float(count_map[int(code)])

    max_count = max(count_map.values())
    num_samples = max_count * len(count_map)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.float32),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


class EmbeddingMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def compute_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pair_count: int,
    target_margin: float,
    prediction_margin: float,
) -> torch.Tensor:
    batch_size = int(predictions.shape[0])
    if batch_size < 2 or pair_count <= 0:
        return predictions.new_zeros(())

    pair_total = min(pair_count, batch_size * (batch_size - 1) // 2)
    index_i = torch.randint(0, batch_size, (pair_total,), device=predictions.device)
    index_j = torch.randint(0, batch_size, (pair_total,), device=predictions.device)
    valid_mask = index_i != index_j
    index_i = index_i[valid_mask]
    index_j = index_j[valid_mask]
    if index_i.numel() == 0:
        return predictions.new_zeros(())

    target_diff = targets[index_i] - targets[index_j]
    informative_mask = target_diff.abs() >= target_margin
    if informative_mask.sum() == 0:
        return predictions.new_zeros(())

    index_i = index_i[informative_mask]
    index_j = index_j[informative_mask]
    target_sign = torch.sign(targets[index_i] - targets[index_j])
    return F.margin_ranking_loss(
        predictions[index_i],
        predictions[index_j],
        target_sign,
        margin=prediction_margin,
        reduction="mean",
    )


def fit_fold_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    mlp_config: dict[str, float | int | str],
    device: torch.device,
) -> nn.Module:
    seed_everything(seed)
    model = EmbeddingMLPRegressor(
        input_dim=int(train_x.shape[1]),
        hidden_dim=int(mlp_config["hidden_dim"]),
        bottleneck_dim=int(mlp_config["bottleneck_dim"]),
        dropout=float(mlp_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(mlp_config["learning_rate"]), weight_decay=float(mlp_config["weight_decay"]))
    if args.loss == "huber":
        regression_loss_fn = nn.HuberLoss(delta=float(args.huber_delta))
    else:
        regression_loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_sampler = None
    if args.rebalance_train_bins:
        train_sampler = build_rebalanced_sampler(train_y, bin_count=int(args.rebalance_bin_count), seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(len(train_dataset), int(args.batch_size)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )

    val_x_tensor = torch.from_numpy(val_x).to(device)
    val_y_tensor = torch.from_numpy(val_y).to(device)
    best_state = None
    best_val_rmse = float("inf")
    stale_epochs = 0

    for _epoch in range(int(args.epochs)):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            regression_loss = regression_loss_fn(predictions, batch_y)
            ranking_loss = compute_ranking_loss(
                predictions=predictions,
                targets=batch_y,
                pair_count=int(args.rank_pair_count),
                target_margin=float(args.rank_target_margin),
                prediction_margin=float(args.rank_prediction_margin),
            )
            loss = regression_loss + float(mlp_config["ranking_weight"]) * ranking_loss
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_x_tensor)
            val_mse = F.mse_loss(val_predictions, val_y_tensor)
            val_rmse = float(torch.sqrt(val_mse + 1e-12).item())

        if val_rmse + 1e-8 < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_tensor(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(device)
        predictions = model(tensor).detach().cpu().numpy().astype(np.float32)
    return predictions


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


def round_robin_ensemble_predictions(
    features: np.ndarray,
    data_frame: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
    mlp_config: dict[str, float | int | str],
    device: torch.device,
    model_label: str,
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
        test_x_full = features[test_mask.to_numpy()]
        fold_predictions = []

        for val_fold in sorted(ensemble_df["fold"].unique().tolist()):
            train_mask = (~test_mask) & (data_frame["fold"] != val_fold)
            val_mask = data_frame["fold"] == val_fold

            train_x_raw = features[train_mask.to_numpy()]
            val_x_raw = features[val_mask.to_numpy()]
            test_x_raw = test_x_full
            train_y_raw = np.asarray(data_frame.loc[train_mask, "value"], dtype=np.float32)
            val_y_raw = np.asarray(data_frame.loc[val_mask, "value"], dtype=np.float32)

            value_min, value_max = fit_minmax_scaler(train_y_raw)
            train_y = transform_targets(train_y_raw, value_min, value_max)
            val_y = transform_targets(val_y_raw, value_min, value_max)

            mean, std = fit_standardizer(train_x_raw)
            train_x = apply_standardizer(train_x_raw, mean, std)
            val_x = apply_standardizer(val_x_raw, mean, std)
            test_x = apply_standardizer(test_x_raw, mean, std)

            regressor = fit_fold_model(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                seed=seed + int(fold_id) + int(val_fold),
                args=args,
                mlp_config=mlp_config,
                device=device,
            )
            current_predictions_scaled = predict_tensor(regressor, test_x, device=device)
            current_predictions = inverse_transform_targets(current_predictions_scaled, value_min, value_max)
            fold_predictions.append(current_predictions)
            prediction_columns[f"prediction_{val_fold}"][test_mask.to_numpy()] = current_predictions

        stacked = np.vstack(fold_predictions)
        predictions[test_mask.to_numpy()] = stacked.mean(axis=0)
        prediction_std[test_mask.to_numpy()] = stacked.std(axis=0)
        print(
            f"[fold-done] model={model_label} seed={seed} test_fold={fold_id} ensemble_members={len(fold_predictions)}",
            flush=True,
        )

    return predictions, prediction_std, prediction_columns, len(fold_ids)


def main() -> int:
    args = parse_args()
    if args.task != "cycpeptmpdb_perm":
        raise ValueError("This pipeline is regression-only and currently supports only cycpeptmpdb_perm.")

    device = resolve_device(args.device)
    data_frame = load_task_frames(args.task, seed=0, prepared_data_root=args.prepared_data_root)["full"]
    data_frame = data_frame.copy().sort_values("fold").reset_index(drop=True)
    models = resolve_models(args)
    seeds = resolve_seeds(args)

    print(
        f"[run-start] task={args.task} device={device} transformer_models={len(models)} seeds={seeds} mlp_hidden={args.hidden_dim} bottleneck={args.bottleneck_dim} ranking_weight={args.ranking_weight}",
        flush=True,
    )

    summary_rows: list[dict[str, object]] = []
    for model_name in models:
        model_variant = model_name.split("/")[-1].replace("-", "_")
        cache_path = args.cache_root / model_variant / args.task / "full.npy"
        print(f"[model-start] transformer model={model_name}", flush=True)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached embeddings not found: {cache_path}")
        features = np.load(cache_path).astype(np.float32)
        mlp_config = resolve_mlp_config(model_name, int(features.shape[1]), args)
        print(
            f"[model-config] transformer model={model_name} feature_dim={features.shape[1]} scale={mlp_config['scale']} mlp={{hidden_dim={mlp_config['hidden_dim']}, bottleneck_dim={mlp_config['bottleneck_dim']}, dropout={mlp_config['dropout']}, lr={mlp_config['learning_rate']}, wd={mlp_config['weight_decay']}, loss={args.loss}, ranking_weight={mlp_config['ranking_weight']}, rebalance_train_bins={args.rebalance_train_bins}, rebalance_bin_count={args.rebalance_bin_count}}}",
            flush=True,
        )

        for seed in seeds:
            run_dir = args.output_root / args.task / "cached_embedding_mlp" / model_variant / f"seed_{seed}"
            if has_completed_run(run_dir):
                summary_rows.append(load_summary_row(run_dir, model_name=model_name, seed=seed))
                print(f"[skip-seed] transformer model={model_name} seed={seed} found completed output", flush=True)
                continue

            print(f"[seed-start] transformer model={model_name} seed={seed}", flush=True)
            predictions, prediction_std, prediction_columns, fold_count = round_robin_ensemble_predictions(
                features=features,
                data_frame=data_frame,
                seed=seed,
                args=args,
                mlp_config=mlp_config,
                device=device,
                model_label=model_variant,
            )

            metrics = compute_regression_metrics_with_mse(
                data_frame["value"].to_numpy(dtype=np.float32),
                predictions,
            )

            payload = {
                "task": args.task,
                "model_name": model_name,
                "seed": seed,
                "status": "ready",
                "feature_source": "cached_transformer_embedding",
                "strategy": "provided_fold_round_robin_ensemble",
                "fold_count": fold_count,
                "mlp_config": {
                    "scale": str(mlp_config["scale"]),
                    "hidden_dim": int(mlp_config["hidden_dim"]),
                    "bottleneck_dim": int(mlp_config["bottleneck_dim"]),
                    "dropout": float(mlp_config["dropout"]),
                    "learning_rate": float(mlp_config["learning_rate"]),
                    "weight_decay": float(mlp_config["weight_decay"]),
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
                    "batch_size": int(args.batch_size),
                    "loss": str(args.loss),
                    "huber_delta": float(args.huber_delta),
                    "ranking_weight": float(mlp_config["ranking_weight"]),
                    "rank_pair_count": int(args.rank_pair_count),
                    "rank_target_margin": float(args.rank_target_margin),
                    "rank_prediction_margin": float(args.rank_prediction_margin),
                    "grad_clip_norm": float(args.grad_clip_norm),
                    "rebalance_train_bins": bool(args.rebalance_train_bins),
                    "rebalance_bin_count": int(args.rebalance_bin_count),
                },
            }

            prediction_frame = build_prediction_frame(
                task_id=args.task,
                model_family="cached_embedding_mlp",
                model_variant=f"mlp-{model_variant}",
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
                model_family="cached_embedding_mlp",
                model_variant=f"mlp-{model_variant}",
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
            (run_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            summary = {"model_name": model_name, "seed": seed, "run_dir": str(run_dir)}
            summary.update(metrics)
            summary_rows.append(summary)
            print(
                f"[seed-done] transformer model={model_name} seed={seed} r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f}",
                flush=True,
            )

    summary_frame = pd.DataFrame(summary_rows).sort_values(["model_name", "seed"]).reset_index(drop=True)
    summary_path = args.output_root / args.task / "cached_embedding_mlp" / "summary_metrics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_path, index=False)
    print(f"[run-done] wrote summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())