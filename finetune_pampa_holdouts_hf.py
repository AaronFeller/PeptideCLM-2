from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = REPO_ROOT / "data" / "PAMPA_clusters.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "runs_hf_pampa_holdouts"
MODEL_TYPES = ("mlm", "mtr", "hybrid")
MODEL_SIZES = ("small", "base", "large")
DEFAULT_MODELS = [
    f"aaronfeller/peptideclm-2-{model_type}-{model_size}"
    for model_size in MODEL_SIZES
    for model_type in MODEL_TYPES
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def normalize_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "value" not in normalized.columns:
        if "PAMPA" not in normalized.columns:
            raise ValueError("Expected a value or PAMPA column in the input CSV.")
        normalized = normalized.rename(columns={"PAMPA": "value"})

    if "SMILES" not in normalized.columns:
        if "smiles" in normalized.columns:
            normalized = normalized.rename(columns={"smiles": "SMILES"})
        else:
            raise ValueError("Expected a SMILES or smiles column in the input CSV.")

    if "fold" not in normalized.columns:
        if "cluster" not in normalized.columns:
            raise ValueError("Expected a fold or cluster column in the input CSV.")
        unique_clusters = sorted(normalized["cluster"].dropna().unique().tolist())
        cluster_to_fold = {cluster_id: fold_id for fold_id, cluster_id in enumerate(unique_clusters)}
        normalized["fold"] = normalized["cluster"].map(cluster_to_fold)

    keep_columns = [column for column in ["SMILES", "value", "cluster", "fold"] if column in normalized.columns]
    normalized = normalized[keep_columns].copy()
    normalized["value"] = normalized["value"].astype(np.float32)
    normalized["fold"] = normalized["fold"].astype(int)
    if "cluster" in normalized.columns:
        normalized["cluster"] = normalized["cluster"].astype(int)
    return normalized.sort_values(["fold", "SMILES"]).reset_index(drop=True)


def resolve_learning_rate(model_name: str) -> float:
    normalized_name = model_name.lower()
    if "-small" in normalized_name:
        return 3e-4
    if "-large" in normalized_name:
        return 5e-5
    return 1e-4


def resolve_embed_dim(model) -> int:
    for attr in ("hidden_size", "embed_dim", "d_model"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer embedding dimension from model config.")


def resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer is missing pad_token_id; this script expects the pretrained tokenizer to define one.")
    return int(tokenizer.pad_token_id)


def build_validation_strata(values: pd.Series, bin_count: int) -> pd.Series | None:
    usable_bins = min(int(bin_count), max(1, values.nunique()))
    if usable_bins <= 1:
        return None
    try:
        strata = pd.qcut(values, q=usable_bins, duplicates="drop")
    except ValueError:
        return None
    counts = strata.value_counts(dropna=False)
    if counts.empty or counts.min() < 2:
        return None
    return strata.astype(str)


def split_train_val_randomly(
    train_pool_df: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
    stratify_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_pool_df) < 2:
        raise ValueError("Need at least two rows outside the held-out fold to create a validation split.")
    val_size = max(1, int(round(len(train_pool_df) * val_fraction)))
    val_size = min(val_size, len(train_pool_df) - 1)
    strata = build_validation_strata(train_pool_df["value"], stratify_bins)
    stratify_values = strata if strata is not None else None
    train_df, val_df = train_test_split(
        train_pool_df,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_values,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def oversample_by_value_bins(train_df: pd.DataFrame, bin_count: int, seed: int) -> pd.DataFrame:
    if bin_count <= 1 or len(train_df) < 2:
        return train_df.copy().reset_index(drop=True)
    usable_bins = min(int(bin_count), max(1, train_df["value"].nunique()))
    if usable_bins <= 1:
        return train_df.copy().reset_index(drop=True)
    binned = pd.cut(train_df["value"], bins=usable_bins, duplicates="drop")
    if binned.isna().all():
        return train_df.copy().reset_index(drop=True)
    working = train_df.copy()
    working["_value_bin"] = binned
    group_sizes = working.groupby("_value_bin", observed=False).size()
    target_size = int(group_sizes.max())
    sampled_groups = []
    for _, group in working.groupby("_value_bin", observed=False):
        if group.empty:
            continue
        sampled_groups.append(group.sample(n=target_size, replace=True, random_state=seed))
    balanced = pd.concat(sampled_groups, ignore_index=True)
    balanced = balanced.drop(columns=["_value_bin"])
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def extract_sequence_output(outputs) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise ValueError("Model output does not contain last_hidden_state.")


def extract_mean_pool(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, "mean_pool"):
        return outputs.mean_pool
    if isinstance(outputs, dict) and "mean_pool" in outputs:
        return outputs["mean_pool"]
    sequence_output = extract_sequence_output(outputs)
    expanded_mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
    summed = (sequence_output * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
            "labels": labels,
        }

    return collate_fn


class MoleculeRegressionDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length: int = 2048):
        encodings = tokenizer(
            list(smiles_list),
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=True,
        )
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encodings["input_ids"]]
        self.attention_masks = [torch.ones_like(ids, dtype=torch.long) for ids in self.input_ids]
        self.labels = torch.tensor(list(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class RegressionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        head_dropout: float,
        transfer_learning: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
        embed_dim = resolve_embed_dim(self.model)
        self.dropout = nn.Dropout(head_dropout)
        self.intermediate_layer = nn.Linear(embed_dim, embed_dim)
        self.regression_head = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.intermediate_layer.weight)
        nn.init.zeros_(self.intermediate_layer.bias)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)
        self.criterion = nn.MSELoss()

        if transfer_learning:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = extract_mean_pool(outputs, attention_mask)
        pooled = self.dropout(pooled)
        hidden = self.intermediate_layer(pooled)
        hidden = self.dropout(hidden)
        return self.regression_head(hidden).squeeze(-1)

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        predictions = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(predictions, batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        predictions = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(predictions, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def predict_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self(batch["input_ids"], batch["attention_mask"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)


class InMemoryBestModelCallback(Callback):
    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_state_dict = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        score = float(metric.detach().cpu().item())
        if self.best_score is None:
            improved = True
        elif self.mode == "min":
            improved = score < self.best_score
        else:
            improved = score > self.best_score
        if not improved:
            return
        self.best_score = score
        self.best_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in pl_module.state_dict().items()
        }

    def restore_best_weights(self, pl_module) -> None:
        if self.best_state_dict is None:
            return
        device = pl_module.device
        restored = {name: tensor.to(device=device) for name, tensor in self.best_state_dict.items()}
        pl_module.load_state_dict(restored)


def predict_from_batches(prediction_batches: list[torch.Tensor]) -> np.ndarray:
    flattened = [batch.detach().float().cpu().view(-1).numpy() for batch in prediction_batches]
    if not flattened:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(flattened).astype(np.float32)


def compute_r2_from_predictions(frame: pd.DataFrame) -> float:
    valid = frame[["value", "prediction"]].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty or len(valid) < 2:
        return float("nan")
    return float(r2_score(valid["value"], valid["prediction"]))


def build_trainer(*, gpu_index: int, max_epochs: int, patience: int, val_check_interval: float):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [int(gpu_index)] if accelerator == "gpu" else 1
    precision = "bf16-mixed" if accelerator == "gpu" else "32-true"
    best_model_callback = InMemoryBestModelCallback(monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[
            best_model_callback,
            EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ],
        log_every_n_steps=100,
        val_check_interval=val_check_interval,
        enable_progress_bar=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
    )
    return trainer, best_model_callback


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        selected_models = list(args.models)
    elif args.model:
        selected_models = [args.model]
    else:
        selected_models = list(DEFAULT_MODELS)
    if args.model_type is not None:
        type_token = f"-{args.model_type}-"
        selected_models = [model_name for model_name in selected_models if type_token in model_name]
    if not selected_models:
        raise ValueError("No models selected for the requested arguments.")
    return selected_models


def resolve_test_folds(data_frame: pd.DataFrame, requested_folds: list[int] | None) -> list[int]:
    available_folds = sorted(data_frame["fold"].unique().tolist())
    if not requested_folds:
        return available_folds
    normalized_folds = sorted({int(fold_id) for fold_id in requested_folds})
    missing_folds = [fold_id for fold_id in normalized_folds if fold_id not in available_folds]
    if missing_folds:
        raise ValueError(
            f"Requested test_folds {missing_folds} are not available in the input data; available folds={available_folds}"
        )
    return normalized_folds


def build_run_dir(output_root: Path, model_name: str) -> Path:
    model_variant = model_name.split("/")[-1].replace(".", "_")
    return output_root / model_variant


def run_model(args: argparse.Namespace, model_name: str, data_frame: pd.DataFrame) -> dict[str, object]:
    run_dir = build_run_dir(args.output_root, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    learning_rate = resolve_learning_rate(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = args.max_length
    pad_token_id = resolve_pad_token_id(tokenizer)
    collate_fn = make_collate_fn(pad_token_id)
    test_folds = resolve_test_folds(data_frame, args.test_folds)
    holdout_results = []
    holdout_metrics = []

    print(
        f"[run] model={model_name} holdouts={test_folds} lr={learning_rate} batch_size={args.batch_size} "
        f"transfer_learning={args.transfer_learning}",
        flush=True,
    )

    for test_fold in test_folds:
        holdout_seed = int(args.seed) * 1000 + int(test_fold)
        seed_everything(holdout_seed)

        test_df = data_frame.loc[data_frame["fold"] == test_fold].copy().reset_index(drop=True)
        train_pool_df = data_frame.loc[data_frame["fold"] != test_fold].copy().reset_index(drop=True)
        train_df, val_df = split_train_val_randomly(
            train_pool_df,
            val_fraction=args.val_fraction,
            seed=holdout_seed,
            stratify_bins=args.val_stratify_bins,
        )
        train_df = oversample_by_value_bins(train_df, args.oversample_train_bins, holdout_seed)

        print(
            f"[holdout] model={model_name} test_fold={test_fold} train_n={len(train_df)} val_n={len(val_df)} test_n={len(test_df)}",
            flush=True,
        )

        train_ds = MoleculeRegressionDataset(train_df["SMILES"], train_df["value"], tokenizer, args.max_length)
        val_ds = MoleculeRegressionDataset(val_df["SMILES"], val_df["value"], tokenizer, args.max_length)
        test_ds = MoleculeRegressionDataset(test_df["SMILES"], test_df["value"], tokenizer, args.max_length)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = RegressionModel(
            model_name=model_name,
            learning_rate=learning_rate,
            head_dropout=args.head_dropout,
            transfer_learning=args.transfer_learning,
        )
        trainer, best_model_callback = build_trainer(
            gpu_index=args.gpu_index,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_check_interval=args.val_check_interval,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_model_callback.restore_best_weights(model)
        prediction_batches = trainer.predict(model, dataloaders=test_loader)
        predictions = predict_from_batches(prediction_batches)

        holdout_frame = test_df.copy()
        holdout_frame["prediction"] = predictions
        holdout_frame["test_fold"] = int(test_fold)
        holdout_frame["holdout_seed"] = int(holdout_seed)
        holdout_results.append(holdout_frame)

        mse = float(np.mean(np.square(holdout_frame["prediction"] - holdout_frame["value"])))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(holdout_frame["prediction"] - holdout_frame["value"])))
        holdout_metrics.append(
            {
                "model_name": model_name,
                "test_fold": int(test_fold),
                "holdout_seed": int(holdout_seed),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_frame = pd.concat(holdout_results, ignore_index=True)
    metrics_frame = pd.DataFrame(holdout_metrics)
    prediction_frame.to_csv(run_dir / "holdout_predictions.csv", index=False)
    metrics_frame.to_csv(run_dir / "holdout_metrics.csv", index=False)
    summary_r2 = compute_r2_from_predictions(prediction_frame)

    summary_row = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "r2": summary_r2,
        "mean_mse": float(metrics_frame["mse"].mean()),
        "mean_rmse": float(metrics_frame["rmse"].mean()),
        "mean_mae": float(metrics_frame["mae"].mean()),
        "output_dir": str(run_dir),
    }
    return summary_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Hugging Face PeptideCLM-2 models on clustered PAMPA holdouts without saving checkpoints."
    )
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_type", choices=MODEL_TYPES, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--val_stratify_bins", type=int, default=5)
    parser.add_argument("--oversample_train_bins", type=int, default=5)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_folds", nargs="+", type=int, default=None)
    parser.add_argument("--transfer_learning", action="store_true")
    args = parser.parse_args()
    if not 0.0 < float(args.val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    return args


def main() -> int:
    args = parse_args()
    if not args.data_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data_csv}")

    models = resolve_models(args)
    data_frame = normalize_input_frame(pd.read_csv(args.data_csv))
    summary_rows = []

    for model_name in models:
        summary_rows.append(run_model(args, model_name, data_frame))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary_rows).sort_values("model_name").reset_index(drop=True)
    summary_frame.to_csv(args.output_root / "summary_metrics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())