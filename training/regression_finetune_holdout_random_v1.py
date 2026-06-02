from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from training.adapters.common import build_metric_frame, build_prediction_frame, write_baseline_outputs
from training.experiment.manifest import REPO_ROOT


torch.set_float32_matmul_precision("high")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
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
MODEL_TYPES = ("mlm", "mtr", "hybrid")
DEFAULT_SEEDS = [101, 202, 303]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def normalize_perm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "value" not in normalized.columns:
        if "PAMPA" not in normalized.columns:
            raise ValueError("Expected a value or PAMPA column in the regression input table.")
        normalized = normalized.rename(columns={"PAMPA": "value"})
    if "fold" not in normalized.columns:
        if "cluster" not in normalized.columns:
            raise ValueError("Expected a fold or cluster column in the regression input table.")
        unique_clusters = sorted(normalized["cluster"].dropna().unique().tolist())
        cluster_to_fold = {cluster_id: fold_id for fold_id, cluster_id in enumerate(unique_clusters)}
        normalized["fold"] = normalized["cluster"].map(cluster_to_fold)
    if "SMILES" not in normalized.columns:
        if "smiles" in normalized.columns:
            normalized = normalized.rename(columns={"smiles": "SMILES"})
        else:
            raise ValueError("Expected a SMILES column in the regression input table.")
    columns = [column for column in ["SMILES", "value", "fold", "cluster"] if column in normalized.columns]
    normalized = normalized[columns].copy()
    normalized["value"] = normalized["value"].astype(np.float32)
    normalized["fold"] = normalized["fold"].astype(int)
    return normalized.sort_values(["fold", "SMILES"]).reset_index(drop=True)


def compute_regression_metrics_with_mse(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        spearman = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": mse,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearman.statistic if np.isfinite(spearman.statistic) else np.nan),
    }


def resolve_model_scale(model_name: str) -> str:
    normalized_name = model_name.lower()
    if "-small" in normalized_name:
        return "small"
    if "-large" in normalized_name:
        return "large"
    return "base"


def resolve_learning_rate(model_name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    scale = resolve_model_scale(model_name)
    if scale == "small":
        return 3e-4
    if scale == "large":
        return 5e-5
    return 1e-4


def resolve_batch_size(model_name: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    scale = resolve_model_scale(model_name)
    if scale == "small":
        return 16
    if scale == "large":
        return 4
    return 8


def resolve_eval_batch_size(model_name: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    scale = resolve_model_scale(model_name)
    if scale == "small":
        return 128
    if scale == "large":
        return 32
    return 64


def resolve_accumulate_grad_batches(model_name: str, override: int | None) -> int:
    if override is not None:
        return max(1, int(override))
    scale = resolve_model_scale(model_name)
    if scale == "small":
        return 1
    if scale == "large":
        return 4
    return 2


def resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return int(tokenizer.pad_token_id)
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return int(tokenizer.pad_token_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return int(tokenizer.pad_token_id)


def resolve_embed_dim(model) -> int:
    for attr in ("hidden_size", "embed_dim", "d_model"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer embedding dimension from model config.")


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


class OldStyleRegressionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        weight_decay: float,
        head_dropout: float,
        random_init_backbone: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if random_init_backbone:
            initialize_backbone_like_old_script(self.model)
        embed_dim = resolve_embed_dim(self.model)
        self.dropout = nn.Dropout(head_dropout)
        self.intermediate_layer = nn.Linear(embed_dim, embed_dim)
        self.regression_head = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.intermediate_layer.weight)
        nn.init.zeros_(self.intermediate_layer.bias)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)
        self.criterion = nn.MSELoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = forward_backbone(self.model, input_ids=input_ids, attention_mask=attention_mask)
        pooled = extract_mean_pool(outputs)
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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


def initialize_backbone_like_old_script(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.data.ndimension() > 1:
            nn.init.kaiming_uniform_(parameter)
        else:
            nn.init.zeros_(parameter)


def build_trainer(
    *,
    log_dir: Path,
    run_name: str,
    gpu_index: int,
    max_epochs: int,
    patience: int,
    val_check_interval: float,
    accumulate_grad_batches: int,
):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=1e-4,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [int(gpu_index)] if accelerator == "gpu" else 1
    precision = "bf16-mixed" if accelerator == "gpu" else "32-true"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=CSVLogger(str(log_dir), name=run_name),
        log_every_n_steps=100,
        val_check_interval=val_check_interval,
        enable_progress_bar=False,
        deterministic=True,
        gradient_clip_val=0.1,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    return trainer, checkpoint_callback


def predict_from_batches(prediction_batches: list[torch.Tensor]) -> np.ndarray:
    flattened = [batch.detach().float().cpu().view(-1).numpy() for batch in prediction_batches]
    if not flattened:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(flattened).astype(np.float32)


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


def build_run_dir(output_root: Path, model_name: str, seed: int) -> Path:
    model_variant = model_name.split("/")[-1].replace(".", "_")
    return output_root / "cycpeptmpdb_perm" / "peptideclm_holdout_random_v1" / model_variant / f"seed_{seed}"


def run_single_experiment(args: argparse.Namespace, model_name: str, seed: int, data_frame: pd.DataFrame):
    run_dir = build_run_dir(args.output_root, model_name, seed)
    log_dir = build_run_dir(args.log_root, model_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_scale = resolve_model_scale(model_name)
    learning_rate = resolve_learning_rate(model_name, args.learning_rate)
    batch_size = resolve_batch_size(model_name, args.batch_size)
    eval_batch_size = resolve_eval_batch_size(model_name, args.eval_batch_size)
    accumulate_grad_batches = resolve_accumulate_grad_batches(model_name, args.accumulate_grad_batches)
    fold_ids = resolve_test_folds(data_frame, args.test_folds)

    print(
        f"[holdout-random] model={model_name} seed={seed} scale={model_scale} "
        f"holdout_ids={fold_ids} val_fraction={args.val_fraction:.2f} "
        f"oversample_train_bins={args.oversample_train_bins} batch_size={batch_size} "
        f"accumulate_grad_batches={accumulate_grad_batches} effective_batch_size={batch_size * accumulate_grad_batches} "
        f"random_init_backbone={args.random_init_backbone}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = args.max_length
    pad_token_id = resolve_pad_token_id(tokenizer)
    collate_fn = make_collate_fn(pad_token_id)
    fold_results = []
    holdout_rows = []

    for test_fold in fold_ids:
        holdout_seed = int(seed) * 1000 + int(test_fold)
        seed_everything(holdout_seed)
        test_df = data_frame.loc[data_frame["fold"] == test_fold].copy().reset_index(drop=True)
        train_pool_df = data_frame.loc[data_frame["fold"] != test_fold].copy().reset_index(drop=True)
        train_df, val_df = split_train_val_randomly(
            train_pool_df,
            val_fraction=args.val_fraction,
            seed=holdout_seed,
            stratify_bins=args.val_stratify_bins,
        )
        if args.oversample_train_bins > 1:
            train_df = oversample_by_value_bins(train_df, args.oversample_train_bins, holdout_seed)

        print(
            f"[holdout] test_fold={test_fold} holdout_seed={holdout_seed} train_n={len(train_df)} "
            f"val_n={len(val_df)} test_n={len(test_df)}",
            flush=True,
        )

        train_ds = MoleculeRegressionDataset(train_df["SMILES"], train_df["value"], tokenizer, max_length=args.max_length)
        val_ds = MoleculeRegressionDataset(val_df["SMILES"], val_df["value"], tokenizer, max_length=args.max_length)
        test_ds = MoleculeRegressionDataset(test_df["SMILES"], test_df["value"], tokenizer, max_length=args.max_length)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = OldStyleRegressionModel(
            model_name=model_name,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            head_dropout=args.head_dropout,
            random_init_backbone=args.random_init_backbone,
        )
        trainer, checkpoint_callback = build_trainer(
            log_dir=log_dir,
            run_name=f"holdout_{test_fold}",
            gpu_index=args.gpu_index,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_check_interval=args.val_check_interval,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        prediction_batches = trainer.predict(model, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
        fold_predictions = predict_from_batches(prediction_batches)

        test_result = test_df.copy()
        test_result["mean_prediction"] = fold_predictions
        test_result["std_prediction"] = 0.0
        test_result["test_fold"] = int(test_fold)
        test_result["holdout_seed"] = int(holdout_seed)
        fold_results.append(test_result)

        holdout_metrics = compute_regression_metrics_with_mse(
            test_result["value"].to_numpy(dtype=np.float32),
            test_result["mean_prediction"].to_numpy(dtype=np.float32),
        )
        holdout_rows.append({
            "model_name": model_name,
            "model_scale": model_scale,
            "seed": seed,
            "test_fold": int(test_fold),
            "holdout_seed": int(holdout_seed),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            **holdout_metrics,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_results = pd.concat(fold_results, ignore_index=True)
    metrics = compute_regression_metrics_with_mse(
        final_results["value"].to_numpy(dtype=np.float32),
        final_results["mean_prediction"].to_numpy(dtype=np.float32),
    )
    model_variant = model_name.split("/")[-1].replace(".", "_")
    prediction_frame = build_prediction_frame(
        task_id="cycpeptmpdb_perm",
        model_family="peptideclm_holdout_random_v1",
        model_variant=model_variant,
        seed=seed,
        split_id="cv_test",
        sample_ids=final_results["SMILES"],
        input_values=final_results["SMILES"],
        true_targets=final_results["value"],
        predictions=final_results["mean_prediction"].to_numpy(dtype=np.float32),
        prediction_type="regression",
        threshold=None,
    )
    metric_frame = build_metric_frame(
        task_id="cycpeptmpdb_perm",
        model_family="peptideclm_holdout_random_v1",
        model_variant=model_variant,
        seed=seed,
        split_id="cv_test",
        metrics=metrics,
        primary_metric_names={"r2", "rmse", "mae"},
    )
    payload = {
        "task": "cycpeptmpdb_perm",
        "model_name": model_name,
        "model_variant": model_variant,
        "model_scale": model_scale,
        "training_mode": "holdout_random_split",
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "effective_batch_size": batch_size * accumulate_grad_batches,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "val_check_interval": args.val_check_interval,
        "val_fraction": args.val_fraction,
        "val_stratify_bins": args.val_stratify_bins,
        "oversample_train_bins": args.oversample_train_bins,
        "head_dropout": args.head_dropout,
        "weight_decay": args.weight_decay,
        "random_init_backbone": bool(args.random_init_backbone),
        "test_folds": fold_ids,
        "status": "ready",
    }
    write_baseline_outputs(
        output_dir=run_dir,
        prediction_frame=prediction_frame,
        metric_frame=metric_frame,
        adapter_metadata=payload,
    )
    final_results.to_csv(run_dir / "cv_predictions_detailed.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(run_dir / "holdout_metrics.csv", index=False)
    (run_dir / "adapter_plan.json").write_text(pd.Series(payload).to_json(indent=2), encoding="utf-8")
    return {"run_dir": run_dir, "metrics": metric_frame}


def update_summary_frame(summary_path: Path, new_rows: list[dict[str, object]]) -> pd.DataFrame:
    new_frame = pd.DataFrame(new_rows)
    if summary_path.exists() and summary_path.stat().st_size > 0:
        existing_frame = pd.read_csv(summary_path)
    else:
        existing_frame = pd.DataFrame()
    if existing_frame.empty:
        return new_frame
    if new_frame.empty:
        return existing_frame
    combined = pd.concat([existing_frame, new_frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model_name", "seed"], keep="last")
    return combined.sort_values(["model_name", "seed"]).reset_index(drop=True)


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.all_models:
        selected_models = list(DEFAULT_MODELS)
    elif args.models:
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


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return list(args.seeds)
    if args.seed is not None:
        return [int(args.seed)]
    return list(DEFAULT_SEEDS)


def has_completed_seed_run(output_root: Path, model_name: str, seed: int) -> bool:
    seed_dir = build_run_dir(output_root, model_name, seed)
    metrics_path = seed_dir / "metrics.csv"
    return metrics_path.exists() and metrics_path.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Old-style random holdout regression fine-tuning for CycPeptMPDB PAMPA.")
    parser.add_argument("--data_csv", type=Path, default=REPO_ROOT / "tmp" / "prepared_data" / "cycpeptmpdb_perm" / "perm_external.csv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_type", choices=MODEL_TYPES, default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs_holdout_random_v1")
    parser.add_argument("--log_root", type=Path, default=REPO_ROOT / "tmp" / "logs" / "regression_holdout_random_v1")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--val_stratify_bins", type=int, default=5)
    parser.add_argument("--oversample_train_bins", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--head_dropout", type=float, default=0.20)
    parser.add_argument("--random_init_backbone", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test_folds", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < float(args.val_fraction) < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    return args


def main() -> int:
    args = parse_args()
    if not args.data_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data_csv}")
    models = resolve_models(args)
    seeds = resolve_seeds(args)
    data_frame = normalize_perm_frame(pd.read_csv(args.data_csv))
    summary_rows = []
    for model_name in models:
        for seed in seeds:
            if not args.force and has_completed_seed_run(args.output_root, model_name, seed):
                print(f"[skip-seed] found completed output for model={model_name} seed={seed}; skipping", flush=True)
                continue
            result = run_single_experiment(args, model_name=model_name, seed=seed, data_frame=data_frame)
            metric_frame = result["metrics"]
            summary = {"model_name": model_name, "model_scale": resolve_model_scale(model_name), "seed": seed, "run_dir": str(result["run_dir"])}
            for _, row in metric_frame.iterrows():
                summary[row["metric_name"]] = row["metric_value"]
            summary_rows.append(summary)
    summary_root = args.output_root / "cycpeptmpdb_perm" / "peptideclm_holdout_random_v1"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "summary_metrics.csv"
    summary_frame = update_summary_frame(summary_path, summary_rows)
    summary_frame.to_csv(summary_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())