from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import random
from pathlib import Path
import sys
import warnings

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tokenizers
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    write_baseline_outputs,
)
from training.experiment.manifest import REPO_ROOT


torch.serialization.add_safe_globals([
    transformers.tokenization_utils_tokenizers.TokenizersBackend,
    tokenizers.Tokenizer,
])

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


def resolve_learning_rate(model_name: str, override: float | None) -> float:
    if override is not None:
        return override
    normalized_name = model_name.lower()
    if "-small" in normalized_name:
        return 1e-5
    if "-base" in normalized_name:
        return 5e-6
    if "-large" in normalized_name:
        return 5e-6
    return 1e-5


def resolve_weight_decay(model_name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    normalized_name = model_name.lower()
    if "-base" in normalized_name:
        return 1e-3
    return 1e-3


def resolve_head_dropout(model_name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    normalized_name = model_name.lower()
    if "-base" in normalized_name:
        return 0.20
    return 0.15


def resolve_warmup_fraction(model_name: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    normalized_name = model_name.lower()
    if "-base" in normalized_name:
        return 0.20
    return 0.10


def resolve_model_scale(model_name: str) -> str:
    normalized_name = model_name.lower()
    if "-small" in normalized_name:
        return "small"
    if "-large" in normalized_name:
        return "large"
    return "base"


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
        return 32
    if scale == "large":
        return 8
    return 16


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


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable_params, total_params


class MoleculeRegressionDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length: int = 2048):
        smiles_list = list(smiles_list)
        labels = list(labels)
        encodings = tokenizer(
            smiles_list,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=True,
        )
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encodings["input_ids"]]
        self.attention_masks = [torch.ones_like(ids, dtype=torch.long) for ids in self.input_ids]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


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


class FullFinetuneRegressionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        total_steps: int,
        head_dropout: float,
        weight_decay: float,
        warmup_fraction: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        embed_dim = resolve_embed_dim(self.model)
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = forward_backbone(self.model, input_ids=input_ids, attention_mask=attention_mask)
        pooled = extract_mean_pool(outputs)
        prediction = self.regression_head(pooled).squeeze(-1)
        return prediction

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        predictions = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(predictions, batch["labels"])
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        predictions = self(batch["input_ids"], batch["attention_mask"])
        mse = self.criterion(predictions, batch["labels"])
        rmse = torch.sqrt(mse + 1e-12)
        self.log("val_loss", mse, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return mse

    def predict_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self(batch["input_ids"], batch["attention_mask"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(self.hparams.warmup_fraction * self.hparams.total_steps)),
            num_training_steps=max(1, self.hparams.total_steps),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def build_trainer(
    log_dir: Path,
    run_name: str,
    gpu_index: int,
    max_epochs: int,
    max_steps: int,
    patience: int,
    accumulate_grad_batches: int,
):
    checkpoint_callback = ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1, filename="best")
    early_stopping_callback = EarlyStopping(
        monitor="val_rmse",
        mode="min",
        patience=patience,
        min_delta=1e-4,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [int(gpu_index)] if accelerator == "gpu" else 1
    precision = "16-mixed" if accelerator == "gpu" else "32-true"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=CSVLogger(str(log_dir), name=run_name),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        deterministic=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    return trainer, checkpoint_callback


def predict_from_batches(prediction_batches: list[torch.Tensor]) -> np.ndarray:
    flattened = [batch.detach().float().cpu().view(-1).numpy() for batch in prediction_batches]
    if not flattened:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(flattened).astype(np.float32)


def resolve_fold_seed(seed: int, test_fold: int, val_fold: int) -> int:
    return int(seed) * 1000 + int(test_fold) * 100 + int(val_fold)


def run_single_fold_job(
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    data_frame: pd.DataFrame,
    test_fold: int,
    val_fold: int,
) -> np.ndarray:
    fold_seed = resolve_fold_seed(seed, test_fold, val_fold)
    seed_everything(fold_seed)
    print(
        f"[worker-start] pid={os.getpid()} model={model_name} seed={seed} "
        f"test_fold={test_fold} val_fold={val_fold} gpu_index={args.gpu_index} "
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()} "
        f"torch_cuda_version={torch.version.cuda}",
        flush=True,
    )
    print(f"[load-tokenizer] model={model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = args.max_length
    pad_token_id = resolve_pad_token_id(tokenizer)
    collate_fn = make_collate_fn(pad_token_id)

    model_variant = model_name.split("/")[-1].replace(".", "_")
    run_dir = args.output_root / "cycpeptmpdb_perm" / "peptideclm_fullft_v2" / model_variant / f"seed_{seed}"
    log_dir = args.log_root / "cycpeptmpdb_perm" / "peptideclm_fullft_v2" / model_variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    model_scale = resolve_model_scale(model_name)
    learning_rate = resolve_learning_rate(model_name, args.learning_rate)
    weight_decay = resolve_weight_decay(model_name, args.weight_decay)
    head_dropout = resolve_head_dropout(model_name, args.head_dropout)
    warmup_fraction = resolve_warmup_fraction(model_name, args.warmup_fraction)
    batch_size = resolve_batch_size(model_name, args.batch_size)
    eval_batch_size = resolve_eval_batch_size(model_name, args.eval_batch_size)
    accumulate_grad_batches = resolve_accumulate_grad_batches(model_name, args.accumulate_grad_batches)
    print(
        f"[config] model={model_name} scale={model_scale} seed={seed} "
        f"target_scaling={args.target_scaling} lr={learning_rate:.2e} "
        f"batch_size={batch_size} eval_batch_size={eval_batch_size} "
        f"accumulate_grad_batches={accumulate_grad_batches} "
        f"head_dropout={head_dropout:.2f} weight_decay={weight_decay:.4f} "
        f"warmup_fraction={warmup_fraction:.2f}",
        flush=True,
    )

    test_df = data_frame.loc[data_frame["fold"] == test_fold].copy().reset_index(drop=True)
    ensemble_df = data_frame.loc[data_frame["fold"] != test_fold].copy().reset_index(drop=True)
    train_df = ensemble_df.loc[ensemble_df["fold"] != val_fold].copy().reset_index(drop=True)
    val_df = ensemble_df.loc[ensemble_df["fold"] == val_fold].copy().reset_index(drop=True)

    if args.target_scaling == "minmax":
        value_min, value_max = fit_minmax_scaler(train_df["value"])
        train_targets = transform_targets(train_df["value"], value_min, value_max)
        val_targets = transform_targets(val_df["value"], value_min, value_max)
        test_targets = transform_targets(test_df["value"], value_min, value_max)
    else:
        value_min, value_max = 0.0, 1.0
        train_targets = train_df["value"].to_numpy(dtype=np.float32)
        val_targets = val_df["value"].to_numpy(dtype=np.float32)
        test_targets = test_df["value"].to_numpy(dtype=np.float32)

    print(
        f"[fold] test_fold={test_fold} val_fold={val_fold} fold_seed={fold_seed} "
        f"train_n={len(train_df)} val_n={len(val_df)} test_n={len(test_df)} "
        f"value_min={value_min:.4f} value_max={value_max:.4f}",
        flush=True,
    )

    train_ds = MoleculeRegressionDataset(train_df["SMILES"], train_targets, tokenizer, max_length=args.max_length)
    val_ds = MoleculeRegressionDataset(val_df["SMILES"], val_targets, tokenizer, max_length=args.max_length)
    test_ds = MoleculeRegressionDataset(test_df["SMILES"], test_targets, tokenizer, max_length=args.max_length)

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

    if args.max_steps is not None:
        total_steps = max(1, int(args.max_steps))
    else:
        optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / accumulate_grad_batches))
        total_steps = max(1, optimizer_steps_per_epoch * args.max_epochs)
    print(f"[load-model] model={model_name}", flush=True)
    model = FullFinetuneRegressionModel(
        model_name=model_name,
        learning_rate=learning_rate,
        total_steps=total_steps,
        head_dropout=head_dropout,
        weight_decay=weight_decay,
        warmup_fraction=warmup_fraction,
    )
    trainable_params, total_params = count_trainable_parameters(model)
    print(
        f"[trainable] trainable_params={trainable_params} total_params={total_params} "
        f"trainable_fraction={trainable_params / max(1, total_params):.4f}",
        flush=True,
    )

    run_name = f"fold{test_fold}_val{val_fold}"
    trainer, checkpoint_callback = build_trainer(
        log_dir=log_dir,
        run_name=run_name,
        gpu_index=args.gpu_index,
        max_epochs=args.max_epochs,
        max_steps=total_steps,
        patience=args.patience,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    print(
        f"[trainer-fit] model={model_name} test_fold={test_fold} val_fold={val_fold} "
        f"accelerator={'gpu' if torch.cuda.is_available() else 'cpu'} gpu_index={args.gpu_index}",
        flush=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"[trainer-predict] model={model_name} test_fold={test_fold} val_fold={val_fold}", flush=True)
    prediction_batches = trainer.predict(model, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
    fold_predictions = predict_from_batches(prediction_batches)
    if args.target_scaling == "minmax":
        fold_predictions = inverse_transform_targets(fold_predictions, value_min, value_max)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_predictions


def write_fold_predictions(output_path: Path, predictions: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_frame = pd.DataFrame(
        {
            "row_idx": np.arange(len(predictions), dtype=np.int64),
            "prediction": np.asarray(predictions, dtype=np.float32),
        }
    )
    artifact_frame.to_csv(output_path, index=False)


def read_fold_predictions(output_path: Path, expected_rows: int) -> np.ndarray:
    artifact_frame = pd.read_csv(output_path)
    required_columns = {"row_idx", "prediction"}
    if not required_columns.issubset(artifact_frame.columns):
        raise ValueError(f"Fold artifact missing required columns at {output_path}")
    ordered_frame = artifact_frame.sort_values("row_idx").reset_index(drop=True)
    predictions = ordered_frame["prediction"].to_numpy(dtype=np.float32)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Fold artifact row count mismatch at {output_path}: expected {expected_rows}, found {len(predictions)}"
        )
    return predictions


def append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_fold_worker_args(
    args: argparse.Namespace,
    test_fold: int,
    val_fold: int,
    output_path: Path,
) -> argparse.Namespace:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.worker_test_fold = int(test_fold)
    worker_args.worker_val_fold = int(val_fold)
    worker_args.worker_output_path = output_path
    return worker_args


def fold_worker_entrypoint(
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    data_frame: pd.DataFrame,
    output_path: Path,
    worker_log_path: Path,
    result_queue,
) -> None:
    with worker_log_path.open("w", encoding="utf-8") as worker_log_handle:
        with contextlib.redirect_stdout(worker_log_handle), contextlib.redirect_stderr(worker_log_handle):
            try:
                predictions = run_single_fold_job(
                    args=args,
                    model_name=model_name,
                    seed=seed,
                    data_frame=data_frame,
                    test_fold=int(args.worker_test_fold),
                    val_fold=int(args.worker_val_fold),
                )
                write_fold_predictions(output_path, predictions)
                result_queue.put({
                    "status": "ok",
                    "test_fold": int(args.worker_test_fold),
                    "val_fold": int(args.worker_val_fold),
                    "output_path": str(output_path),
                })
            except Exception as exc:  # pragma: no cover - surfaced to parent via queue
                result_queue.put(
                    {
                        "status": "error",
                        "test_fold": int(args.worker_test_fold),
                        "val_fold": int(args.worker_val_fold),
                        "output_path": str(output_path),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                raise


def run_parallel_val_fold_workers(
    args: argparse.Namespace,
    model_name: str,
    seed: int,
    data_frame: pd.DataFrame,
    test_fold: int,
    val_folds: list[int],
    run_dir: Path,
) -> dict[int, np.ndarray]:
    if mp.get_start_method(allow_none=True) not in {None, "fork"}:
        raise RuntimeError("parallel_val_folds requires the 'fork' multiprocessing start method on this host")
    max_parallel = max(1, min(int(args.parallel_val_folds), len(val_folds)))
    test_df = data_frame.loc[data_frame["fold"] == test_fold].copy().reset_index(drop=True)
    artifact_root = run_dir / "_fold_jobs" / f"test_fold_{test_fold}"
    worker_log_root = artifact_root / "logs"
    worker_log_root.mkdir(parents=True, exist_ok=True)
    predictions_by_fold: dict[int, np.ndarray] = {}
    mp_context = mp.get_context("fork")

    for batch_start in range(0, len(val_folds), max_parallel):
        batch_val_folds = val_folds[batch_start:batch_start + max_parallel]
        result_queue = mp_context.Queue()
        processes: list[tuple[int, Path, Path, mp.Process]] = []
        for val_fold in batch_val_folds:
            output_path = artifact_root / f"val_fold_{val_fold}.csv"
            worker_log_path = worker_log_root / f"val_fold_{val_fold}.log"
            if output_path.exists():
                output_path.unlink()
            if worker_log_path.exists():
                worker_log_path.unlink()
            worker_args = build_fold_worker_args(args=args, test_fold=test_fold, val_fold=val_fold, output_path=output_path)
            print(
                f"[parallel-launch] test_fold={test_fold} val_fold={val_fold} gpu_index={args.gpu_index} log={worker_log_path}",
                flush=True,
            )
            process = mp_context.Process(
                target=fold_worker_entrypoint,
                args=(worker_args, model_name, seed, data_frame, output_path, worker_log_path, result_queue),
            )
            process.start()
            processes.append((val_fold, output_path, worker_log_path, process))

        try:
            process_by_fold = {val_fold: process for val_fold, _, _, process in processes}
            output_path_by_fold = {val_fold: output_path for val_fold, output_path, _, _ in processes}
            log_path_by_fold = {val_fold: worker_log_path for val_fold, _, worker_log_path, _ in processes}
            results_by_fold: dict[int, dict[str, object]] = {}
            pending_folds = set(batch_val_folds)

            while pending_folds:
                payload = result_queue.get()
                val_fold = int(payload["val_fold"])
                results_by_fold[val_fold] = payload
                process = process_by_fold[val_fold]
                process.join(timeout=5)
                if process.is_alive():
                    raise RuntimeError(
                        f"Fold worker did not exit cleanly for model={model_name} seed={seed} test_fold={test_fold} "
                        f"val_fold={val_fold}. See {log_path_by_fold[val_fold]}"
                    )
                if payload["status"] != "ok" or process.exitcode != 0:
                    raise RuntimeError(
                        f"Fold worker failed for model={model_name} seed={seed} test_fold={test_fold} "
                        f"val_fold={val_fold} exit_code={process.exitcode} "
                        f"error={payload.get('error_type', 'UnknownError')}: {payload.get('error_message', 'no message')}. "
                        f"See {log_path_by_fold[val_fold]}"
                    )
                predictions_by_fold[val_fold] = read_fold_predictions(
                    output_path_by_fold[val_fold], expected_rows=len(test_df)
                )
                pending_folds.remove(val_fold)
        except Exception:
            for _, _, _, process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            result_queue.close()
            result_queue.join_thread()
            raise

        result_queue.close()
        result_queue.join_thread()

    return predictions_by_fold


def run_fold_worker(args: argparse.Namespace, model_name: str, seed: int, data_frame: pd.DataFrame) -> None:
    if args.worker_test_fold is None or args.worker_val_fold is None or args.worker_output_path is None:
        raise ValueError("Worker mode requires test fold, validation fold, and output path.")
    predictions = run_single_fold_job(
        args=args,
        model_name=model_name,
        seed=seed,
        data_frame=data_frame,
        test_fold=int(args.worker_test_fold),
        val_fold=int(args.worker_val_fold),
    )
    write_fold_predictions(args.worker_output_path, predictions)


def run_single_experiment(args: argparse.Namespace, model_name: str, seed: int, data_frame: pd.DataFrame) -> dict[str, Path | pd.DataFrame]:
    model_variant = model_name.split("/")[-1].replace(".", "_")
    run_dir = args.output_root / "cycpeptmpdb_perm" / "peptideclm_fullft_v2" / model_variant / f"seed_{seed}"
    log_dir = args.log_root / "cycpeptmpdb_perm" / "peptideclm_fullft_v2" / model_variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    model_scale = resolve_model_scale(model_name)
    learning_rate = resolve_learning_rate(model_name, args.learning_rate)
    weight_decay = resolve_weight_decay(model_name, args.weight_decay)
    head_dropout = resolve_head_dropout(model_name, args.head_dropout)
    warmup_fraction = resolve_warmup_fraction(model_name, args.warmup_fraction)
    batch_size = resolve_batch_size(model_name, args.batch_size)
    eval_batch_size = resolve_eval_batch_size(model_name, args.eval_batch_size)
    accumulate_grad_batches = resolve_accumulate_grad_batches(model_name, args.accumulate_grad_batches)
    fold_ids = sorted(data_frame["fold"].unique().tolist())
    print(
        f"[ensemble] model={model_name} seed={seed} holdout_folds={len(fold_ids)} parallel_val_folds={args.parallel_val_folds}",
        flush=True,
    )

    for test_fold in fold_ids:
        test_df = data_frame.loc[data_frame["fold"] == test_fold].copy().reset_index(drop=True)
        ensemble_df = data_frame.loc[data_frame["fold"] != test_fold].copy().reset_index(drop=True)
        val_folds = sorted(ensemble_df["fold"].unique().tolist())

        if args.parallel_val_folds > 1 and len(val_folds) > 1:
            predictions_by_fold = run_parallel_val_fold_workers(
                args=args,
                model_name=model_name,
                seed=seed,
                data_frame=data_frame,
                test_fold=test_fold,
                val_folds=val_folds,
                run_dir=run_dir,
            )
        else:
            predictions_by_fold = {}
            for val_fold in val_folds:
                predictions_by_fold[val_fold] = run_single_fold_job(
                    args=args,
                    model_name=model_name,
                    seed=seed,
                    data_frame=data_frame,
                    test_fold=test_fold,
                    val_fold=val_fold,
                )

        for val_fold in val_folds:
            test_df.loc[:, f"prediction_{val_fold}"] = predictions_by_fold[val_fold]

        prediction_columns = sorted(column for column in test_df.columns if column.startswith("prediction_"))
        test_df.loc[:, "mean_prediction"] = test_df[prediction_columns].mean(axis=1)
        test_df.loc[:, "std_prediction"] = test_df[prediction_columns].std(axis=1)
        fold_results.append(test_df)

    final_results = pd.concat(fold_results, ignore_index=True)
    metrics = compute_regression_metrics_with_mse(
        final_results["value"].to_numpy(dtype=np.float32),
        final_results["mean_prediction"].to_numpy(dtype=np.float32),
    )

    prediction_frame = build_prediction_frame(
        task_id="cycpeptmpdb_perm",
        model_family="peptideclm_fullft_v2",
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
        model_family="peptideclm_fullft_v2",
        model_variant=model_variant,
        seed=seed,
        split_id="cv_test",
        metrics=metrics,
        primary_metric_names={"r2", "rmse", "mae"},
    )

    payload = {
        "task": "cycpeptmpdb_perm",
        "model_name": model_name,
        "model_scale": model_scale,
        "training_mode": "full_finetune",
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "fold_count": len(fold_ids),
        "status": "ready",
        "head_dropout": head_dropout,
        "weight_decay": weight_decay,
        "warmup_fraction": warmup_fraction,
        "target_scaling": args.target_scaling,
        "parallel_val_folds": int(args.parallel_val_folds),
    }
    (run_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_baseline_outputs(
        output_dir=run_dir,
        prediction_frame=prediction_frame,
        metric_frame=metric_frame,
        adapter_metadata=payload,
    )
    final_results.to_csv(run_dir / "cv_predictions_detailed.csv", index=False)
    return {"run_dir": run_dir, "metrics": metric_frame, "predictions": final_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-backbone regression ensemble fine-tuning for the CycPeptMPDB PAMPA benchmark.")
    parser.add_argument("--data_csv", type=Path, default=REPO_ROOT / "tmp" / "prepared_data" / "cycpeptmpdb_perm" / "perm_external.csv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_type", choices=MODEL_TYPES, default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs_full_regression_v2")
    parser.add_argument("--log_root", type=Path, default=REPO_ROOT / "tmp" / "logs" / "regression_full_v2")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--warmup_fraction", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_scaling", choices=("minmax", "none"), default="none")
    parser.add_argument("--parallel_val_folds", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Rerun requested seeds even if output metrics already exist.")
    parser.add_argument("--worker_test_fold", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_val_fold", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output_path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


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
    model_variant = model_name.split("/")[-1].replace(".", "_")
    seed_dir = output_root / "cycpeptmpdb_perm" / "peptideclm_fullft_v2" / model_variant / f"seed_{seed}"
    metrics_path = seed_dir / "metrics.csv"
    return metrics_path.exists() and metrics_path.stat().st_size > 0


def main() -> int:
    args = parse_args()
    if not args.data_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data_csv}")
    if args.parallel_val_folds < 1:
        raise ValueError("parallel_val_folds must be at least 1")

    worker_mode = any(
        value is not None for value in (args.worker_test_fold, args.worker_val_fold, args.worker_output_path)
    )
    models = resolve_models(args)
    seeds = resolve_seeds(args)
    data_frame = normalize_perm_frame(pd.read_csv(args.data_csv))
    if worker_mode:
        if len(models) != 1 or len(seeds) != 1:
            raise ValueError("Worker mode requires exactly one model and one seed.")
        run_fold_worker(args, model_name=models[0], seed=seeds[0], data_frame=data_frame)
        return 0

    summary_rows = []
    for model_name in models:
        for seed in seeds:
            if not args.force and has_completed_seed_run(args.output_root, model_name, seed):
                print(f"[skip-seed] found completed output for model={model_name} seed={seed}; skipping", flush=True)
                continue
            result = run_single_experiment(args, model_name=model_name, seed=seed, data_frame=data_frame)
            metric_frame = result["metrics"]
            summary = {"model_name": model_name, "seed": seed, "run_dir": str(result["run_dir"])}
            for _, row in metric_frame.iterrows():
                summary[row["metric_name"]] = row["metric_value"]
            summary_rows.append(summary)

    summary_root = args.output_root / "cycpeptmpdb_perm" / "peptideclm_fullft_v2"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "summary_metrics.csv"
    summary_frame = update_summary_frame(summary_path, summary_rows)
    summary_frame.to_csv(summary_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())