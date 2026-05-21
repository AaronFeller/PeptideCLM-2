from __future__ import annotations

import argparse
import json
import math
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
from peft import LoraConfig, TaskType, get_peft_model
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
    return 3e-4


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


def infer_lora_target_modules(model) -> list[str]:
    candidate_suffixes = [
        "qkv_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "query",
        "key",
        "value",
        "Wqkv",
        "c_attn",
    ]
    discovered = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = module_name.split(".")[-1]
        if leaf_name in candidate_suffixes:
            discovered.add(leaf_name)
    if discovered:
        return sorted(discovered)
    raise ValueError("Unable to infer LoRA target modules from backbone linear layers.")


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


def rebalance_regression_bins(train_df: pd.DataFrame, seed: int, bins: int = 5) -> pd.DataFrame:
    if len(train_df) < bins * 2:
        return train_df.copy().reset_index(drop=True)
    try:
        binned = pd.qcut(train_df["value"], q=bins, duplicates="drop")
    except ValueError:
        return train_df.copy().reset_index(drop=True)
    if getattr(binned, "nunique", lambda: 0)() <= 1:
        return train_df.copy().reset_index(drop=True)
    grouped = train_df.assign(_bin=binned).groupby("_bin", observed=False)
    max_size = int(grouped.size().max())
    balanced = [
        group.sample(n=max_size, replace=True, random_state=seed)
        for _, group in grouped
    ]
    return pd.concat(balanced, ignore_index=True).drop(columns="_bin")


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


class LoRARegressionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        total_steps: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        head_dropout: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        target_modules = infer_lora_target_modules(base_model)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(base_model, peft_config)
        embed_dim = resolve_embed_dim(base_model)
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, 1),
        )
        self.criterion = nn.SmoothL1Loss()

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
        smooth_l1 = self.criterion(predictions, batch["labels"])
        mse = nn.functional.mse_loss(predictions, batch["labels"])
        rmse = torch.sqrt(mse + 1e-12)
        self.log("val_loss", smooth_l1, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return smooth_l1

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
            num_warmup_steps=max(1, int(0.1 * self.hparams.total_steps)),
            num_training_steps=max(1, self.hparams.total_steps),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def build_trainer(log_dir: Path, run_name: str, gpu_index: int, max_epochs: int, patience: int):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best")
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
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
        log_every_n_steps=10,
        enable_progress_bar=False,
        deterministic=True,
        gradient_clip_val=0.1,
    )
    return trainer, checkpoint_callback


def predict_from_batches(prediction_batches: list[torch.Tensor]) -> np.ndarray:
    flattened = [batch.detach().float().cpu().view(-1).numpy() for batch in prediction_batches]
    if not flattened:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(flattened).astype(np.float32)


def run_single_experiment(args: argparse.Namespace, model_name: str, seed: int, data_frame: pd.DataFrame) -> dict[str, Path | pd.DataFrame]:
    seed_everything(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = args.max_length
    pad_token_id = resolve_pad_token_id(tokenizer)
    collate_fn = make_collate_fn(pad_token_id)

    model_variant = model_name.split("/")[-1].replace(".", "_")
    run_dir = args.output_root / "cycpeptmpdb_perm" / "peptideclm_lora_v2" / model_variant / f"seed_{seed}"
    log_dir = args.log_root / "cycpeptmpdb_perm" / "peptideclm_lora_v2" / model_variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    learning_rate = resolve_learning_rate(model_name, args.learning_rate)
    fold_ids = sorted(data_frame["fold"].unique().tolist())

    for test_fold in fold_ids:
        test_df = data_frame.loc[data_frame["fold"] == test_fold].copy().reset_index(drop=True)
        ensemble_df = data_frame.loc[data_frame["fold"] != test_fold].copy().reset_index(drop=True)

        for val_fold in sorted(ensemble_df["fold"].unique().tolist()):
            train_df = ensemble_df.loc[ensemble_df["fold"] != val_fold].copy().reset_index(drop=True)
            val_df = ensemble_df.loc[ensemble_df["fold"] == val_fold].copy().reset_index(drop=True)
            train_df = rebalance_regression_bins(train_df, seed=seed + int(test_fold) + int(val_fold))

            train_ds = MoleculeRegressionDataset(train_df["SMILES"], train_df["value"], tokenizer, max_length=args.max_length)
            val_ds = MoleculeRegressionDataset(val_df["SMILES"], val_df["value"], tokenizer, max_length=args.max_length)
            test_ds = MoleculeRegressionDataset(test_df["SMILES"], test_df["value"], tokenizer, max_length=args.max_length)

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

            total_steps = max(1, math.ceil(len(train_loader)) * args.max_epochs)
            model = LoRARegressionModel(
                model_name=model_name,
                learning_rate=learning_rate,
                total_steps=total_steps,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                head_dropout=args.head_dropout,
                weight_decay=args.weight_decay,
            )

            run_name = f"fold{test_fold}_val{val_fold}"
            trainer, checkpoint_callback = build_trainer(
                log_dir=log_dir,
                run_name=run_name,
                gpu_index=args.gpu_index,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            prediction_batches = trainer.predict(model, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
            test_df.loc[:, f"prediction_{val_fold}"] = predict_from_batches(prediction_batches)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        model_family="peptideclm_lora_v2",
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
        model_family="peptideclm_lora_v2",
        model_variant=model_variant,
        seed=seed,
        split_id="cv_test",
        metrics=metrics,
        primary_metric_names={"r2", "rmse", "mae"},
    )

    payload = {
        "task": "cycpeptmpdb_perm",
        "model_name": model_name,
        "seed": seed,
        "learning_rate": learning_rate,
        "fold_count": len(fold_ids),
        "status": "ready",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
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
    parser = argparse.ArgumentParser(description="LoRA-based regression ensemble fine-tuning for the CycPeptMPDB PAMPA benchmark.")
    parser.add_argument("--data_csv", type=Path, default=REPO_ROOT / "tmp" / "prepared_data" / "cycpeptmpdb_perm" / "perm_external.csv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_type", choices=MODEL_TYPES, default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "tmp" / "runs_lora_regression_v2")
    parser.add_argument("--log_root", type=Path, default=REPO_ROOT / "tmp" / "logs" / "regression_lora_v2")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


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


def has_completed_model_run(output_root: Path, model_name: str) -> bool:
    model_variant = model_name.split("/")[-1].replace(".", "_")
    model_root = output_root / "cycpeptmpdb_perm" / "peptideclm_lora_v2" / model_variant
    if not model_root.exists():
        return False
    for seed_dir in sorted(model_root.glob("seed_*")):
        metrics_path = seed_dir / "metrics.csv"
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            return True
    return False


def main() -> int:
    args = parse_args()
    if not args.data_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data_csv}")

    data_frame = normalize_perm_frame(pd.read_csv(args.data_csv))
    models = resolve_models(args)
    seeds = resolve_seeds(args)

    summary_rows = []
    for model_name in models:
        if has_completed_model_run(args.output_root, model_name):
            print(f"[skip-model] found completed output for model={model_name}; skipping all requested seeds", flush=True)
            continue
        for seed in seeds:
            result = run_single_experiment(args, model_name=model_name, seed=seed, data_frame=data_frame)
            metric_frame = result["metrics"]
            summary = {"model_name": model_name, "seed": seed, "run_dir": str(result["run_dir"])}
            for _, row in metric_frame.iterrows():
                summary[row["metric_name"]] = row["metric_value"]
            summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    summary_root = args.output_root / "cycpeptmpdb_perm" / "peptideclm_lora_v2"
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_root / "summary_metrics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())