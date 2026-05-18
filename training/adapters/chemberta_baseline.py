from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    compute_classification_metrics,
    compute_regression_metrics,
    get_task_spec,
    load_task_frames,
    write_baseline_outputs,
)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    max_length: int


class SmilesDataset(Dataset):
    def __init__(self, smiles: pd.Series, labels: pd.Series):
        self.smiles = smiles.astype(str).tolist()
        self.labels = labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {"smiles": self.smiles[idx], "label": self.labels[idx]}


class ChembertaHead(nn.Module):
    def __init__(self, model_name: str, task_type: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "dim", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "embed_dim")
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_size, 1)
        self.task_type = task_type

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.head(self.dropout(pooled))
        return logits.squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChemBERTa-77M-MTR finetuning baseline.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prepared_data_root", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_index: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def collate_fn(tokenizer: AutoTokenizer, max_length: int, task_type: str):
    dtype = torch.float32

    def inner(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        smiles = [item["smiles"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=dtype)
        encodings = tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

    return inner


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def train_model(
    *,
    model_name: str,
    task_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_column: str,
    label_column: str,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, AutoTokenizer]:
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    model = ChembertaHead(model_name, task_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()

    train_loader = DataLoader(
        SmilesDataset(train_df[input_column], train_df[label_column]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, config.max_length, task_type),
    )
    val_loader = DataLoader(
        SmilesDataset(val_df[input_column], val_df[label_column]),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, config.max_length, task_type),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0

    for _epoch in range(config.max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, tokenizer


def predict(model: nn.Module, tokenizer: AutoTokenizer, smiles: pd.Series, config: TrainConfig, task_type: str, device: torch.device) -> np.ndarray:
    dataset = SmilesDataset(smiles, pd.Series(np.zeros(len(smiles), dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn(tokenizer, config.max_length, task_type))
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            batch_output = logits.detach().cpu().numpy()
            outputs.append(batch_output)
    predictions = np.concatenate(outputs, axis=0) if outputs else np.array([], dtype=np.float32)
    if task_type == "classification":
        return 1.0 / (1.0 + np.exp(-predictions))
    return predictions


def train_and_predict_fixed(args: argparse.Namespace, task_spec, config: TrainConfig, device: torch.device) -> tuple[np.ndarray, pd.DataFrame, dict]:
    frames = load_task_frames(args.task, args.seed, args.prepared_data_root)
    train_df = frames["train"]
    test_df = frames["test"]
    input_column = task_spec.input_column
    label_column = task_spec.label_column

    if "val" in frames:
        val_df = frames["val"]
        model, tokenizer = train_model(
            model_name=args.model_name,
            task_type=task_spec.task_type,
            train_df=train_df[[input_column, label_column]],
            val_df=val_df[[input_column, label_column]],
            input_column=input_column,
            label_column=label_column,
            config=config,
            seed=args.seed,
            device=device,
        )
        predictions = predict(model, tokenizer, test_df[input_column], config, task_spec.task_type, device)
        return predictions, test_df, {"strategy": "fixed_train_val_test", "fold_count": 1}

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_y = train_df[label_column].to_numpy(dtype=np.int32)
    fold_predictions = []
    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(train_df, train_y), start=1):
        model, tokenizer = train_model(
            model_name=args.model_name,
            task_type=task_spec.task_type,
            train_df=train_df.iloc[fit_idx][[input_column, label_column]],
            val_df=train_df.iloc[val_idx][[input_column, label_column]],
            input_column=input_column,
            label_column=label_column,
            config=config,
            seed=args.seed + fold_index,
            device=device,
        )
        fold_predictions.append(predict(model, tokenizer, test_df[input_column], config, task_spec.task_type, device))
    predictions = np.mean(np.vstack(fold_predictions), axis=0)
    return predictions, test_df, {"strategy": "five_fold_test_ensemble", "fold_count": 5}


def train_and_predict_regression_cv(args: argparse.Namespace, task_spec, config: TrainConfig, device: torch.device) -> tuple[np.ndarray, pd.DataFrame, dict]:
    full_df = load_task_frames(args.task, args.seed, args.prepared_data_root)["full"].copy().sort_values("fold").reset_index(drop=True)
    fold_ids = sorted(full_df["fold"].unique().tolist())
    predictions = np.zeros(len(full_df), dtype=np.float32)
    input_column = task_spec.input_column
    label_column = task_spec.label_column

    for fold_id in fold_ids:
        test_mask = full_df["fold"] == fold_id
        val_fold = fold_ids[(fold_ids.index(fold_id) - 1) % len(fold_ids)]
        val_mask = full_df["fold"] == val_fold
        train_mask = ~(test_mask | val_mask)
        model, tokenizer = train_model(
            model_name=args.model_name,
            task_type=task_spec.task_type,
            train_df=full_df.loc[train_mask, [input_column, label_column]],
            val_df=full_df.loc[val_mask, [input_column, label_column]],
            input_column=input_column,
            label_column=label_column,
            config=config,
            seed=args.seed + int(fold_id),
            device=device,
        )
        predictions[test_mask.to_numpy()] = predict(model, tokenizer, full_df.loc[test_mask, input_column], config, task_spec.task_type, device)

    return predictions, full_df, {"strategy": "provided_fold_cv", "fold_count": len(fold_ids)}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args.task,
        "model_name": args.model_name,
        "seed": args.seed,
        "status": "ready" if not args.dry_run else "dry_run",
    }
    (args.output_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if args.dry_run:
        return 0

    task_spec = get_task_spec(args.task)
    config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_length=args.max_length,
    )
    device = get_device(args.gpu_index)
    model_family = "chemberta77m"
    model_variant = args.model_name.split("/")[-1].replace(".", "_")

    if task_spec.task_type == "classification":
        predictions, test_df, metadata = train_and_predict_fixed(args, task_spec, config, device)
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
        predictions, test_df, metadata = train_and_predict_regression_cv(args, task_spec, config, device)
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
            "learning_rate": args.learning_rate,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "max_length": args.max_length,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder CLI for the ChemBERTa-77M-MTR baseline.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args.task,
        "model_name": args.model_name,
        "seed": args.seed,
        "status": "placeholder",
    }
    (args.output_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if args.dry_run:
        return 0
    raise SystemExit("ChemBERTa baseline execution is not implemented yet; use --dry_run for queue validation.")


if __name__ == "__main__":
    raise SystemExit(main())