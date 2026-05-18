import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def get_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        if "last_layer" in outputs:
            value = outputs["last_layer"]
            if isinstance(value, dict) and "last_hidden_state" in value:
                return value["last_hidden_state"]
            return value
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def infer_backbone_dim(config) -> int:
    dim = (
        getattr(config, "hidden_size", None)
        or getattr(config, "embed_dim", None)
        or getattr(config, "d_model", None)
        or getattr(config, "dim", None)
    )
    if dim is None:
        raise ValueError(f"Could not infer embedding dimension from config: {config}")
    return int(dim)


def build_species_env_onehot(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    train_meta = train_df[["Species", "Environment"]].copy().fillna("UNK").astype(str)
    test_meta = test_df[["Species", "Environment"]].copy().fillna("UNK").astype(str)

    train_ohe = pd.get_dummies(train_meta, columns=["Species", "Environment"], prefix=["Species", "Environment"])
    test_ohe = pd.get_dummies(test_meta, columns=["Species", "Environment"], prefix=["Species", "Environment"])

    # Align test columns to train-only feature space.
    test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

    feature_cols = list(train_ohe.columns)
    x_train = train_ohe.to_numpy(dtype=np.float32)
    x_test = test_ohe.to_numpy(dtype=np.float32)
    return x_train, x_test, feature_cols


@dataclass
class EncodedData:
    smiles: list[str]
    labels: np.ndarray
    meta_features: np.ndarray


class PepMSNDSpeciesEnvDataset(Dataset):
    def __init__(
        self,
        encoded_data: EncodedData,
        tokenizer,
        max_length: int = 2048,
    ):
        self.encodings = tokenizer(
            encoded_data.smiles,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.meta_features = torch.tensor(encoded_data.meta_features, dtype=torch.float32)
        self.labels = torch.tensor(encoded_data.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["meta_features"] = self.meta_features[idx]
        item["labels"] = self.labels[idx]
        return item


class SpeciesEnvHybridClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        meta_dim: int,
        learning_rate: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        backbone_dim = infer_backbone_dim(self.backbone.config)

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + meta_dim, backbone_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, meta_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = get_last_hidden_state(outputs)
        pooled = masked_mean_pool(last_hidden_state, attention_mask)
        fused = torch.cat([pooled, meta_features], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["meta_features"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["meta_features"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)


def run_inference(model, data_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            meta_features = batch["meta_features"].to(device)
            logits = model(input_ids, attention_mask, meta_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PepMSND model with Species/Environment one-hot fusion.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=list(range(1, 11)))
    parser.add_argument("--data_dir", type=str, default="main/data/PepMSND_data")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    train_csv = os.path.join(args.data_dir, f"X_train{args.fold}.csv")
    test_csv = os.path.join(args.data_dir, f"X_test{args.fold}.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    x_train_meta, x_test_meta, feature_cols = build_species_env_onehot(train_df, test_df)
    print(f"[INFO] Species/Environment one-hot dimension: {len(feature_cols)}")

    train_data = EncodedData(
        smiles=train_df["SMILES"].astype(str).tolist(),
        labels=train_df["label"].to_numpy(dtype=np.float32),
        meta_features=x_train_meta,
    )
    test_data = EncodedData(
        smiles=test_df["SMILES"].astype(str).tolist(),
        labels=test_df["label"].to_numpy(dtype=np.float32),
        meta_features=x_test_meta,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    train_ds = PepMSNDSpeciesEnvDataset(train_data, tokenizer, max_length=args.max_length)
    val_ds = PepMSNDSpeciesEnvDataset(test_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SpeciesEnvHybridClassifier(
        model_name=args.model_name,
        meta_dim=x_train_meta.shape[1],
        learning_rate=args.learning_rate,
    )

    os.makedirs(args.save_path, exist_ok=True)
    logger = CSVLogger(save_dir=args.save_path, name="logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    ckpt_path = checkpoint_callback.best_model_path
    if ckpt_path:
        model = SpeciesEnvHybridClassifier.load_from_checkpoint(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    infer_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    probs, labels = run_inference(model, infer_loader, device)
    out_df = pd.DataFrame(
        {
            "true_label": labels.astype(float),
            "predicted_prob": probs.astype(float),
            "predicted_label": (probs >= 0.5).astype(int),
        }
    )
    out_csv = os.path.join(args.save_path, f"preds_fold{args.fold}.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved predictions: {out_csv}")


if __name__ == "__main__":
    main()
