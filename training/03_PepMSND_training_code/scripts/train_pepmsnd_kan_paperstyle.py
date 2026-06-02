import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kan import KANLinear
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


META_DROP_FOR_DESC = ["ID", "PMID", "SMILES", "label", "Length", "SE-3", "Species", "Environment"]


def get_model_specific_lrs(model_name: str) -> Tuple[float, float]:
    name = model_name.lower()
    if name.endswith("_sm") or "peptidemlm_sm" in name:
        # ~40M params: keep prior setting.
        return 1e-5, 1e-3
    if name.endswith("_base") or "peptidemlm_base" in name:
        # ~120M params: scale up from previous conservative settings.
        return 8e-6, 8e-4
    if name.endswith("_lg") or "peptidemlm_lg" in name:
        # ~340M params: scale up too, but keep slightly below base for stability.
        return 7e-6, 7e-4
    # Fallback for unexpected model names.
    return 1e-5, 1e-3


def get_model_specific_freeze_epochs(model_name: str) -> int:
    name = model_name.lower()
    if name.endswith("_sm") or "peptidemlm_sm" in name:
        return 3
    if name.endswith("_base") or "peptidemlm_base" in name:
        return 5
    if name.endswith("_lg") or "peptidemlm_lg" in name:
        # Let the large backbone adapt sooner.
        return 2
    return 3


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
        raise ValueError(f"Could not infer embedding dim from config: {config}")
    return int(dim)


@dataclass
class PrecomputedFeatures:
    kan_train: np.ndarray
    kan_test: np.ndarray
    meta_train: np.ndarray
    meta_test: np.ndarray


def build_paperstyle_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    append_meta_to_kan: bool,
    clip_quantile: float,
    max_abs_descriptor: float,
) -> PrecomputedFeatures:
    # 1) Descriptor block: scale train stats only (paper pretreatment behavior)
    # Keep float64 during statistics to avoid overflow on very large descriptor values.
    desc_train = train_df.drop(columns=META_DROP_FOR_DESC).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    desc_test = test_df.drop(columns=META_DROP_FOR_DESC).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    desc_train = np.nan_to_num(desc_train, nan=0.0, posinf=0.0, neginf=0.0)
    desc_test = np.nan_to_num(desc_test, nan=0.0, posinf=0.0, neginf=0.0)

    if 0.5 < clip_quantile < 1.0:
        lo_q = 1.0 - clip_quantile
        low = np.quantile(desc_train, lo_q, axis=0)
        high = np.quantile(desc_train, clip_quantile, axis=0)
        desc_train = np.clip(desc_train, low, high)
        desc_test = np.clip(desc_test, low, high)

    if max_abs_descriptor > 0:
        desc_train = np.clip(desc_train, -max_abs_descriptor, max_abs_descriptor)
        desc_test = np.clip(desc_test, -max_abs_descriptor, max_abs_descriptor)

    mean = desc_train.mean(axis=0)
    std = desc_train.std(axis=0)
    std[std < 1e-8] = 1.0

    desc_train = (desc_train - mean) / std
    desc_test = (desc_test - mean) / std

    # 2) Species/Environment numeric features
    meta_train = train_df[["Species", "Environment"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    meta_test = test_df[["Species", "Environment"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    # NOTE: PepMSND repo has an inconsistency: pretreatment appends meta to KAN,
    # while Models/KAN.py expects in_features=140. Keep this configurable.
    if append_meta_to_kan:
        kan_train = np.hstack([desc_train, meta_train]).astype(np.float32)
        kan_test = np.hstack([desc_test, meta_test]).astype(np.float32)
    else:
        kan_train = desc_train.astype(np.float32)
        kan_test = desc_test.astype(np.float32)

    return PrecomputedFeatures(
        kan_train=kan_train,
        kan_test=kan_test,
        meta_train=meta_train,
        meta_test=meta_test,
    )


class PepMSNDPaperDataset(Dataset):
    def __init__(
        self,
        smiles: list[str],
        labels: np.ndarray,
        kan_features: np.ndarray,
        meta_features: np.ndarray,
        tokenizer,
        max_length: int = 2048,
    ):
        self.encodings = tokenizer(
            smiles,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.kan_features = torch.tensor(kan_features, dtype=torch.float32)
        self.meta_features = torch.tensor(meta_features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["kan_features"] = self.kan_features[idx]
        item["meta_features"] = self.meta_features[idx]
        item["labels"] = self.labels[idx]
        return item


class PaperStyleKANBranch(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = KANLinear(in_dim, 128)
        self.fc2 = KANLinear(128, 128)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.norm(x)
        x = self.relu(self.fc2(x))
        return x


class PeptideCLMBackbonePlusPaperKAN(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        kan_input_dim: int,
        backbone_learning_rate: float = 1e-5,
        head_learning_rate: float = 1e-3,
        freeze_backbone_epochs: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        backbone_dim = infer_backbone_dim(self.backbone.config)

        self.seq_proj = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.ReLU(),
        )

        self.kan_branch = PaperStyleKANBranch(kan_input_dim)

        # Paper-style: KAN fusion and then append 2 metadata features at final step.
        self.fusion_fc1 = KANLinear(128 + 128, 256)
        self.fusion_bn = nn.BatchNorm1d(256)
        self.fusion_fc2 = KANLinear(256, 64)
        self.out_fc = KANLinear(64 + 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self._backbone_frozen = False
        self.val_probs = []
        self.val_labels = []

    def _set_backbone_frozen(self, frozen: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        # Keep mode explicit to avoid backbone modules remaining in eval unintentionally.
        if frozen:
            self.backbone.eval()
        else:
            self.backbone.train()
        self._backbone_frozen = frozen

    @staticmethod
    def _best_mcc_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        y_true_bin = (y_true >= 0.5).astype(int)
        unique_thresholds = np.unique(y_prob)
        if unique_thresholds.size == 0:
            return 0.5, float("nan")
        candidates = np.concatenate(
            [
                [np.nextafter(unique_thresholds[0], -np.inf)],
                unique_thresholds,
                [np.nextafter(unique_thresholds[-1], np.inf)],
            ]
        )
        best_t = float(candidates[0])
        best_mcc = -np.inf
        for t in candidates:
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_true_bin == 1) & (y_pred == 1))
            tn = np.sum((y_true_bin == 0) & (y_pred == 0))
            fp = np.sum((y_true_bin == 0) & (y_pred == 1))
            fn = np.sum((y_true_bin == 1) & (y_pred == 0))
            denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if denom == 0:
                mcc = -1.0
            else:
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denom)
            if (mcc > best_mcc + 1e-12) or (
                abs(mcc - best_mcc) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
            ):
                best_mcc = float(mcc)
                best_t = float(t)
        return best_t, best_mcc

    def forward(self, input_ids, attention_mask, kan_features, meta_features):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = get_last_hidden_state(outputs)
        pooled = masked_mean_pool(last_hidden, attention_mask)
        seq_feat = self.seq_proj(pooled)

        kan_feat = self.kan_branch(kan_features)

        x = torch.cat([seq_feat, kan_feat], dim=1)
        x = self.relu(self.fusion_fc1(x))
        x = self.fusion_bn(x)
        x = self.dropout(x)
        x = self.relu(self.fusion_fc2(x))
        x = torch.cat([x, meta_features], dim=1)
        logits = self.out_fc(x).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["kan_features"],
            batch["meta_features"],
        )
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["kan_features"],
            batch["meta_features"],
        )
        loss = self.loss_fn(logits, batch["labels"])
        probs = torch.sigmoid(logits).detach().cpu()
        labels = batch["labels"].detach().cpu()
        self.val_probs.append(probs)
        self.val_labels.append(labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_probs:
            probs = torch.cat(self.val_probs).numpy()
            labels = torch.cat(self.val_labels).numpy()
            best_t, best_mcc = self._best_mcc_threshold(labels, probs)
            self.log("val_mcc_best", float(best_mcc), prog_bar=True)
            self.log("val_thr_best", float(best_t), prog_bar=False)
        self.val_probs.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.seq_proj.parameters())
            + list(self.kan_branch.parameters())
            + list(self.fusion_fc1.parameters())
            + list(self.fusion_bn.parameters())
            + list(self.fusion_fc2.parameters())
            + list(self.out_fc.parameters())
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_learning_rate},
                {"params": head_params, "lr": self.hparams.head_learning_rate},
            ]
        )
        return optimizer

    def on_train_epoch_start(self):
        freeze_epochs = int(self.hparams.freeze_backbone_epochs)
        if self.current_epoch < freeze_epochs and not self._backbone_frozen:
            self._set_backbone_frozen(True)
        elif self.current_epoch >= freeze_epochs and self._backbone_frozen:
            self._set_backbone_frozen(False)


def run_inference(model, data_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kan_features = batch["kan_features"].to(device)
            meta_features = batch["meta_features"].to(device)

            logits = model(input_ids, attention_mask, kan_features, meta_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels)

    return np.concatenate(probs_all), np.concatenate(labels_all)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Aaron backbone + paper-style KAN fusion on PepMSND.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=list(range(1, 11)))
    parser.add_argument("--data_dir", type=str, default="main/data/PepMSND_data")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--head_learning_rate", type=float, default=None)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append_meta_to_kan", action="store_true")
    parser.add_argument("--monitor_metric", type=str, default="val_mcc_best", choices=["val_mcc_best", "val_loss"])
    parser.add_argument("--clip_quantile", type=float, default=0.999)
    parser.add_argument("--max_abs_descriptor", type=float, default=1e6)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    default_backbone_lr, default_head_lr = get_model_specific_lrs(args.model_name)
    default_freeze_epochs = get_model_specific_freeze_epochs(args.model_name)
    if args.backbone_learning_rate is None:
        args.backbone_learning_rate = default_backbone_lr
    if args.head_learning_rate is None:
        args.head_learning_rate = default_head_lr
    if args.freeze_backbone_epochs is None:
        args.freeze_backbone_epochs = default_freeze_epochs

    print(
        "[INFO] Learning rates: "
        f"backbone_lr={args.backbone_learning_rate:.2e}, "
        f"head_lr={args.head_learning_rate:.2e}"
    )
    print(f"[INFO] freeze_backbone_epochs={args.freeze_backbone_epochs}")

    train_csv = os.path.join(args.data_dir, f"X_train{args.fold}.csv")
    test_csv = os.path.join(args.data_dir, f"X_test{args.fold}.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    feat = build_paperstyle_features(
        train_df,
        test_df,
        append_meta_to_kan=args.append_meta_to_kan,
        clip_quantile=args.clip_quantile,
        max_abs_descriptor=args.max_abs_descriptor,
    )
    mode = "descriptors+species_env" if args.append_meta_to_kan else "descriptors_only"
    print(f"[INFO] KAN input dim: {feat.kan_train.shape[1]} ({mode})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    train_ds = PepMSNDPaperDataset(
        smiles=train_df["SMILES"].astype(str).tolist(),
        labels=train_df["label"].to_numpy(dtype=np.float32),
        kan_features=feat.kan_train,
        meta_features=feat.meta_train,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_ds = PepMSNDPaperDataset(
        smiles=test_df["SMILES"].astype(str).tolist(),
        labels=test_df["label"].to_numpy(dtype=np.float32),
        kan_features=feat.kan_test,
        meta_features=feat.meta_test,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

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

    model = PeptideCLMBackbonePlusPaperKAN(
        model_name=args.model_name,
        kan_input_dim=feat.kan_train.shape[1],
        backbone_learning_rate=args.backbone_learning_rate,
        head_learning_rate=args.head_learning_rate,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )

    os.makedirs(args.save_path, exist_ok=True)
    logger = CSVLogger(save_dir=args.save_path, name="logs")

    monitor_mode = "max" if args.monitor_metric == "val_mcc_best" else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "checkpoints"),
        monitor=args.monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        filename="best-{epoch:02d}",
    )
    early_stopping = EarlyStopping(monitor=args.monitor_metric, mode=monitor_mode, patience=args.patience)

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
        model = PeptideCLMBackbonePlusPaperKAN.load_from_checkpoint(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    probs, labels = run_inference(model, val_loader, device)
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
