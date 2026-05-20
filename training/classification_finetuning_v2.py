import argparse
import os
import random
import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import transformers
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tokenizers

# Add this block at the top of your script
torch.serialization.add_safe_globals([
    transformers.tokenization_utils_tokenizers.TokenizersBackend,
    tokenizers.Tokenizer
])

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

CLASSIFICATION_DATASET_FILES = {
    "amp": "amp",
    "AmpHGT": "amp",
    "THPep": "THPep",
    "CellPPD": "CellPPD",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


###############################################
# DATASET + COLLATE
###############################################

class MoleculeDataset(Dataset):
    """
    Pre-tokenize everything once in __init__, so __getitem__ is O(1)
    and doesn't spawn threads or touch HF internals.
    """
    def __init__(self, smiles_list, labels, tokenizer, max_length=None):
        smiles_list = list(smiles_list)
        labels = list(labels)
        max_length=2048

        # Batch tokenization: one call instead of per-sample
        encodings = tokenizer(
            smiles_list,
            truncation=True,
            padding="longest",
            max_length=max_length,
            add_special_tokens=True,
        )

        self.input_ids = [
            torch.tensor(x, dtype=torch.long) for x in encodings["input_ids"]
        ]
        self.attention_masks = [
            torch.tensor(x, dtype=torch.long) for x in encodings["attention_mask"]
        ]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        input_ids = [x["input_ids"] for x in batch]
        masks = [x["attention_mask"] for x in batch]
        labels = torch.stack([x["label"] for x in batch])

        padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)

        return {
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
            "labels": labels,
        }

    return collate_fn

###############################################
# MODEL
###############################################

class PeptideModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate, target, tokenizer, total_steps):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer
        self.total_steps = total_steps
        
        # 1. Load Base Model
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 2. Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16, # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["qkv_proj"] # Target attention layers
        )
        
        # 3. Wrap with LoRA
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()
        
        # ... (rest of your FC layers remain the same)
        dim = self.model.config.embed_dim
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(dim, 1)

        if target == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.SmoothL1Loss()

        self.learning_rate = learning_rate
        self.target = target

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )["mean_pool"]

        x = self.fc1(out)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze(-1)

    def training_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Warmup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * self.total_steps), 
            num_training_steps=self.total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


###############################################
# TRAIN FUNCTION
###############################################

def train_model(
    train_smiles,
    train_labels,
    val_smiles,
    val_labels,
    target,
    model_name,
    gpu,
    run_name,
    log_dir,
    batch_size=None,
    max_epochs=10,
    learning_rate=3e-4,
    
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    train_ds = MoleculeDataset(train_smiles, train_labels, tokenizer)
    val_ds = MoleculeDataset(val_smiles, val_labels, tokenizer)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    # VERY IMPORTANT: keep num_workers low on this janky box
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    total_steps = (len(train_smiles) // batch_size) * max_epochs

    model = PeptideModel(
        model_name=model_name,
        learning_rate=learning_rate,
        target=target,
        tokenizer=tokenizer,
        total_steps=total_steps
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cuda",
        devices=[int(args.gpu_index)],
        val_check_interval=0.5,
        logger=CSVLogger(log_dir, name=run_name),
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path
    model = PeptideModel.load_from_checkpoint(
        best_path,
        model_name=model_name,
        learning_rate=learning_rate,
        target=target,
        tokenizer=tokenizer,
        weights_only=False,
    )

    return model


###############################################
# TEST / INFERENCE
###############################################

def evaluate_on_test_set(model, test_smiles, test_labels, model_name, batch_size=None):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    test_ds = MoleculeDataset(test_smiles, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    model.eval()
    model.to("cuda")

    preds = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()

            logits = model(ids, mask)
            preds.extend(logits.cpu().numpy())

    return np.array(preds)


###############################################
# TASK TYPE INFERENCE
###############################################

def infer_task(labels):
    if labels.dtype == object:
        return "classification"
    if labels.nunique() <= 10:
        return "classification"
    return "regression"


###############################################
# MAIN
###############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(CLASSIFICATION_DATASET_FILES))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--log_dir", type=str, default="tmp/logs/classification")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    dataset_prefix = CLASSIFICATION_DATASET_FILES[dataset]
    gpu = args.gpu_index if args.gpu_index is not None else args.gpu
    model_name = args.model_name
    batch_size = args.batch_size
    save_path = args.output_dir or args.save_path
    data_dir = args.data_dir
    log_dir = args.log_dir
    seed_everything(args.seed)

    train_df = pd.read_csv(os.path.join(data_dir, f"{dataset_prefix}_train.csv"))
    val_path = os.path.join(data_dir, f"{dataset_prefix}_val.csv")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.csv")

    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    run_name = f"{dataset}_{model_name.split('/')[-1]}_seed{args.seed}"

    task = infer_task(train_df["label"])

    ###############################################
    # TRAIN + EVAL PATHS
    ###############################################

    if val_df is not None and test_df is not None:
        model = train_model(train_df["smiles"], train_df["label"],
                            val_df["smiles"], val_df["label"],
                            task, model_name, gpu, run_name, log_dir, batch_size=batch_size)

        preds = evaluate_on_test_set(model, test_df["smiles"], test_df["label"], model_name, batch_size=64)

        out = pd.DataFrame({
            "smiles": test_df["smiles"],
            "true_label": test_df["label"],
            "predicted_label": preds,
        })

        os.makedirs(save_path, exist_ok=True)
        out.to_csv(f"{save_path}/{dataset}_{model_name.split('/')[-1]}_results.csv", index=False)

    else:
        # fallback to 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

        all_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            tr = train_df.iloc[train_idx]
            va = train_df.iloc[val_idx]

            model = train_model(tr["smiles"], tr["label"],
                                va["smiles"], va["label"],
                                task, model_name, gpu, run_name, log_dir, batch_size=batch_size)

            if test_df is not None:
                preds = evaluate_on_test_set(model, test_df["smiles"], test_df["label"], model_name, batch_size=64)

                df = pd.DataFrame({
                    "smiles": test_df["smiles"],
                    "true_label": test_df["label"],
                    "predicted_label": preds,
                    "fold": fold,
                })
                all_results.append(df)
                continue
            elif test_df is None:
                # evaluate on created val set
                preds = evaluate_on_test_set(model, va["smiles"], va["label"], model_name, batch_size=64)

            df = pd.DataFrame({
                "smiles": va["smiles"],
                "true_label": va["label"],
                "predicted_label": preds,
                "fold": fold,
            })
            all_results.append(df)

        final = pd.concat(all_results)
        os.makedirs(save_path, exist_ok=True)
        final.to_csv(f"{save_path}/{dataset}_{model_name.split('/')[-1]}_results.csv", index=False)