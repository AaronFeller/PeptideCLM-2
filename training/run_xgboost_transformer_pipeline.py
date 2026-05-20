from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

# Mount repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

from training.adapters.common import (
    build_metric_frame,
    build_prediction_frame,
    compute_classification_metrics,
    get_task_spec,
    load_task_frames,
    write_baseline_outputs,
)

# Secure Serialization Allowlist
import tokenizers
import transformers
torch.serialization.add_safe_globals([
    transformers.tokenization_utils_tokenizers.TokenizersBackend,
    tokenizers.Tokenizer
])

TASKS = ["amp_hgt", "cellppd", "thpep"]
MODELS = [
    "aaronfeller/peptideclm-2-mtr-large",
    "aaronfeller/peptideclm-2-mlm-large",
    "aaronfeller/peptideclm-2-hybrid-large"
]
SEEDS = [101, 202, 303]
CACHE_ROOT = REPO_ROOT / "tmp" / "embeddings"
RUNS_ROOT = REPO_ROOT / "tmp" / "runs4"


@torch.no_grad()
def compute_embeddings(model, tokenizer, device: torch.device, smiles_list: list[str], batch_size: int = 32) -> np.ndarray:
    model.eval()
    embeddings = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(inputs.input_ids, mask=inputs.attention_mask)
        # Extract pre-calculated native pooling attribute
        batch_emb = outputs.mean_pool.cpu().numpy().astype(np.float32)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)


def get_cached_embeddings(model, tokenizer, device: torch.device, smiles_list: list[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"    [CACHE HIT] Loading embeddings from {cache_path}")
        return np.load(cache_path)
    
    print(f"    [COMPUTING] Generating embeddings to save at {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = compute_embeddings(model, tokenizer, device, smiles_list)
    np.save(cache_path, embeddings)
    return embeddings


def fit_classification_with_val(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, seed: int) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=512,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Targeting device: {device}")

    for model_name in MODELS:
        clean_name = model_name.split("/")[-1].replace("-", "_")
        print(f"\n{'='*60}\nPROCESSING MODEL: {model_name}\n{'='*60}")
        
        # 1. LOAD MODEL (ONCE PER MODEL)
        print("[1/3] Loading Transformer Backbone into VRAM...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        
        # Dictionary to hold the features in memory so we don't have to reload from disk right away
        feature_cache = {}

        # 2. GENERATE OR LOAD EMBEDDINGS
        print("[2/3] Resolving Task Embeddings...")
        for task in TASKS:
            print(f"  -> Task: {task}")
            frames = load_task_frames(task, seed=101) # Seed 0 for fetching fixed data frames
            train_df, test_df = frames["train"], frames["test"]
            smiles_col = "smiles" if "smiles" in train_df.columns else "SMILES"

            task_cache_dir = CACHE_ROOT / clean_name / task
            
            # Train and Test are always present
            train_x = get_cached_embeddings(model, tokenizer, device, train_df[smiles_col].tolist(), task_cache_dir / "train.npy")
            test_x  = get_cached_embeddings(model, tokenizer, device, test_df[smiles_col].tolist(), task_cache_dir / "test.npy")
            
            val_x = None
            if "val" in frames:
                val_x = get_cached_embeddings(model, tokenizer, device, frames["val"][smiles_col].tolist(), task_cache_dir / "val.npy")
                
            feature_cache[task] = {
                "train_x": train_x, "test_x": test_x, "val_x": val_x, "frames": frames
            }

        # FREE VRAM BEFORE XGBOOST
        del model
        del tokenizer
        torch.cuda.empty_cache()

        # 3. RUN XGBOOST SEEDS
        print("[3/3] Executing XGBoost Ensembles...")
        for task in TASKS:
            task_spec = get_task_spec(task)
            frames = feature_cache[task]["frames"]
            train_df, test_df = frames["train"], frames["test"]
            actual_smiles_col = "smiles" if "smiles" in test_df.columns else "SMILES"

            train_x = feature_cache[task]["train_x"]
            test_x = feature_cache[task]["test_x"]
            train_y = train_df["label"].to_numpy(dtype=np.int32)
            val_x = feature_cache[task]["val_x"]
            
            for seed in SEEDS:
                print(f"    -> Fitting {task} | Seed {seed}")
                output_dir = RUNS_ROOT / task / "xgboost_transformer" / clean_name / f"seed_{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Model Family identification
                model_family = "xgboost_transformer"
                model_variant = f"xgboost-{clean_name}"
                
                if val_x is not None:
                    # Fixed Train/Val/Test Strategy
                    val_y = frames["val"]["label"].to_numpy(dtype=np.int32)
                    xgb = fit_classification_with_val(train_x, train_y, val_x, val_y, seed)
                    test_pred = xgb.predict_proba(test_x)[:, 1]
                    metadata = {"strategy": "fixed_train_val_test", "fold_count": 1}
                else:
                    # 5-Fold Cross Validation Strategy (AmpHGT / CellPPD typically)
                    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    fold_predictions = []
                    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(train_x, train_y), start=1):
                        xgb = fit_classification_with_val(
                            train_x[fit_idx], train_y[fit_idx], 
                            train_x[val_idx], train_y[val_idx], 
                            seed + fold_idx
                        )
                        fold_predictions.append(xgb.predict_proba(test_x)[:, 1])
                    test_pred = np.mean(np.vstack(fold_predictions), axis=0)
                    metadata = {"strategy": "five_fold_test_ensemble", "fold_count": 5}

                # Evaluation & Save
                metrics = compute_classification_metrics(test_df["label"].to_numpy(dtype=np.int32), test_pred)
                
                prediction_frame = build_prediction_frame(
                    task_id=task, model_family=model_family, model_variant=model_variant,
                    seed=seed, split_id="test", sample_ids=test_df[task_spec.sample_id_column],
                    input_values=test_df[actual_smiles_col], true_targets=test_df["label"],
                    predictions=test_pred, prediction_type="probability", threshold=0.5,
                )
                
                metric_frame = build_metric_frame(
                    task_id=task, model_family=model_family, model_variant=model_variant,
                    seed=seed, split_id="test", metrics=metrics,
                    primary_metric_names={"mcc", "auroc", "f1", "r2", "rmse", "mae"},
                )

                payload = {
                    "task": task, "model_name": model_name, "seed": seed, "status": "ready"
                }

                (output_dir / "adapter_plan.json").write_text(json.dumps(payload, indent=2))
                write_baseline_outputs(
                    output_dir=output_dir,
                    prediction_frame=prediction_frame,
                    metric_frame=metric_frame,
                    adapter_metadata={**payload, **metadata},
                )


if __name__ == "__main__":
    main()