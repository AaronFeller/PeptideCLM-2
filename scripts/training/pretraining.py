print("Starting the script with imports...")
import warnings
import os
import copy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
# import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer
from model.MTR_model import MTR_model, pl_model
from rdkit import Chem
import lightning as pl

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')

# Environment setup
print("Enabling Flash SDP for better performance.")
torch.backends.cuda.enable_flash_sdp(True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')

def ensure_tensor(data):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data
    try:
        return torch.tensor(np.array(data), dtype=torch.float32)
    except Exception as e:
        print(f"Error converting to tensor: {e}")
        return None

def load_model(vocab_size, model_size):
    # Check for valid model size
    if model_size not in ["Small", "Medium", "Large", "Huge"]:
        raise ValueError(f"Invalid model size: {model_size}. Choose from 'Small', 'Medium', 'Large', or 'Huge'.")

    # Define model configurations
    model_configs = {
        "Small": {"ffn_hidden_dim": 768, "embed_dim": 512, "num_heads": 8, "num_blocks": 14}, #31.65M
        "Medium": {"ffn_hidden_dim": 1024, "embed_dim": 768, "num_heads": 12, "num_blocks": 24}, #113.96M
        "Large": {"ffn_hidden_dim": 2048, "embed_dim": 1024, "num_heads": 16, "num_blocks": 32}, #336.54M
        "Huge": {"ffn_hidden_dim": 2048, "embed_dim": 1280, "num_heads": 20, "num_blocks": 36}, #520.32M
    }

    # Common parameters for all model sizes
    common = {
        "vocab_size": vocab_size,
        "output_dim": vocab_size,
        "max_seq_len": 2048,
        "num_tasks": 1,  # Default for MLM mode
        "head_sizes": [99],  # Placeholder for multi-task mode
        **model_configs[model_size]  # Merge size-specific configs
    }

    # Instantiate and return the model
    return MTR_model(**common)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            'smiles': row['smiles'],
            'rdkit_descs': row.drop(['smiles', 'split', 'source']).values.tolist()
        }
    
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, tokenizer=None, train_batch_size=64, val_batch_size=256, num_workers=8, dataset_option=4):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dataset_option = dataset_option

        # First, load all your data
        self.load_data()  # 
        # Sample the data here once loading is complete
        self.sample_data()

    def load_data(self):
        # Load the full datasets
        if self.dataset_option == 1:
            self.train_pubchem_data = pq.read_table(f"{self.data_dir}/train_pubchem.parquet").to_pandas()
            self.train_lmsd_data = pq.read_table(f"{self.data_dir}/train_lmsd.parquet").to_pandas()
            self.train_esm_data = pq.read_table(f"{self.data_dir}/train_esm.parquet").to_pandas()
        elif self.dataset_option == 2:
            self.train_pubchem_data = pq.read_table(f"{self.data_dir}/train_pubchem.parquet").to_pandas()
            self.train_lmsd_data = pq.read_table(f"{self.data_dir}/train_lmsd.parquet").to_pandas()
        elif self.dataset_option == 3:
            self.train_esm_data = pq.read_table(f"{self.data_dir}/train_esm.parquet").to_pandas()

    def sample_data(self):
        # Now that data has been loaded, sample the datasets as required
        if self.dataset_option == 1:
        # Train with PubChem, LMSD, and ESM
            self.train_pubchem_data = self.train_pubchem_data.sample(n=10_000_000)  # Sample 10,000,000 rows from the pubchem data
            self.train_lmsd_data = self.train_lmsd_data.sample(n=5 * len(self.train_lmsd_data), replace=True)  # Upsample
            self.train_pubchem_dataset = CustomDataset(self.train_pubchem_data)
            self.train_lmsd_dataset = CustomDataset(self.train_lmsd_data)
            self.train_esm_dataset = CustomDataset(self.train_esm_data)
        # Train with PubChem and LMSD
        elif self.dataset_option == 2:
            self.train_pubchem_data = self.train_pubchem_data.sample(n=10_000_000)  # Sample 10,000,000 rows from the pubchem data
            self.train_lmsd_data = self.train_lmsd_data.sample(n=5 * len(self.train_lmsd_data), replace=True)  # Upsample
            self.train_pubchem_dataset = CustomDataset(self.train_pubchem_data)
            self.train_lmsd_dataset = CustomDataset(self.train_lmsd_data)
        # Train with ESM only
        elif self.dataset_option == 3:
            self.train_esm_dataset = CustomDataset(self.train_esm_data)

    def train_dataloader(self):
        # Combine the datasets using ChainDataset
        if self.dataset_option == 1:
            datasets = [self.train_pubchem_dataset, self.train_lmsd_dataset, self.train_esm_dataset]
        elif self.dataset_option == 2:
            datasets = [self.train_pubchem_dataset, self.train_lmsd_dataset]
        elif self.dataset_option == 3:
            datasets = [self.train_esm_dataset]

        # If you're using IterableDataset, ensure to handle it accordingly
        combined_dataset = CustomDataset(pd.concat([dataset.data for dataset in datasets], ignore_index=True))

        return DataLoader(
            combined_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, is_training=True),
        )

    def val_dataloader(self):
        val_data = pq.read_table(f"{self.data_dir}/validation.parquet").to_pandas()
        # subsample to 20000 rows
        val_data = val_data.sample(n=20_000)
        val_dataset = CustomDataset(val_data)

        return DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, is_training=False),
        )

def collate_fn(batch, tokenizer, is_training=False):
    """
    Unified collate function for both training and validation
    """
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    
    # Extract data from batch
    smiles = [item['smiles'] for item in batch]
    rdkit_descs = [item['rdkit_descs'] for item in batch]
    rdkit_descs = ensure_tensor(rdkit_descs)

    input_ids = []
    labels = []
    
    # Process each SMILES string
    for smile in smiles:
        if is_training:
            # Randomize SMILES during training
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    continue
                smile = Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=True)
            except Exception as e:
                print(f"Error processing SMILES '{smile}': {e}")
                continue
        
        # Tokenize
        encoded = tokenizer(smile, add_special_tokens=False, truncation=True, max_length=2046)
        input_ids.append(encoded['input_ids'])
        labels.append(copy.deepcopy(encoded['input_ids']))
    
    if not input_ids:  # Handle empty batch
        return None
    
    # Apply masking
    masking_percentage = args.masking_percentage

    # Span masking logic
    if args.span == True:
        average_span_length = 3.5  # Average length of spans
        stddev_span_length = 1.0  # Standard deviation for Gaussian distribution of spans

        for i in range(len(input_ids)):
            seq_length = len(input_ids[i])
            num_tokens_to_mask = int(seq_length * masking_percentage)  # Total tokens to mask
            masked_tokens_count = 0  # Counter for masked tokens
            masked_positions = set()  # Track masked positions to avoid overlaps

            while masked_tokens_count < num_tokens_to_mask:
                # Sample span length from a Gaussian distribution
                span_length = max(1, int(np.random.normal(average_span_length, stddev_span_length)))  # Ensure span length >= 1
                if masked_tokens_count + span_length > num_tokens_to_mask:
                    span_length = num_tokens_to_mask - masked_tokens_count  # Adjust if span overshoots

                # Randomly pick a starting position for the span
                start_pos = torch.randint(0, seq_length, (1,)).item()
                end_pos = min(seq_length, start_pos + span_length)

                # Check for overlap
                for pos in range(start_pos, end_pos):
                    if pos in masked_positions:
                        end_pos = start_pos  # Adjust end_pos to exclude overlap
                        break

                # Record positions and apply masking
                masked_positions.update(range(start_pos, end_pos))
                for pos in range(start_pos, end_pos):
                    input_ids[i][pos] = mask_token_id

                masked_tokens_count += (end_pos - start_pos)

    # If not using span masking, apply random masking
    else:
        for i in range(len(input_ids)):
            seq_length = len(input_ids[i])
            mask_positions = torch.randperm(seq_length)[:int(seq_length * masking_percentage)]
            for pos in mask_positions:
                input_ids[i][pos] = mask_token_id
    
    # Add special tokens and pad
    max_length = max(len(seq) for seq in input_ids) + 2

    if args.tokenizer == 'kmer':
        input_ids = [
            [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id] + [pad_token_id] * (max_length - len(ids) - 2)
            for ids in input_ids
        ]
        labels = [
            [tokenizer.cls_token_id] + lbls + [tokenizer.sep_token_id] + [-100] * (max_length - len(lbls) - 2)
            for lbls in labels
        ]

    elif args.tokenizer == 'atomistic':
        input_ids = [
            [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id] + [pad_token_id] * (max_length - len(ids) - 2)
            for ids in input_ids
        ]
        labels = [
            [tokenizer.bos_token_id] + lbls + [tokenizer.eos_token_id] + [-100] * (max_length - len(lbls) - 2)
            for lbls in labels
        ]
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    pad_mask = (input_ids != pad_token_id) # (B, T)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pad_masking": pad_mask,  # (B, T)
        "rdkit_descs": rdkit_descs
    }

###################################
########## MAIN FUNCTION ##########
###################################

def main(args=None):
    if args is None:
        # cancel run
        print("No arguments provided. Exiting the script.")
        return
    print("Starting the training process...")
    
    # Configuration
    config = {
        'model_size': args.model_size,  # Size of the model to train
        'learning_rate': args.learning_rate,  # Learning rate for the optimizer
        'total_steps': args.total_steps,  # Total number of training steps
        'warmup_steps': args.warmup_steps,  # Number of warmup steps
        'weight_decay': args.weight_decay,  # Weight decay for the optimizer
        'batch_size': args.batch_size,  # Batch size for training
        'num_workers': args.num_workers,  # Number of workers for data loading
        'alpha': args.alpha,  # Alpha parameter for the model
        'beta': args.beta,  # Beta parameter for the model
    }
    
    # Load tokenizer
    if args.tokenizer == 'kmer':
        tokenizer = AutoTokenizer.from_pretrained("aaronfeller/PeptideMTR")
    elif args.tokenizer == 'atomistic':
        tokenizer = AutoTokenizer.from_pretrained("novonordisk-red/PubChemBERT-large")

    tokenizer.model_max_length = 2048
    
    if args.model_size == "Huge":
        training_strategy = 'fsdp'  # Use more minibatches for Huge models
        minibatches = 1
    elif args.model_size == "Large":
        training_strategy = 'ddp'
        minibatches = 4
    elif args.model_size == "Medium":
        training_strategy = 'ddp'
        minibatches = 2
    elif args.model_size == "Small":
        training_strategy = 'ddp'
        minibatches = 1

    # adjust configuration for large models
    config['batch_size'] = int(config['batch_size']/minibatches)  # Reduce batch size for large models to fit in memory
    accumulate_grad_batches = minibatches  # Accumulate gradients to simulate larger batch size

    print(f"Configuration: {config}")
    run_name = args.save_name

    # Setup data module
    data_module = CustomDataModule(
        data_dir='data/ARVF_dataset',
        tokenizer=tokenizer,
        train_batch_size=config['batch_size'],
        val_batch_size=config['batch_size'] * 16,  # Validation batch size can be larger
        num_workers=config['num_workers'],
        dataset_option=args.dataset_option  # Pass the dataset option
    )
    
    # Load model
    model = load_model(vocab_size=tokenizer.vocab_size, model_size=config['model_size'])
    model = pl_model(
        model,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        total_steps=config['total_steps'],
        alpha=config['alpha'],
        beta=config['beta'],
    )
    
    # Setup logging and callbacks
    wandb_logger = WandbLogger(
        project="PeptideMTR",
        name=run_name,
        log_model=True,
        save_dir="checkpoints/"
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/{run_name}',
            filename=f'{config["model_size"]}-{{epoch:02d}}-{{step:.0f}}-{{val_loss:.3f}}',
            save_top_k=3,
            mode='min',
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            verbose=True,
            mode='min',
            patience=1_000_000
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup trainer
    trainer = Trainer(
        max_steps=config['total_steps'],
        log_every_n_steps=100,
        accumulate_grad_batches=accumulate_grad_batches,
        accelerator="gpu",
        devices=-1,
        precision='bf16-mixed',
        strategy=training_strategy,
        val_check_interval=1000*minibatches,  # Validate every 1000 steps, adjusted for minibatches
        logger=wandb_logger,
        enable_checkpointing=True,
        default_root_dir="checkpoints/",
        enable_progress_bar=False,
        callbacks=callbacks,
        gradient_clip_val=0.1,  # Clip gradients to avoid exploding gradients
    )
    
    # Train
    trainer.fit(model, data_module)

    print("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MTR model with specified configurations.")
    parser.add_argument('--model_size', type=str, choices=['Small', 'Medium', 'Large', 'Huge'], default='Small', help="Size of the model to train.")
    parser.add_argument('--total_steps', type=int, default=100_000, help="Total number of training steps.")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument('--alpha', type=float, default=1, help="Alpha parameter for the model.")
    parser.add_argument('--beta', type=float, default=0, help="Beta parameter for the model.")
    parser.add_argument('--warmup_steps', type=int, default=5000, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for the optimizer.")
    parser.add_argument('--dataset_option', type=int, choices=[1, 2, 3], default=2, help="Choose dataset option: 1 for pubchem, 2 for pubchem+ESMAtlas, 3 for ESMAtlas.")
    parser.add_argument('--tokenizer', type=str, default='aaronfeller/PeptideMTR', help="Tokenizer to use for the model.")
    parser.add_argument('--masking_percentage', type=float, default=0.15, help="Percentage of tokens to mask during training.")
    parser.add_argument('--span', action='store_true', help="Use span masking instead of random masking.")
    parser.add_argument('--save_name', type=str, default='MTR_training', help="Name for saving the model checkpoints.")
    args = parser.parse_args()

    main(args)