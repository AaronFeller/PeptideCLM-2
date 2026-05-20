import pandas as pd
import os
import itertools

from datasets import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoTokenizer, AutoModel

torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import argparse


# build script to tokenize the dataset
def tokenize_function(examples, tokenizer):
    # Tokenizing the SMILES strings
    tokenized_inputs = tokenizer(
        examples['SMILES'], padding=False, truncation=True, max_length=2048
    )
    
    # convert list of examples['value'] to float from long
    examples['value'] = [float(v) for v in examples['value']]

    # The tokenized_inputs must return lists for each key in the output dict
    return {
        'input_ids': tokenized_inputs['input_ids'],  # this should be a list
        'target': examples['value'],  # this should also be a list
    }

# collate function to set padding to longest sequence in the batch
def collate_fn(batch, tokenizer):
    # Extract input_ids and targets from the batch
    input_ids = [item['input_ids'] for item in batch]
    targets = [item['target'] for item in batch]

    # Pad input_ids to the longest sequence in the batch
    padded_input_ids = nn.utils.rnn.pad_sequence(
        [ids.detach().clone() for ids in input_ids], batch_first=True, padding_value=tokenizer.pad_token_id
    )

    # Convert padded_input_ids to a tensor
    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.int)

    # Create attention mask where 1 indicates a real token and 0 indicates padding
    attention_mask = (padded_input_ids != tokenizer.pad_token_id) # (B, T)

    # Convert targets to a tensor
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask,
        'target': targets_tensor,
    }


class RegressionModel(pl.LightningModule):
    def __init__(self, model, embed_dim=768, learning_rate=1e-4):
        super(RegressionModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Use the provided embedding dimension dynamically
        int_layer_dim = embed_dim 

        self.intermediate_layer = nn.Linear(embed_dim, int_layer_dim)  
        self.regression_head = nn.Linear(int_layer_dim, 1)  
        self.dropout = nn.Dropout(0.2)  

    def forward(self, input_ids, attention_mask=None):
        # 1. Call your custom model pass
        outputs = self.model(input_ids, mask=attention_mask)
        
        # 2. Extract the pre-calculated native mean pool directly from the MLMOutput object
        mean_pool = outputs.mean_pool  
        
        # 3. Route cleanly into your downstream regression layers
        intermediate_output = self.intermediate_layer(mean_pool)  
        intermediate_output = self.dropout(intermediate_output)  
        prediction = self.regression_head(intermediate_output)

        return prediction
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['target']
        targets = targets.unsqueeze(1)
        prediction = self(input_ids, attention_mask)

        loss = nn.functional.mse_loss(prediction, targets)  

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['target']
        targets = targets.unsqueeze(1)
        prediction = self(input_ids, attention_mask)

        loss = nn.functional.mse_loss(prediction, targets)  

        self.log('val_loss', loss, on_epoch=True, prog_bar=False, sync_dist=False)
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prediction = self(input_ids, attention_mask)
        return {'prediction': prediction}
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def main(
    model_name, file_path, batch_size, 
    transfer_learning, save_dir, save_name, 
    checkpoint=None, random_init=False, 
    seed=None, gpu_index=None
):
    if seed is None:
        raise ValueError("Please provide a seed using --seed.")
    pl.seed_everything(seed, workers=True)
    
    df = pd.read_csv(file_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = 2048
    
    # Load backbone dynamically via AutoModel
    backbone_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    # Extract structural embedding dimension directly from model configuration
    embed_dim = getattr(backbone_model.config, "hidden_size", getattr(backbone_model.config, "embed_dim", 768))

    # Apply heuristic learning rates based on model tier strings
    if "small" in model_name.lower():
        learning_rate = 3e-4
    elif "large" in model_name.lower():
        learning_rate = 5e-5
    else:
        learning_rate = 1e-4

    print(f"Processing {file_path} using architecture {model_name} with learning rate {learning_rate} and batch size {batch_size}")

    results = []

    # Iterate through each fold in the DataFrame
    for test_fold in df['fold'].unique():

        print(f"Processing test fold {test_fold}...")
        ensemble_df = df[df['fold'] != test_fold]
        test_df = df[df['fold'] == test_fold]

        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
        test_dataset.set_format(type='torch', columns=['input_ids', 'target'])
        
        # Increased num_workers to 16 to resolve data loading bottleneck warnings
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16, collate_fn=lambda x: collate_fn(x, tokenizer))

        for fold in ensemble_df['fold'].unique():

            train_df = ensemble_df[ensemble_df['fold'] != fold]
            val_df = ensemble_df[ensemble_df['fold'] == fold]

            train_df = train_df.groupby(pd.cut(train_df['value'], bins=5)).apply(lambda x: x.sample(n=train_df.groupby(pd.cut(train_df['value'], bins=5)).size().max(), replace=True, random_state=seed)).reset_index(drop=True)

            train_dataset = Dataset.from_pandas(train_df)
            train_dataset = train_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
            train_dataset.set_format(type='torch', columns=['input_ids', 'target'])

            val_dataset = Dataset.from_pandas(val_df)
            val_dataset = val_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
            val_dataset.set_format(type='torch', columns=['input_ids', 'target'])

            # Increased num_workers to 16 to maximize node core utilization
            train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=lambda x: collate_fn(x, tokenizer))
            val_dataset_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, collate_fn=lambda x: collate_fn(x, tokenizer))

            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=4,  
                mode='min',
                verbose=True
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',  
                filename='best-checkpoint',  
                save_top_k=1,  
                mode='min',  
            )

            # Load weights if a valid local checkpoint is supplied
            if checkpoint and os.path.exists(checkpoint):
                ckpt_data = torch.load(checkpoint, map_location='cpu')
                state_dict = ckpt_data['state_dict'] if 'state_dict' in ckpt_data else ckpt_data
                backbone_model.load_state_dict(state_dict, strict=False)

            if random_init:
                for param in backbone_model.parameters():
                    if param.data.ndimension() > 1:  
                        nn.init.kaiming_uniform_(param)
                    else:  
                        nn.init.zeros_(param)

            regression_model = RegressionModel(backbone_model, embed_dim=embed_dim, learning_rate=learning_rate)

            for param in itertools.chain(
                regression_model.intermediate_layer.parameters(),
                regression_model.regression_head.parameters()
            ):
                if param.data.ndimension() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

            if transfer_learning:
                for param in regression_model.parameters():
                    param.requires_grad = False
                for param in regression_model.regression_head.parameters():
                    param.requires_grad = True
                for param in regression_model.intermediate_layer.parameters():
                    param.requires_grad = True
            else:
                for param in regression_model.parameters():
                    param.requires_grad = True

            trainer = pl.Trainer(
                callbacks=[early_stopping_callback, checkpoint_callback],
                max_epochs=10,
                accelerator='gpu',
                devices=[int(gpu_index)],  
                precision='bf16-mixed',
                log_every_n_steps=100,
                val_check_interval=0.2,  
                enable_progress_bar=False,
                gradient_clip_val=0.1,
            )

            print(f"Starting training for fold {test_fold}-{fold} with model {model_name} and learning rate {learning_rate}")
            trainable_params = [name for name, param in regression_model.named_parameters() if param.requires_grad]
            print(f"Trainable parameters length: {len(trainable_params)}")
            trainer.fit(regression_model, train_dataloaders=train_dataset_loader, val_dataloaders=val_dataset_loader)

            print(f"Starting evaluation...")
            best_checkpoint_path = checkpoint_callback.best_model_path  
            state_dict = torch.load(best_checkpoint_path, map_location='cpu')['state_dict']

            backbone_model.eval()
            regression_model = RegressionModel(backbone_model, embed_dim=embed_dim, learning_rate=0)
            regression_model.load_state_dict(state_dict)
            
            for param in regression_model.parameters():
                param.requires_grad = False
                
            predicted_values = []
            regression_model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            regression_model.to(device)

            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    predictions = regression_model.predict_step(batch, batch_idx=None)['prediction']
                    predicted_values.extend(predictions.view(-1).tolist())
            
            # FIXED INDENTATION: Assigned predictions outside the batch loop to prevent partial overwrite bugs
            test_df.loc[:, f'prediction_{fold}'] = predicted_values
            
        results.append(test_df)

    final_results = pd.concat(results, ignore_index=True)

    final_results['mean_prediction'] = final_results.filter(like='prediction_').mean(axis=1)
    final_results['std_prediction'] = final_results.filter(like='prediction_').std(axis=1)

    os.makedirs(save_dir, exist_ok=True)
    final_results.to_csv(f'{save_dir}/{save_name}.csv', index=False)
    print(f"Finished processing {model_name}")
    print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MTR model on external dataset")
    parser.add_argument('-i', '--file_path', type=str, default=None,
                        help='Named dataset alias (stab, fibril, perm) or a direct CSV path')
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Direct CSV path overriding --file_path aliases')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--transfer_learning', action='store_true', 
                        help='Use transfer learning from a pre-trained model')
    parser.add_argument('--model', type=str, default=None, 
                        help='Model architecture to use for fine-tuning')
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='Directory to save the fine-tuned model and results')
    parser.add_argument('--save_name', type=str, default='finetuning_results',
                        help='Filename to save the results of the fine-tuning process')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a specific checkpoint to load for fine-tuning')
    parser.add_argument('--random_init', action='store_true',
                        help='Randomly initialize the MTR model instead of loading a pre-trained checkpoint')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for fold ordering and training reproducibility')
    parser.add_argument('--gpu_index', type=int, default=0,
                        help='Explicit GPU index to isolate execution')
    args = parser.parse_args()
    
    if args.data_csv is not None:
        file_path = args.data_csv
    elif args.file_path is None:
        raise ValueError("Please provide a file path using --file_path argument.")
    elif args.file_path == 'stab':
        file_path = 'finetuning/stability_external.csv'
    elif args.file_path == 'fibril':
        file_path = 'finetuning/fibril_external.csv'
    elif args.file_path == 'perm':
        file_path = 'finetuning/perm_external.csv'
    elif args.file_path.endswith('.csv'):
        file_path = args.file_path
    else:
        raise ValueError("Unsupported --file_path value. Use stab, fibril, perm, or a direct CSV path.")

    batch_size = args.batch_size
    transfer_learning = args.transfer_learning
    model_name = args.model
    if model_name is None:
        raise ValueError("Please provide a model name using --model argument.")
    save_dir = args.save_dir

    main(model_name, file_path, batch_size, transfer_learning, save_dir, 
         args.save_name, checkpoint=args.checkpoint, random_init=args.random_init, 
         seed=args.seed, gpu_index=args.gpu_index)