import pandas as pd
import os
import itertools

from datasets import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoTokenizer
from models.MTR_model import MTR_model
from transformers import AutoTokenizer

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
    def __init__(self, model, common=None, learning_rate=1e-4):
        super(RegressionModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

        # int_layer_dim = 512
        int_layer_dim = common["embed_dim"]  # Use the same dimension as the model's embedding dimension

        # this uses an intermediate layer in the regression head
        self.intermediate_layer = nn.Linear(common["embed_dim"], int_layer_dim)  # Optional intermediate layer
        self.regression_head = nn.Linear(int_layer_dim, 1)  # Final regression layer to output a single value
        self.dropout = nn.Dropout(0.2)  # Optional dropout layer for regularization

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, mask=attention_mask)
        # use the bos token's representation as the input to the regression head
        # intermediate_output = self.intermediate_layer(outputs[0][:, 0, :])  # Use the first token's representation

        # pull the embeddings from the model output
        x = outputs[0]  # Get the sequence output from the model

        # Apply dropout to the embeddings if needed
        x = self.dropout(x)  # Apply dropout to the sequence output

        if attention_mask is not None:
            # Expand mask to match x's shape
            # Mask will now be (batch_size, seq_len, embed_dim)
            attention_mask = attention_mask.unsqueeze(-1)  # Add embedding dim: shape becomes (batch_size, seq_len, 1)
            # Apply the mask to the embeddings
            masked_x = x * attention_mask  # Shape remains (batch_size, seq_len, embed_dim)
            # Sum embeddings across the sequence dimension, considering only masked tokens
            sum_x = masked_x.sum(dim=1)  # Shape = (batch_size, embed_dim)
            # Sum the mask along sequence dimension to get the number of non-padded tokens
            attention_mask_sum = attention_mask.sum(dim=1).clamp(min=1e-6)  # Shape = (batch_size, 1)
            # Compute a mean for masked tokens
            mean_pool = sum_x / attention_mask_sum  # Shape = (batch_size, embed_dim)
        else:
            # No mask provided: Mean-pool over the entire sequence dimension
            mean_pool = x.mean(dim=1)  # Shape = (batch_size, embed_dim)

        # use mean pooled output instead of the first token's representation
        intermediate_output = self.intermediate_layer(mean_pool)  # Use the mean-pooled output
        intermediate_output = self.dropout(intermediate_output)  # Apply dropout to the intermediate output
        prediction = self.regression_head(intermediate_output)

        # If you want to use the sequence output directly without an intermediate layer, uncomment the following line:
        # prediction = self.regression_head(outputs[0][:, 0, :])  # Use the first token's representation

        return prediction

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['target']
        targets = targets.unsqueeze(1)
        prediction = self(input_ids, attention_mask)

        # MSE loss
        loss = nn.functional.mse_loss(prediction, targets)  # Mean Squared Error loss

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['target']
        targets = targets.unsqueeze(1)
        prediction = self(input_ids, attention_mask)

        # MSE loss
        loss = nn.functional.mse_loss(prediction, targets)  # Mean Squared Error loss

        self.log('val_loss', loss, on_epoch=True, prog_bar=False, sync_dist=False)
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prediction = self(input_ids, attention_mask)
        return {'prediction': prediction}
        
    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0),
        #     'monitor': 'val_loss'
        # }

        return optimizer #, [scheduler]

        # # Set the number of warmup steps
        # self.warmup_steps = 10  # Adjust this value based on your training

        # # Set up the warmup scheduler
        # scheduler = []  # Will hold the warmup + plateau schedulers
        # if self.warmup_steps > 0:
        #     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #         optimizer,
        #         start_factor=0.1,  # Start at 10% of initial learning rate
        #         total_iters=self.warmup_steps,  # Number of warmup steps
        #     )

        #     # ReduceLROnPlateau scheduler for post-warmup LR reduction
        #     plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,
        #         mode="min",    # Minimize the monitored metric (e.g., validation loss)
        #         patience=0,    # Wait 0 epochs without improvement before reducing LR
        #         factor=0.5,    # Reduce LR by multiplying it with this factor
        #     )

        #     # Combine warmup and plateau schedulers
        #     scheduler = [
        #         {
        #             'scheduler': warmup_scheduler,
        #             'interval': 'step',       # Warmup is updated every step
        #             'frequency': 1            # Apply scheduling every step
        #         },
        #         {
        #             'scheduler': plateau_scheduler,
        #             'interval': 'epoch',      # Plateau is updated every epoch
        #             'frequency': 1,
        #             'monitor': 'val_loss',    # Plateau monitors validation loss
        #         },
        #     ]

        # return [optimizer], scheduler

def main(model_name, file_path, batch_size, transfer_learning, save_dir, save_name, checkpoint=None, random_init=False):
    # Load the dataset
    df = pd.read_csv(file_path)

    # normalize the 'value' column (Does not seem to improve results)
    # df['value'] = (df['value'] - df['value'].mean()) / (df['value'].std() + 1e-8)  # Add a small constant to avoid division by zero

    tokenizer = AutoTokenizer.from_pretrained("aaronfeller/PeptideMTR")
    tokenizer.model_max_length = 2048
    vocab_size = tokenizer.vocab_size

    # Define model-specific configurations
    model_configs = {
        "MLM-MTR_small": { # 32M
            "common": dict(
                vocab_size=vocab_size,
                output_dim=vocab_size,
                max_seq_len=2048,
                ffn_hidden_dim=768,
                embed_dim=512,
                num_heads=8,
                num_blocks=14,
                num_tasks=1,  # default for MLM mode
                head_sizes=[99],  # will be set in multi-task mode
            ),
            "checkpoint": checkpoint,
            "learning_rate": 3e-4,
        },
        "MLM-MTR_medium": { # 114M is ~3x the size of 32M
            "common": dict(
                vocab_size=vocab_size,
                output_dim=vocab_size,
                max_seq_len=2048,
                ffn_hidden_dim=1024,
                embed_dim=768,
                num_heads=12,
                num_blocks=24,
                num_tasks=1,  # default for MLM mode
                head_sizes=[99],  # will be set in multi-task mode
            ),
            "checkpoint": checkpoint,
            "learning_rate": 1e-4,
        },
        "MLM-MTR_large": { # 337M is ~3x the size of 114M
            "common": dict(
                vocab_size=vocab_size,
                output_dim=vocab_size,
                max_seq_len=2048,
                ffn_hidden_dim=2048,
                embed_dim=1024,
                num_heads=16,
                num_blocks=32,
                num_tasks=1,  # default for MLM mode
                head_sizes=[99],  # will be set in multi-task mode
            ),
            "checkpoint": checkpoint,
            "learning_rate": 5e-5,
        },
    }

    # Load the appropriate configuration based on the model_name
    if model_name in model_configs:
        config = model_configs[model_name]
        mtr_model = MTR_model(**config["common"])
        checkpoint = config["checkpoint"]
        learning_rate = config["learning_rate"]
        common = config["common"]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    print(f"Processing {file_path} using architecture {model_name} with learning rate {learning_rate} and batch size {batch_size}")

    # set results list to store results for each test_fold
    results = []

    # Iterate through each fold in the DataFrame
    for test_fold in df['fold'].unique():

        print(f"Processing test fold {test_fold}...")
        ensemble_df = df[df['fold'] != test_fold]
        test_df = df[df['fold'] == test_fold]

        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
        test_dataset.set_format(type='torch', columns=['input_ids', 'target'])
        test_dataset = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=lambda x: collate_fn(x, tokenizer))

        for fold in ensemble_df['fold'].unique():

            train_df = ensemble_df[ensemble_df['fold'] != fold]
            val_df = ensemble_df[ensemble_df['fold'] == fold]

            # # bin and sample the training set to have the same number of samples per bin
            train_df = train_df.groupby(pd.cut(train_df['value'], bins=5)).apply(lambda x: x.sample(n=train_df.groupby(pd.cut(train_df['value'], bins=5)).size().max(), replace=True, random_state=42)).reset_index(drop=True)

            train_dataset = Dataset.from_pandas(train_df)
            train_dataset = train_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
            train_dataset.set_format(type='torch', columns=['input_ids', 'target'])

            val_dataset = Dataset.from_pandas(val_df)
            val_dataset = val_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
            val_dataset.set_format(type='torch', columns=['input_ids', 'target'])

            # create dataloaders
            train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: collate_fn(x, tokenizer))
            val_dataset = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=lambda x: collate_fn(x, tokenizer))

            # Early stopping callback
            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=4,  # Number of epochs with no improvement after which training will be stopped
                mode='min',
                verbose=True
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',  # Metric to monitor
                filename='best-checkpoint',  # Name of the checkpoint files
                save_top_k=1,  # Save only the best model
                mode='min',  # We want to minimize validation loss
            )

            # run_name = f"{model_name.replace('/', '-')}_fold_{fold}_external"
            # wandb_logger = WandbLogger(
            #     project="PeptideMTR",
            #     name=run_name,
            #     log_model=True,
            #     save_dir="checkpoints/"
            # )

            # load state dict from checkpoint
            state_dict = torch.load(checkpoint, map_location='cpu')['state_dict']

            # load the state dict into the model
            mtr_model.load_state_dict(state_dict, strict=False)

            if args.random_init:
            # randomly intialize the mtr_model
                for param in mtr_model.parameters():
                    if param.data.ndimension() > 1:  # Apply Xavier to weight tensors
                        nn.init.kaiming_uniform_(param)
                    else:  # Initialize biases as well
                        nn.init.zeros_(param)

            regression_model = RegressionModel(mtr_model, common=common, learning_rate=learning_rate)

            # randomly initialize the regression model to xavier uniform
            for param in itertools.chain(
                regression_model.intermediate_layer.parameters(),
                regression_model.regression_head.parameters()
            ):
                if param.data.ndimension() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

            # If transfer learning is enabled, leave the regression head trainable and freeze the rest of the model
            if transfer_learning:
                # set all parameters to .requires_grad = False
                for param in regression_model.parameters():
                    param.requires_grad = False
                # set the regression head parameters to .requires_grad = True
                # This allows the regression head to be trained while keeping the rest of the model frozen
                for param in regression_model.regression_head.parameters():
                    param.requires_grad = True
                for param in regression_model.intermediate_layer.parameters():
                    param.requires_grad = True
            else:
                # If transfer learning is not enabled, all parameters are trainable
                for param in regression_model.parameters():
                    param.requires_grad = True

            # Create Trainer instance
            trainer = pl.Trainer(
                callbacks=[early_stopping_callback, checkpoint_callback],
                max_epochs=10,
                accelerator='gpu',
                devices=1,
                precision='bf16-mixed',
                log_every_n_steps=100,
                # strategy='ddp_find_unused_parameters_true',
                val_check_interval=0.2,  # Validate every 0.2 epochs
                enable_progress_bar=False,
                gradient_clip_val=0.1,
            )

            # Start training
            print(f"Starting training for fold {test_fold}-{fold} with model {model_name} and learning rate {learning_rate}")
            # print trainable parameters
            trainable_params = [name for name, param in regression_model.named_parameters() if param.requires_grad]
            print(f"Trainable parameters length: {len(trainable_params)}")
            trainer.fit(regression_model, train_dataloaders=train_dataset, val_dataloaders=val_dataset)

            print(f"Starting evaluation...")
            # get the best model from the checkpoint
            # This will load the best model based on the validation loss
            best_checkpoint_path = checkpoint_callback.best_model_path  # Get the path of the best model
            state_dict = torch.load(best_checkpoint_path, map_location='cpu')['state_dict']

            # Reload the model for evaluation
            mtr_model.eval()
            regression_model = RegressionModel(mtr_model, common=common, learning_rate=0)
            regression_model.load_state_dict(state_dict)
            
            # freeze all parameters for evaluation
            for param in regression_model.parameters():
                param.requires_grad = False
                
            # Make predictions on the test set
            predicted_values = []

            # Ensure the model is in evaluation mode
            regression_model.eval()
            # model to gpu
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            regression_model.to(device)

            for batch in test_dataset:
                # put batch on the same device as the model
                batch = {k: v.to(device) for k, v in batch.items()}
                # Make predictions
                with torch.no_grad():
                    predictions = regression_model.predict_step(batch, batch_idx=None)['prediction']

                    # Squeeze the prediction if it's a 2D tensor and convert to a list
                    predicted_values.extend(predictions.squeeze().tolist())

            # Add predictions to the test DataFrame (which is added per fold)
            test_df.loc[:, f'prediction_{fold}'] = predicted_values

        # Store results for further analysis
        results.append(test_df)

    # Concatenate results for all folds
    final_results = pd.concat(results, ignore_index=True)

    final_results['mean_prediction'] = final_results.filter(like='prediction_').mean(axis=1)
    final_results['std_prediction'] = final_results.filter(like='prediction_').std(axis=1)

    # Save the results to a CSV file
    os.makedirs(save_dir, exist_ok=True)

    final_results.to_csv(f'{save_dir}/{save_name}.csv', index=False)
    print(f"Finished processing {model_name}")
    print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MTR model on external dataset")
    parser.add_argument('-i', '--file_path', type=str, default=None, choices=['stab', 'fibril', 'perm'], 
                        help='Path to the external dataset CSV file')
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
    args = parser.parse_args()

    if args.file_path is None:
        raise ValueError("Please provide a file path using --file_path argument.")
    elif args.file_path == 'stab':
        file_path = 'finetuning/stability_external.csv'
    elif args.file_path == 'fibril':
        file_path = 'finetuning/fibril_external.csv'
    elif args.file_path == 'perm':
        file_path = 'finetuning/perm_external.csv'

    batch_size = args.batch_size
    transfer_learning = args.transfer_learning
    model_name = args.model
    if model_name is None:
        raise ValueError("Please provide a model name using --model argument.")
    save_dir = args.save_dir

    main(model_name, file_path, batch_size, transfer_learning, save_dir, args.save_name, checkpoint=args.checkpoint, random_init=args.random_init)
