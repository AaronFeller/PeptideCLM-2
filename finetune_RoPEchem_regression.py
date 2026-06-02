# print("You're starting training!")

# Timing
import datetime
# print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))

# Define the main function
def main():
    # set up arguments ##############################################################
    
    def str_to_list(s):
        return [int(i) for i in s.split(',')]
    
    import argparse
    # -lr $lr -dr $do -wd $wd -bs $bs -lrd $lrd
    parser = argparse.ArgumentParser(description='Train a Transformer model on a peptide dataset')
    parser.add_argument('-d', '--directory', type=str, required=True, 
                        help='Directory name.')
    parser.add_argument('-m', '--model', type=int, required=True, choices=[0, 1, 2],
                        help='0 is pep+sm, 1 is pep, 2 is sm')
    parser.add_argument('-f', '--fold', type=int, required=True,
                        help='Fold number.')
    parser.add_argument('-ho', '--holdout', type=int, required=True,
                        help='Holdout number.')
    parser.add_argument('-g', '--gpu', type=str_to_list, required=True,
                        help='Comma-separated GPU numbers. E.g., "0,1,2,3" or a subset like "0,2,4".')
    # parser.add_argument('-t', '--training_data', type=str, required=True,
    #                     help='Path to input in SMILES format')
    # parser.add_argument('-v', '--validation_data', type=str, required=True,
    #                     help='Path to validation in SMILES format')
    parser.add_argument('-lr', '--learning_rate', default=1e-6, required=False,
                        help='Learning rate (default is 1e-6)')
    parser.add_argument('-dr', '--dropout_rate', default=0.15, required=False,
                        help='Dropout rate (default is 0.15)')
    parser.add_argument('-wd', '--weight_decay', default=0.001, required=False,
                        help='Weight decay (default is 0.001)')
    parser.add_argument('-bs', '--batch_size', required=True,
                        help='Batch size')
    parser.add_argument('-lrd', '--lr_decay', default=False, required=False,
                        help='Learning rate decay (default is False)')
    parser.add_argument('-hl', '--hidden_layers', default=6, required=False,
                        help='Number of hidden layers (default is 6)')
    parser.add_argument('-ah', '--attention_heads', default=12, required=False,
                        help='Number of attention heads (default is 12)')
    parser.add_argument('-es', '--embedding_size', default=768, required=False,
                        help='Embedding size (default is 768)')
    parser.add_argument('-hs', '--hidden_size', default=768, required=False,
                        help='Hidden size (default is 768)')
    parser.add_argument('-is', '--intermediate_size', default=3072, required=False,
                        help='Intermediate size (default is 3072)')
    parser.add_argument('-w', '--workers', default=0, required=False,
                        help='Number of workers for data loader (default is 0)')
    # parser.add_argument('-f', '--fold', required=True)

    args = parser.parse_args()
    
    # Load packages
    # print("Loading libraries...")

    # Set up environment variables
    import os

    # Import libraries
    from collections import OrderedDict
    import argparse
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    # from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    # from pytorch_lightning.strategies import FSDPStrategy
    # from lightning.pytorch.profilers import SimpleProfiler
    from torchvision.transforms import Compose
    from roformer import RoFormerForSequenceClassification, RoFormerConfig #, RoFormerForMaskedLM,
    from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
    # import deepchem as dc

    # Import local functions and classes
    from functions.data_funcs_finetune import reset_parameters
    # from models.my_model_classification import pepLM
    from models.my_model_regression import pepLM
    from dataloader.dataloader_finetune import SMILES_Dataset, SMILES_to_input

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.empty_cache()
    
    # Load Tokenizer ####################################################################
    vocab_file = 'tokenizer/new_vocab.txt'
    splits_file = 'tokenizer/new_splits.txt'
    
    tokenizer = SMILES_SPE_Tokenizer(vocab_file, splits_file)
    
    # Data loader for kfold models ###############################################################
    # new loader for clusters below
    cluster_list = [1, 2, 3, 4, 5, 6]
    cluster_list.remove(args.holdout)
    cluster_test = cluster_list[args.fold - 1]
    cluster_train = [cluster for cluster in cluster_list if cluster != cluster_test]
    
    # load test file
    test = pd.read_csv(f'revision/training_data/clusters/cluster_{cluster_test}.csv', header=0, names=['ids', 'y'], sep=',')
    # make empty df

    train = pd.DataFrame(columns=['ids', 'y'])
    for cluster in cluster_train:
        train_add = pd.read_csv(f'revision/training_data/clusters/cluster_{cluster}.csv', header=0, names=['ids', 'y'], sep=',')
        train = pd.concat([train, train_add], ignore_index=True)

    
    # training_file = f'revision/training_data/kmeans_regression/train_{args.fold}.csv'
    # test_file = f'revision/training_data/kmeans_regression/cluster_{args.fold}.csv'

    # train test split
    # train = pd.read_csv(training_file, header=0, names=['ids', 'y'], sep=',')
    # test = pd.read_csv(test_file, header=0, names=['ids', 'y'], sep=',')


    composed = Compose([SMILES_to_input(tokenizer, int(f'{args.embedding_size}'))])
    train_dataset = SMILES_Dataset(train, transform=composed)
    val_dataset = SMILES_Dataset(test, transform=composed)

    train_loader = DataLoader(train_dataset, num_workers=int(f'{args.workers}'), batch_size=int(f'{args.batch_size}'), shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=int(f'{args.workers}'), batch_size=int(f'{args.batch_size}'), shuffle=False)
    
    # Data loader ###############################################################
    # training_file = f'training_data/kmeans_regression/train_{args.fold}.csv'
    # test_file = f'training_data/kmeans_regression/test_{args.fold}.csv'

    # train = pd.read_csv(training_file, header=0, names=['ids', 'y'], sep=',')
    # test = pd.read_csv(test_file, header=0, names=['ids', 'y'], sep=',')

    # chemical_file = 'training_data/regression_data/ncats_regression_FIXED.csv'
    # chemical = pd.read_csv(chemical_file, header=0, names=['ids', 'y'], sep=',')

    # combine train and test
    # train = pd.concat([train, test, chemical], ignore_index=True)
    
    # composed = Compose([SMILES_to_input(tokenizer, int(f'{args.embedding_size}'))])
    
    # train_dataset = SMILES_Dataset(train, transform=composed)
    # val_dataset = SMILES_Dataset(test, transform=composed)
    
    # train_loader = DataLoader(train_dataset, num_workers=int(f'{args.workers}'), batch_size=int(f'{args.batch_size}'), shuffle=True)
    # val_loader = DataLoader(val_dataset, num_workers=int(f'{args.workers}'), batch_size=int(f'{args.batch_size}'), shuffle=False)

    
    
    # training_file = f'training_data/kmeans_regression/train_1.csv'
    # test_file = f'training_data/kmeans_regression/test_1.csv'
    # validation_file = f'training_data/kmeans_regression/kmeans_regression_validation.csv'
    # chemical_file = 'training_data/regression_data/ncats_regression_FIXED.csv'
    
    # train = pd.read_csv(training_file, header=0, names=['ids', 'y'], sep=',')
    # test = pd.read_csv(test_file, header=0, names=['ids', 'y'], sep=',')
    # chemical = pd.read_csv(chemical_file, header=0, names=['ids', 'y'], sep=',')
    # # combine train and test
    # train = pd.concat([train, test, chemical], ignore_index=True)
    
    # validate = pd.read_csv(validation_file, header=0, names=['ids', 'y'], sep=',')

    # composed = Compose([SMILES_to_input(tokenizer, int(f'{args.embedding_size}'))])
    
    # train_dataset = SMILES_Dataset(train, transform=composed)
    # val_dataset = SMILES_Dataset(validate, transform=composed)
    
    # train_loader = DataLoader(train_dataset, num_workers=int(f'{args.workers}'), batch_size=int(f'{args.batch_size}'), shuffle=True)
    # val_loader = DataLoader(val_dataset, num_workers=int(f'{args.workers}'), batch_size=int(f'{args.batch_size}'), shuffle=False)
        
    # Model parameters ###############################################################
    # save a log of the model parameters
    if not os.path.exists(f'checkpoints/{args.directory}'):
        os.makedirs(f'checkpoints/{args.directory}')
    with open(f'checkpoints/{args.directory}/model_params.txt', 'w') as f:
        f.write(f'Model training started at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n') 
        f.write(f'learning_rate: {args.learning_rate}\n')
        f.write(f'dropout_rate: {args.dropout_rate}\n')
        f.write(f'batch_size: {args.batch_size}\n')
        f.write(f'vocab_size: {tokenizer.vocab_size}\n')
        f.write(f'num_hidden_layers: {args.hidden_layers}\n')
        f.write(f'num_attention_heads: {args.attention_heads}\n')
        f.write(f'embedding_size: {args.embedding_size}\n')
        f.write(f'hidden_size: {args.hidden_size}\n')
        f.write(f'intermediate_size: {args.intermediate_size}\n')
        f.write(f'workers: {args.workers}\n')

    # set model configuration
    config = RoFormerConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_size=int(f'{args.embedding_size}'),
        max_position_embeddings=int(f'{args.embedding_size}'),
        num_hidden_layers=int(f'{args.hidden_layers}'),
        num_attention_heads=int(f'{args.attention_heads}'),
        hidden_size=int(f'{args.hidden_size}'),
        intermediate_size=int(f'{args.intermediate_size}'),
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        is_decoder=False,
        num_labels=1,
        hidden_dropout_prob=float(f'{args.dropout_rate}'),
        attention_probs_dropout_prob=float(f'{args.dropout_rate}'),
        problem_type='regression'
    )
    
    # model = RoFormerForMaskedLM(config=config)
    model = RoFormerForSequenceClassification(config=config)
    model = reset_parameters(model)
    # load checkpoint
    if args.model == 0:
        model_checkpoint_path = 'checkpoints/2024-05-28_FirstPepLM_onRandom_peps/last.ckpt'
    elif args.model == 1:
        model_checkpoint_path = 'checkpoints/2024-06-28_pretrain_peptide_data_continue/last.ckpt'
    elif args.model == 2:
        model_checkpoint_path = 'checkpoints/2024-06-21_pretrain_chem_data/last.ckpt'
    
    checkpoint = torch.load(model_checkpoint_path,
                            map_location='cpu')

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[6:] # remove `model.`
    #     name = name.replace('roberta', 'roformer')
        new_state_dict[name] = v

    # remove these keys: Unexpected key(s) in state_dict: "cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias"
    new_state_dict.pop('cls.predictions.bias')
    new_state_dict.pop('cls.predictions.transform.dense.weight')
    new_state_dict.pop('cls.predictions.transform.dense.bias')
    new_state_dict.pop('cls.predictions.transform.LayerNorm.weight')
    new_state_dict.pop('cls.predictions.transform.LayerNorm.bias')
    new_state_dict.pop('cls.predictions.decoder.weight')
    new_state_dict.pop('cls.predictions.decoder.bias')

    # load weights from checkpoint
    model.load_state_dict(new_state_dict, strict=False)


    # freeze layers for transfer learning
    # for name, param in model.named_parameters():
    #     if 'roformer.embeddings' in name:
    #         param.requires_grad = False
    #     if 'roformer.encoder.layer.0' in name:
    #         param.requires_grad = False
    #     if 'roformer.encoder.layer.1' in name:
    #         param.requires_grad = False
    #     if 'roformer.encoder.layer.2' in name:
    #         param.requires_grad = False
    #     if 'roformer.encoder.layer.3' in name:
    #         param.requires_grad = False
    #     if 'roformer.encoder.layer.4' in name:
    #         param.requires_grad = False
    #     if 'roformer.encoder.layer.5' in name:
    #         param.requires_grad = False
    #     if 'classifier' in name:
    #         param.requires_grad = True
    
    # Count model parameters
    # print(f"Number of parameters: {count_parameters(model)/1e6:.2f}M")
    
    pepLM_model = pepLM(model, learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), use_scheduler=args.lr_decay)
    
    # Training ###############################################################
    # print("Training model...")
    # print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
    
    torch.set_float32_matmul_precision('high') # medium, high, or highest available (medium for speed, highest for accuracy)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mse',
        dirpath=f'checkpoints/{args.directory}',
        filename='PepLM_kfold_regression-{epoch:02d}-{step:.0f}-{val_mse:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # profiler = SimpleProfiler(dirpath=f'checkpoints/{args.directory}', filename='prof_logs')

    # early_stop_callback = EarlyStopping(
    #     monitor='val_mse',
    #     mode='min',
    #     min_delta=0.001,
    #     patience=20
    # )
    
    # strategy = FSDPStrategy(
    #     use_orig_params=True
    # )
    
    # make gpu list from arg.gpu
    
    
    
    trainer = pl.Trainer(
        # max_epochs=100,
        max_steps=10000,
        log_every_n_steps=10,
        accumulate_grad_batches=1,
        # gradient_clip_val=1.0,
        # val_check_interval=0.1,
        check_val_every_n_epoch=1,
        accelerator='gpu',
        devices=args.gpu, # [4,5,6,7],#-1, # for all GPUs
        enable_checkpointing=True,
        default_root_dir=f'checkpoints/{args.directory}',
        callbacks= [checkpoint_callback, lr_monitor], # [early_stop_callback, checkpoint_callback, lr_monitor],
        precision='16-mixed',
        strategy=FSDPStrategy(sharding_strategy='SHARD_GRAD_OP',
                                state_dict_type='full',), # 'ddp',#'fsdp',
        # profiler= profiler
        reload_dataloaders_every_n_epochs=5,
        )
    
    trainer.fit(
        model=pepLM_model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )
    
    print("Current Time =", datetime.datetime.now().strftime("%H:%M:%S"))
    print("Done!")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()