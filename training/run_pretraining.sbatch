#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --time=0-20:00:00
#SBATCH --partition=###YOUR_PARTITION_HERE###
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=28
#SBATCH --mem=2000000M
#SBATCH --job-name=YOUR_JOBNAME_HERE
#SBATCH --output=/your/dir/here/%j.output.log
#SBATCH --error=/your/dir/here/%j.error.log

# Activate the virtual environment
source .venv/bin/activate

# dataset options: 1:PubChem+LMSD+ESMAtlas, 2:PubChem+LMSD, 3:ESMAtlas
# --span for span masking, otherwise random masking will be used
# --tokenizer atomistic for atomistic tokenizer, otherwise the PeptideCLM-2 tokenizer will be used
# --model_size Small, Medium, Large
# --alpha and --beta are the weights for the loss function, default is 0.6 (MLM) and 0.4 (MTR)
# --warmup_steps is the number of warmup steps
# --weight_decay is the weight decay for the optimizer, default is 0.01

srun python training_MTR_v3.py \
    --total_steps 100000 \
    --learning_rate 3e-4 \
    --batch_size 64 \
    --num_workers 8 \
    --model_size Large \
    --alpha 0.6 \
    --beta 0.4 \
    --warmup_steps 5000 \
    --weight_decay 0.01 \
    --dataset_option 1 \
    --tokenizer kmer \
    --masking_percentage 0.25 \
    --span \
    --save_name Large_MLM-MTR_allData_kmerTok_25p_SpanMask_lr-3e-4
###############################################