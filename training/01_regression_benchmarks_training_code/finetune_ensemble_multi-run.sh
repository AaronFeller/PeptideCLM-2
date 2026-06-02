#!/usr/bin/env bash

# make variable for date time without spaces
date_time=$(date +%Y-%m-%d_%H-%M-%S)

for iter in 1; do

    for model in MLM-MTR_small; do
        # # random init
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'random_init_round${iter}' --checkpoint 'checkpoints/Small-MTR_KmerToks_PubChem-ESMAtlas_20percent-spanMask/Small-epoch=02-step=99274-val_loss=0.124.ckpt' --random_init"

        # # MLM only
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'MLM_only_round${iter}' --checkpoint 'checkpoints/MLM_only_Small_Kmers_lr-0.0003_bs-64/Small-epoch=01-step=93550-val_loss=0.055.ckpt'"

        # # deepchem tokenizer
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'deepchem_tokenizer_round${iter}' --checkpoint 'checkpoints/atomistic_tokenizer_Small_Kmers_lr-0.0003_bs-64/Small-epoch=01-step=98550-val_loss=0.012.ckpt'"

        # # random_masking 25% ablation 2
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'MLM-MTR_random_masking_round${iter}' --checkpoint 'checkpoints/data_ablation_2_MLM-MTR_Small_Kmers_lr-0.0003_bs-64/Small-epoch=02-step=99274-val_loss=0.100.ckpt'"

        # # span_masking 25% ablation 2
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'MLM-MTR_span_masking_round${iter}' --checkpoint 'checkpoints/SpanMasking_Small_Kmers_lr-0.0003_bs-64/Small-epoch=02-step=99274-val_loss=0.176.ckpt'"

        # MTR only (nomasking)
        sbatch --time=0-08:00:00 \
            --partition=cu_0001 \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=20 \
            --mem=50000M \
            --job-name=novo_clm_finetune_${model} \
            --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
            --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
            --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'MTR_only_no_masking_round${iter}' --checkpoint 'checkpoints/Small-MTR-only_noMask_lr-3e-4/last.ckpt'"

        # # MTR only with span_masking 
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'MTR_only_span_masking_round${iter}' --checkpoint 'checkpoints/Small-MTR-only_25p-spanMask_lr-3e-4/last.ckpt'"

        # # MLM-MTR 15% span_masking 
        # sbatch --time=0-08:00:00 \
        #     --partition=cu_0001 \
        #     --nodes=1 \
        #     --ntasks-per-node=1 \
        #     --gpus-per-node=1 \
        #     --cpus-per-task=20 \
        #     --mem=50000M \
        #     --job-name=novo_clm_finetune_${model} \
        #     --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
        #     --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
        #     --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name 'MLM-MTR_15p_span_masking_round${iter}' --checkpoint 'checkpoints/MTR-Small_kmer_span_15percent/Small-epoch=02-step=98274-val_loss=0.094.ckpt'"

    done
done

for iter in 1 2 3; do

    for model in MLM-MTR_large; do
        # span mask 25%
        sbatch --time=0-08:00:00 \
            --partition=cu_0001 \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=20 \
            --mem=50000M \
            --job-name=novo_clm_finetune_${model} \
            --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
            --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
            --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name span_mask_25p_round${iter} --checkpoint 'checkpoints/Large-MTR_KmerToks_PubChem-ESMAtlas_25percent-spanMask/Large-epoch=02-step=79274-val_loss=0.149.ckpt'"

        # random mask 25%
        sbatch --time=0-08:00:00 \
            --partition=cu_0001 \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=20 \
            --mem=50000M \
            --job-name=novo_clm_finetune_${model} \
            --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
            --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
            --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name random_mask_25p_round${iter} --checkpoint 'checkpoints/MLM-MTR_Medium_Kmers_lr-0.0003_bs-64_alpha-0.6_beta-0.4_TOP-PERFORMANCE/Medium-epoch=01-step=99550-val_loss=0.060.ckpt'"
    # span mask 25%
    done

    for model in MLM-MTR_medium; do
        # span mask 25%
        sbatch --time=0-08:00:00 \
            --partition=cu_0001 \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=20 \
            --mem=50000M \
            --job-name=novo_clm_finetune_${model} \
            --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
            --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
            --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name span_mask_25p_round${iter} --checkpoint 'checkpoints/Medium-MTR_KmerToks_PubChem-ESMAtlas_25percent-spanMask/Medium-epoch=02-step=98274-val_loss=0.146.ckpt'"

        # random mask 25%
        sbatch --time=0-08:00:00 \
            --partition=cu_0001 \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=20 \
            --mem=50000M \
            --job-name=novo_clm_finetune_${model} \
            --output=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.output.log \
            --error=/dcai/projects02/cu_0001/users/arvf/novo_clm/output/%j.error.log \
            --wrap="srun python finetune_ensemble.py -i perm --model ${model} --save_dir 'finetuning/triplicate/${model}' -bs 16 --save_name random_mask_25p_round${iter} --checkpoint 'checkpoints/Working_clip0.1_MLM-MTR_Large_Kmers_lr-0.0003_bs-16/Large-epoch=01-step=98550-val_loss=0.043.ckpt'"
    done
done
