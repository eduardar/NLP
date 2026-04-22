#!/bin/bash
#SBATCH --job-name=t5_scratch
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00  # Scratch takes longer to converge than fine-tuning
#SBATCH --output=logs/t5_scratch_%j.log

eval "$(conda shell.bash hook)"
conda activate nlp_hw4

cd /home/er3410/hw4/release/part-2

# Note: We do NOT pass the --finetune flag here
python3 train_t5_scratch.py \
    --experiment_name "t5_scratch_extra_credit" \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --max_n_epochs 100 \
    --use_wandb