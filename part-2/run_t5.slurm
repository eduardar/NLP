#!/bin/bash
#SBATCH --job-name=hw4_t5_ft
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/t5_ft_%j.log

eval "$(conda shell.bash hook)"
conda activate nlp_hw4

cd /home/er3410/hw4/release/part-2
mkdir -p logs results records

python3 train_t5.py \
    --finetune \
    --experiment_name "t5_ft_experiment" \
    --batch_size 16 \
    --test_batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --use_wandb
