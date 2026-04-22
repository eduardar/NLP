#!/bin/bash
#SBATCH --job-name=hw4_bert
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/bert_%j.log

# Initialize conda and activate the environment
eval "$(conda shell.bash hook)"
conda activate nlp_hw4

# UPDATED: Use your actual home directory path
cd /home/er3410/hw4/release/part-1

# Optional: Ensure directories exist before running
mkdir -p out out_augmented logs

# Install dependencies (optional if you already ran it, but safe to keep)
pip install -r requirements.txt

# Debug test first
python3 main.py --train --eval --debug_train

python3 main.py \
    --train \
    --train_augmented \
    --eval \
    --eval_transformed \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5