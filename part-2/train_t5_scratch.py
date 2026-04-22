import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (initialize_model, initialize_optimizer_and_scheduler,
                      save_model, load_model_from_checkpoint, setup_wandb)
from load_data import load_t5_data, TOKENIZER
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser(description='T5 from-scratch training loop')

    # NOTE: do NOT pass --finetune when running this script
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--optimizer_type', type=str, default='AdamW', choices=['AdamW'])

    # ---------------------------------------------------------------------------
    # FIX: Learning rate for from-scratch training
    # 1e-3 is too high and causes instability. Use 3e-4 with warmup instead.
    # ---------------------------------------------------------------------------
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['none', 'cosine', 'linear'])
    parser.add_argument('--num_warmup_epochs', type=int, default=5,
                        help='More warmup needed for scratch training')
    parser.add_argument('--max_n_epochs', type=int, default=100)
    parser.add_argument('--patience_epochs', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='t5_scratch_experiment')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)

    return parser.parse_args()


def train_epoch(args, model, train_loader, optimizer, scheduler):
    """
    FIX: Use model's built-in loss (labels argument) instead of manual
    cross-entropy on hand-shifted targets.  This is identical to the
    fix in train_t5.py.
    """
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    for encoder_input, encoder_mask, labels, _, __ in tqdm(train_loader):
        optimizer.zero_grad()

        encoder_input = encoder_input.to(DEVICE)
        encoder_mask  = encoder_mask.to(DEVICE)
        labels        = labels.to(DEVICE)

        outputs = model(
            input_ids      = encoder_input,
            attention_mask = encoder_mask,
            labels         = labels,
        )
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens    = (labels != -100).sum().item()
            total_loss   += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def eval_epoch(args, model, dev_loader,
               gt_sql_path, model_sql_path,
               gt_record_path, model_record_path):
    """
    FIX: Use model.generate() without overriding decoder_input_ids.
    T5 sets decoder_start_token_id = 0 internally. Beam search (num_beams=4)
    helps significantly for a from-scratch model that has higher uncertainty.
    """
    model.eval()
    total_loss   = 0.0
    total_tokens = 0
    all_generated = []

    with torch.no_grad():
        for encoder_input, encoder_mask, labels, _, __ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask  = encoder_mask.to(DEVICE)
            labels        = labels.to(DEVICE)

            outputs = model(
                input_ids      = encoder_input,
                attention_mask = encoder_mask,
                labels         = labels,
            )
            num_tokens    = (labels != -100).sum().item()
            total_loss   += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

            generated_ids = model.generate(
                input_ids      = encoder_input,
                attention_mask = encoder_mask,
                max_new_tokens = 512,
                num_beams      = 4,
                early_stopping = True,
            )
            decoded = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated.extend(decoded)

    os.makedirs(os.path.dirname(model_sql_path)    or '.', exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path) or '.', exist_ok=True)
    save_queries_and_records(all_generated, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(1 for m in error_msgs if m) / len(error_msgs) if error_msgs else 0
    eval_loss  = total_loss / total_tokens if total_tokens > 0 else 0.0

    return eval_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    all_generated = []

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask  = encoder_mask.to(DEVICE)

            generated_ids = model.generate(
                input_ids      = encoder_input,
                attention_mask = encoder_mask,
                max_new_tokens = 512,
                num_beams      = 4,
                early_stopping = True,
            )
            decoded = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated.extend(decoded)

    os.makedirs(os.path.dirname(model_sql_path)    or '.', exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path) or '.', exist_ok=True)
    save_queries_and_records(all_generated, model_sql_path, model_record_path)
    print(f'Test inference done. SQL saved to {model_sql_path}')


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0
    model_type     = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    gt_sql_path       = 'data/dev.sql'
    gt_record_path    = 'records/ground_truth_dev.pkl'
    model_sql_path    = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f'Epoch {epoch}: train loss = {tr_loss:.4f}')

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        print(f'Epoch {epoch}: dev loss={eval_loss:.4f}  F1={record_f1:.4f}  '
              f'RecEM={record_em:.4f}  SqlEM={sql_em:.4f}  err={error_rate*100:.1f}%')

        if args.use_wandb:
            wandb.log({
                'train/loss'     : tr_loss,
                'dev/loss'       : eval_loss,
                'dev/record_f1'  : record_f1,
                'dev/record_em'  : record_em,
                'dev/sql_em'     : sql_em,
                'dev/error_rate' : error_rate,
            }, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)

        if epochs_since_improvement >= args.patience_epochs:
            print(f'Early stopping after {epoch + 1} epochs.')
            break

    return model


def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    print('Training T5 from scratch (no pretrained weights)...')
    model = train(args, model, train_loader, dev_loader, optimizer, scheduler)

    best_model = load_model_from_checkpoint(args, best=True)
    if best_model is not None:
        model = best_model
    model.eval()

    model_type        = 'ft' if args.finetune else 'scr'
    gt_sql_path       = 'data/dev.sql'
    gt_record_path    = 'records/ground_truth_dev.pkl'
    model_sql_path    = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    dev_loss, dev_f1, dev_em, dev_sql_em, dev_err = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    print(f'Final Dev: loss={dev_loss:.4f}  F1={dev_f1:.4f}  SqlEM={dev_sql_em:.4f}  '
          f'err={dev_err*100:.1f}%')

    # -----------------------------------------------------------------------
    # IMPORTANT: the autograder expects exactly these filenames for EC:
    #   t5_ft_experiment_ec_test.sql
    #   t5_ft_experiment_ec_test.pkl
    # -----------------------------------------------------------------------
    test_sql_path    = 'results/t5_ft_experiment_ec_test.sql'
    test_record_path = 'records/t5_ft_experiment_ec_test.pkl'
    test_inference(args, model, test_loader, test_sql_path, test_record_path)


if __name__ == '__main__':
    main()