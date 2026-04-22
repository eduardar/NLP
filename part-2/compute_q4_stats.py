#!/usr/bin/env python3
"""
Compute Q4 dataset statistics for the T5 text-to-SQL homework assignment.
This script computes statistics BEFORE and AFTER preprocessing (tokenization).
"""

import os
from collections import Counter
from transformers import T5TokenizerFast
import numpy as np

def load_lines(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def compute_raw_stats(nl_texts, sql_texts):
    """Compute statistics on raw (untokenized) data"""
    stats = {
        'num_examples': len(nl_texts),
        'nl_mean_length': np.mean([len(text) for text in nl_texts]),
        'nl_min_length': min([len(text) for text in nl_texts]),
        'nl_max_length': max([len(text) for text in nl_texts]),
        'sql_mean_length': np.mean([len(text) for text in sql_texts]),
        'sql_min_length': min([len(text) for text in sql_texts]),
        'sql_max_length': max([len(text) for text in sql_texts]),
    }
    
    # Compute vocabulary for raw text (unique words)
    nl_vocab = set()
    sql_vocab = set()
    
    for text in nl_texts:
        nl_vocab.update(text.lower().split())
    for text in sql_texts:
        sql_vocab.update(text.lower().split())
    
    stats['nl_vocab_size'] = len(nl_vocab)
    stats['sql_vocab_size'] = len(sql_vocab)
    
    return stats

def compute_tokenized_stats(nl_texts, sql_texts, tokenizer):
    """Compute statistics after T5 tokenization"""
    stats = {
        'num_examples': len(nl_texts),
    }
    
    # Tokenize all texts
    nl_token_lengths = []
    sql_token_lengths = []
    all_nl_tokens = set()
    all_sql_tokens = set()
    
    for text in nl_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        nl_token_lengths.append(len(tokens))
        all_nl_tokens.update(tokens)
    
    for text in sql_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        sql_token_lengths.append(len(tokens))
        all_sql_tokens.update(tokens)
    
    stats['nl_mean_token_length'] = np.mean(nl_token_lengths)
    stats['nl_min_token_length'] = min(nl_token_lengths)
    stats['nl_max_token_length'] = max(nl_token_lengths)
    stats['nl_vocab_size'] = len(all_nl_tokens)
    
    stats['sql_mean_token_length'] = np.mean(sql_token_lengths)
    stats['sql_min_token_length'] = min(sql_token_lengths)
    stats['sql_max_token_length'] = max(sql_token_lengths)
    stats['sql_vocab_size'] = len(all_sql_tokens)
    
    # Get tokenizer vocab size
    stats['tokenizer_vocab_size'] = tokenizer.vocab_size
    
    return stats

def main():
    # Load data
    data_folder = 'data'
    
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    print("=" * 80)
    print("TABLE 1: DATA STATISTICS BEFORE PREPROCESSING (RAW TEXT)")
    print("=" * 80)
    
    train_raw = compute_raw_stats(train_nl, train_sql)
    dev_raw = compute_raw_stats(dev_nl, dev_sql)
    
    print(f"{'Statistics Name':<40} {'Train':>15} {'Dev':>15}")
    print("-" * 70)
    print(f"{'Number of examples':<40} {train_raw['num_examples']:>15} {dev_raw['num_examples']:>15}")
    print(f"{'Mean sentence length (chars)':<40} {train_raw['nl_mean_length']:>15.2f} {dev_raw['nl_mean_length']:>15.2f}")
    print(f"{'Mean SQL query length (chars)':<40} {train_raw['sql_mean_length']:>15.2f} {dev_raw['sql_mean_length']:>15.2f}")
    print(f"{'Vocabulary size (natural language)':<40} {train_raw['nl_vocab_size']:>15} {dev_raw['nl_vocab_size']:>15}")
    print(f"{'Vocabulary size (SQL)':<40} {train_raw['sql_vocab_size']:>15} {dev_raw['sql_vocab_size']:>15}")
    print()
    
    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("=" * 80)
    print("TABLE 2: DATA STATISTICS AFTER PREPROCESSING (TOKENIZED)")
    print("=" * 80)
    
    train_tok = compute_tokenized_stats(train_nl, train_sql, tokenizer)
    dev_tok = compute_tokenized_stats(dev_nl, dev_sql, tokenizer)
    
    print(f"Model: google-t5/t5-small")
    print()
    print(f"{'Statistics Name':<40} {'Train':>15} {'Dev':>15}")
    print("-" * 70)
    print(f"{'Mean NL tokens per example':<40} {train_tok['nl_mean_token_length']:>15.2f} {dev_tok['nl_mean_token_length']:>15.2f}")
    print(f"{'Mean SQL tokens per example':<40} {train_tok['sql_mean_token_length']:>15.2f} {dev_tok['sql_mean_token_length']:>15.2f}")
    print(f"{'NL vocabulary size (unique tokens)':<40} {train_tok['nl_vocab_size']:>15} {dev_tok['nl_vocab_size']:>15}")
    print(f"{'SQL vocabulary size (unique tokens)':<40} {train_tok['sql_vocab_size']:>15} {dev_tok['sql_vocab_size']:>15}")
    print(f"{'Tokenizer vocabulary size':<40} {train_tok['tokenizer_vocab_size']:>15} {dev_tok['tokenizer_vocab_size']:>15}")
    print()
    
    # Additional statistics
    print("=" * 80)
    print("ADDITIONAL STATISTICS")
    print("=" * 80)
    
    print(f"\nNatural Language (Train):")
    print(f"  - Min token length: {train_tok['nl_min_token_length']}")
    print(f"  - Max token length: {train_tok['nl_max_token_length']}")
    print(f"  - Mean token length: {train_tok['nl_mean_token_length']:.2f}")
    
    print(f"\nSQL Query (Train):")
    print(f"  - Min token length: {train_tok['sql_min_token_length']}")
    print(f"  - Max token length: {train_tok['sql_max_token_length']}")
    print(f"  - Mean token length: {train_tok['sql_mean_token_length']:.2f}")
    
    print(f"\nNatural Language (Dev):")
    print(f"  - Min token length: {dev_tok['nl_min_token_length']}")
    print(f"  - Max token length: {dev_tok['nl_max_token_length']}")
    print(f"  - Mean token length: {dev_tok['nl_mean_token_length']:.2f}")
    
    print(f"\nSQL Query (Dev):")
    print(f"  - Min token length: {dev_tok['sql_min_token_length']}")
    print(f"  - Max token length: {dev_tok['sql_max_token_length']}")
    print(f"  - Mean token length: {dev_tok['sql_mean_token_length']:.2f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
