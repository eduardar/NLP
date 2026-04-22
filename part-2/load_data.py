import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast

# Initialize the tokenizer once at the global level
TOKENIZER = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# -----------------------------------------------------------------------
# FIX 1: BOS token for T5
# The correct decoder start token for T5 is the PAD token (id=0), NOT
# <extra_id_0> (id=32099).  T5ForConditionalGeneration.generate() uses
# config.decoder_start_token_id which is set to 0 (pad_token_id) by
# default.  Using 32099 as the forced first token causes the decoder to
# start in an unexpected state and the model either copies the encoder
# input or generates garbage.
# -----------------------------------------------------------------------
PAD_IDX   = TOKENIZER.pad_token_id   # 0
BOS_ID    = TOKENIZER.pad_token_id   # 0  ← correct decoder start for T5


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.data_folder = data_folder
        self.encoder_texts, self.decoder_texts = self.process_data(data_folder, split)

    def process_data(self, data_folder, split):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            encoder_texts = [line.strip() for line in f.readlines()]

        sql_path = os.path.join(data_folder, f'{split}.sql')
        if os.path.exists(sql_path):
            with open(sql_path, 'r') as f:
                decoder_texts = [line.strip() for line in f.readlines()]
        else:
            decoder_texts = [''] * len(encoder_texts)

        return encoder_texts, decoder_texts

    def __len__(self):
        return len(self.encoder_texts)

    def __getitem__(self, idx):
        return {
            'encoder_text': self.encoder_texts[idx],
            'decoder_text': self.decoder_texts[idx]
        }


def normal_collate_fn(batch):
    """
    Collate function for train / dev splits.

    FIX 2: Correct label masking for the cross-entropy loss.
    T5's built-in `.forward()` accepts a `labels` argument directly.
    When you pass `labels`, the model:
      (a) automatically right-shifts them to build decoder_input_ids
          (prepending decoder_start_token_id = pad id = 0), and
      (b) replaces padding positions with -100 so CrossEntropyLoss
          ignores them — no manual masking needed.

    Doing the shift manually (as in the original code) was off-by-one
    and left the EOS token out of the target, causing truncated outputs.

    We therefore return `labels` (the raw tokenized SQL ids with -100
    padding) instead of hand-rolled decoder_input_ids / decoder_targets.
    """
    encoder_texts = [item['encoder_text'] for item in batch]
    decoder_texts = [item['decoder_text'] for item in batch]

    # Encode NL inputs
    encoder_encoded = TOKENIZER(
        encoder_texts,
        padding=True,
        truncation=True,
        max_length=128,          # NL queries are short (mean=17 tokens)
        return_tensors='pt'
    )

    # FIX 3: use text_target / as_target_tokenizer so the tokenizer
    # knows these are decoder outputs (adds EOS, applies correct vocab).
    with TOKENIZER.as_target_tokenizer():
        decoder_encoded = TOKENIZER(
            decoder_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    # Replace padding token ids in the labels by -100 so loss ignores them
    labels = decoder_encoded['input_ids'].clone()
    labels[labels == TOKENIZER.pad_token_id] = -100

    # initial_decoder_inputs kept for API compatibility with training loop
    batch_size = len(batch)
    initial_decoder_inputs = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)

    return (
        encoder_encoded['input_ids'],
        encoder_encoded['attention_mask'],
        labels,               # used directly as `labels=` in model forward
        labels,               # decoder_targets placeholder (same tensor)
        initial_decoder_inputs
    )


def test_collate_fn(batch):
    """Collate function for test split (no SQL targets)."""
    encoder_texts = [item['encoder_text'] for item in batch]

    encoder_encoded = TOKENIZER(
        encoder_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    batch_size = len(batch)
    initial_decoder_inputs = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)

    return encoder_encoded['input_ids'], encoder_encoded['attention_mask'], initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = (split == 'train')
    collate_fn = normal_collate_fn if split != 'test' else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, 'train')
    dev_loader   = get_dataloader(test_batch_size, 'dev')
    test_loader  = get_dataloader(test_batch_size, 'test')
    return train_loader, dev_loader, test_loader
