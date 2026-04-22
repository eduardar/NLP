import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    print(f"Training completed. Saving Model to {save_dir}...")
    model.save_pretrained(save_dir)
    # Also save tokenizer so AutoModel can reload it later
    tokenizer.save_pretrained(save_dir)
    return model

# Core evaluation function - UPDATED to take model object directly
def do_eval(model, eval_dataloader, device, out_file):
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")
    
    with open(out_file, "w") as f:
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

            for pred, label in zip(predictions, batch["labels"]):
                f.write(f"{pred.item()}\n")
                f.write(f"{label.item()}\n")
                
    score = metric.compute()
    return score

def create_augmented_dataloader(args, dataset):
    raw_train_data = dataset["train"]
    sample_size = min(5000, len(raw_train_data))
    sampled_data = raw_train_data.shuffle(seed=42).select(range(sample_size))
    
    # custom_transform comes from your utils.py
    augmented_data = sampled_data.map(custom_transform, load_from_cache_file=False)
    combined_data = datasets.concatenate_datasets([raw_train_data, augmented_data])
    
    tokenized_combined = combined_data.map(tokenize_function, batched=True, load_from_cache_file=False)
    tokenized_combined = tokenized_combined.remove_columns(["text"])
    tokenized_combined = tokenized_combined.rename_column("label", "labels")
    tokenized_combined.set_format("torch")
    
    return DataLoader(tokenized_combined, shuffle=True, batch_size=args.batch_size)

def create_transformed_dataloader(args, dataset, debug_transformation):
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print(f"Original: {small_dataset[k]['text'][:100]}...")
            print(f"Transformed: {small_transformed_dataset[k]['text'][:100]}...")
            print('=' * 30)
        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    return DataLoader(transformed_tokenized_dataset, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_augmented", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_transformed", action="store_true")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true")
    parser.add_argument("--debug_transformation", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    dataset = load_dataset("imdb", ignore_verifications=True)
    
    # Initial Model Variable
    model = None

    # 1. Train original
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        train_dataloader = DataLoader(dataset["train"].shuffle(seed=42).select(range(4000)) if args.debug_train else dataset["train"].map(tokenize_function, batched=True).remove_columns(["text"]).rename_column("label", "labels").with_format("torch"), shuffle=True, batch_size=args.batch_size)
        model = do_train(args, model, train_dataloader, save_dir="./out")
        args.model_dir = "./out"

    # 2. Train augmented
    if args.train_augmented:
        aug_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        model = do_train(args, model, aug_dataloader, save_dir="./out_augmented")
        args.model_dir = "./out_augmented"

    # 3. Handle Evaluations
    if args.eval or args.eval_transformed:
        # If we didn't just train the model, we need to load it from the disk
        if model is None:
            if os.path.exists(args.model_dir):
                print(f"Loading model from {args.model_dir}...")
                model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
            else:
                print(f"Error: No model found in {args.model_dir}. Did you forget --train?")
                exit()

        # Original Test Set
        if args.eval:
            out_file = os.path.basename(os.path.normpath(args.model_dir)) + "_original.txt"
            eval_dataloader = DataLoader(dataset["test"].map(tokenize_function, batched=True).remove_columns(["text"]).rename_column("label", "labels").with_format("torch"), batch_size=args.batch_size)
            score = do_eval(model, eval_dataloader, device, out_file)
            print(f"Original Test Score: {score}")

        # Transformed Test Set
        if args.eval_transformed:
            out_file = os.path.basename(os.path.normpath(args.model_dir)) + "_transformed.txt"
            eval_trans_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
            score = do_eval(model, eval_trans_dataloader, device, out_file)
            print(f"Transformed Test Score: {score}")