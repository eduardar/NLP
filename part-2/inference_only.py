import torch
from tqdm import tqdm
from t5_utils import initialize_model, load_model_from_checkpoint
from load_data import load_t5_data, TOKENIZER
from utils import save_queries_and_records
import argparse, os

args = argparse.Namespace(
    finetune=True,
    experiment_name="t5_ft_experiment",
    optimizer_type="AdamW",
    learning_rate=5e-5,
    weight_decay=0.01,
    scheduler_type="cosine",
    num_warmup_epochs=1,
    max_n_epochs=20,
    patience_epochs=5,
    batch_size=16,
    test_batch_size=32,
)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Loading best checkpoint...")
model = load_model_from_checkpoint(args, best=True)
model.eval()

_, _, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

print("Running test inference...")
all_generated = []
with torch.no_grad():
    for encoder_input, encoder_mask, _ in tqdm(test_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask  = encoder_mask.to(DEVICE)
        generated_ids = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True,
        )
        decoded = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        all_generated.extend(decoded)

os.makedirs('results', exist_ok=True)
os.makedirs('records', exist_ok=True)
save_queries_and_records(
    all_generated,
    'results/t5_ft_experiment_test.sql',
    'records/t5_ft_experiment_test.pkl'
)
print("Done! Submit these two files:")
print("  results/t5_ft_experiment_test.sql")
print("  records/t5_ft_experiment_test.pkl")