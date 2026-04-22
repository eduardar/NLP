import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    '''
    Initialize wandb logging.
    '''
    wandb.init(
        project="nlp_hw4_t5",
        name=args.experiment_name,
        config=vars(args)
    )

def initialize_model(args):
    '''
    Initializes T5-small either from pretrained weights or from scratch config.
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)
    
    # Attach the tokenizer
    model.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def save_model(checkpoint_dir, model, best):
    '''
    Saves the model state dict. 
    'best=True' saves as best_model.pt, 'best=False' saves as last_model.pt.
    '''
    mkdir(checkpoint_dir)
    filename = 'best_model.pt' if best else 'last_model.pt'
    save_path = os.path.join(checkpoint_dir, filename)
    
    # We save the state_dict to save space
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model_from_checkpoint(args, best):
    '''
    Re-initializes the model architecture and loads the saved weights.
    '''
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    filename = 'best_model.pt' if best else 'last_model.pt'
    load_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        return None

    # 1. Initialize a fresh model skeleton
    model = initialize_model(args)
    
    # 2. Load the weights into that skeleton
    model.load_state_dict(torch.load(load_path, map_location=DEVICE))
    model.to(DEVICE)
    
    print(f"Successfully loaded model from {load_path}")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        # Default to standard AdamW if type is unknown
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result