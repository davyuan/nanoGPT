"""
DeepSpeed training script for large language models.
This script uses DeepSpeed's ZeRO optimization to train large models across multiple GPUs.

To run with DeepSpeed on multiple GPUs:
$ deepspeed --num_gpus=2 train_deepspeed.py

Also supports config files:
$ deepspeed --num_gpus=2 train_deepspeed.py config/train_gpt2.py

DeepSpeed enables training of very large models by sharding model parameters, gradients, and optimizer states across GPUs.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import warnings
import wandb
import traceback
import json
    

# Suppress specific warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*The 'repr' attribute.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank
from model import GPTConfig, GPT
from util import get_batch, estimate_loss

# DeepSpeed imports
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available. Install with: pip install deepspeed")

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out_deepspeed'
eval_interval = 1000  # More frequent evaluation for debugging
log_interval = 1
eval_iters = 50      # Reduced from 200 to make evaluation faster
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2_xl' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
# Alternative configurations for different scenarios:

# Recommended for GPT-2 XL on 2x RTX 4090 with ZeRO Stage 3:
gradient_accumulation_steps = 8  # Effective batch size will be calculated by DeepSpeed
batch_size = 2 # Micro-batch size per GPU - increased for better GPU utilization
block_size = 1024
# model
n_layer = 48  # GPT-2 XL configuration - using DeepSpeed to handle large model
n_head = 25   # GPT-2 XL configuration
n_embd = 1600 # GPT-2 XL configuration
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# DeepSpeed settings
deepspeed_config_path = 'deepspeed_config_stable.json' # path to DeepSpeed configuration file
deepspeed_out_dir = 'out_deepspeed' # separate output directory for DeepSpeed checkpoints
deepspeed_skip_checkpoints = False # set to True to skip checkpoint saving entirely (for debugging)
local_rank = -1 # DeepSpeed will set this automatically
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

def load_deepspeed_config():
    """
    Load and configure DeepSpeed settings from JSON file.
    Updates "auto" parameters with actual training values.
    
    Returns:
        dict: DeepSpeed configuration dictionary
    """
    if os.path.exists(deepspeed_config_path):
        with open(deepspeed_config_path, 'r') as f:
            ds_config = json.load(f)
        print_master(f"Loaded DeepSpeed config from {deepspeed_config_path}")
    else:
        # Fallback to default config if file doesn't exist  
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto", 
            "gradient_accumulation_steps": "auto",
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": "auto", "betas": "auto", "eps": 1e-8, "weight_decay": "auto"}
            },
            "zero_optimization": {"stage": 3},
            "fp16": {"enabled": "auto"},
            "bf16": {"enabled": "auto"}
        }
        print_master(f"Warning: {deepspeed_config_path} not found, using default config")
    
    # Update config with training parameters (only if using "auto" values)
    if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
        ds_config["train_micro_batch_size_per_gpu"] = batch_size
    if ds_config.get("gradient_accumulation_steps") == "auto":
        ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    if ds_config.get("gradient_clipping") == "auto":
        ds_config["gradient_clipping"] = grad_clip
    
    # Calculate train_batch_size if set to "auto"
    if ds_config.get("train_batch_size") == "auto":
        # train_batch_size = micro_batch_size * gradient_accumulation_steps * num_gpus
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        micro_batch = int(ds_config["train_micro_batch_size_per_gpu"])
        grad_accum = int(ds_config["gradient_accumulation_steps"])
        ds_config["train_batch_size"] = micro_batch * grad_accum * world_size
    
    # Ensure all batch size parameters are integers (not strings)
    if isinstance(ds_config.get("train_micro_batch_size_per_gpu"), str):
        ds_config["train_micro_batch_size_per_gpu"] = int(ds_config["train_micro_batch_size_per_gpu"])
    if isinstance(ds_config.get("gradient_accumulation_steps"), str):
        ds_config["gradient_accumulation_steps"] = int(ds_config["gradient_accumulation_steps"])
    if isinstance(ds_config.get("train_batch_size"), str):
        ds_config["train_batch_size"] = int(ds_config["train_batch_size"])
        
    # Update optimizer params if using "auto"
    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        optimizer_params = ds_config["optimizer"]["params"]
        if optimizer_params.get("lr") == "auto":
            optimizer_params["lr"] = learning_rate
        if optimizer_params.get("betas") == "auto":
            optimizer_params["betas"] = [beta1, beta2]
        if optimizer_params.get("weight_decay") == "auto":
            optimizer_params["weight_decay"] = weight_decay
    
    # Update scheduler params if using "auto" 
    if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
        scheduler_params = ds_config["scheduler"]["params"]
        if scheduler_params.get("warmup_min_lr") == "auto":
            scheduler_params["warmup_min_lr"] = min_lr
        if scheduler_params.get("warmup_max_lr") == "auto":
            scheduler_params["warmup_max_lr"] = learning_rate
        if scheduler_params.get("warmup_num_steps") == "auto":
            scheduler_params["warmup_num_steps"] = warmup_iters
        if scheduler_params.get("total_num_steps") == "auto":
            scheduler_params["total_num_steps"] = max_iters
    
    # Enable fp16 or bf16 based on dtype
    if "fp16" in ds_config and ds_config["fp16"].get("enabled") == "auto":
        ds_config["fp16"]["enabled"] = (dtype == 'float16')
    if "bf16" in ds_config and ds_config["bf16"].get("enabled") == "auto":
        ds_config["bf16"]["enabled"] = (dtype == 'bfloat16')
    
    return ds_config

# Check if DeepSpeed is available
if not DEEPSPEED_AVAILABLE:
    print("Error: DeepSpeed is required but not installed. Install with: pip install deepspeed")
    exit(1)

# Initialize DeepSpeed distributed training
deepspeed.init_distributed(dist_backend='nccl')

# Import all global variables that may have been updated by configurator.py

# Set up data directory (needed for model initialization)
data_dir = os.path.join('data', dataset)

# Set up device and process info first
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
torch.cuda.set_device(local_rank)
device = f'cuda:{local_rank}'
device_type = 'cuda'  # DeepSpeed uses CUDA

# Determine master process after DeepSpeed initialization
master_process = local_rank == 0

# Custom print function that only prints from master process
def print_master(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)

# Create DeepSpeed output directory
if master_process:
    os.makedirs(deepspeed_out_dir, exist_ok=True)
print_master(f"DeepSpeed checkpoints will be saved to: {deepspeed_out_dir}")

# Set up data loading
tokens_per_iter = gradient_accumulation_steps * world_size * batch_size * block_size
print_master(f"tokens per iteration will be: {tokens_per_iter:,}")
print_master(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {world_size} processes * {batch_size} batch size * {block_size} max seq len")

# Model initialization
iter_num = 0
best_val_loss = 1e9

# Load model and setup
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print_master("Initializing a new model from scratch")
    try:
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print_master(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    except (FileNotFoundError, KeyError):
        print_master("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        meta_vocab_size = 50304
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print_master(f"Resuming DeepSpeed training from {deepspeed_out_dir}")
    # For DeepSpeed, we need model structure info to initialize the model
    # The actual training state (iter_num, best_val_loss) will be loaded from DeepSpeed checkpoint
    try:
        # Try to load regular checkpoint for model structure info only
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            checkpoint_model_args = checkpoint['model_args']
            # Only extract model architecture parameters
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            print_master(f"Found model structure info from {ckpt_path}")
        else:
            print_master("No regular checkpoint found, using default model structure")
            meta_vocab_size = 50304  # fallback
            model_args['vocab_size'] = meta_vocab_size
    except Exception as e:
        print_master(f"Could not load checkpoint info: {e}, using defaults")
        meta_vocab_size = 50304
        model_args['vocab_size'] = meta_vocab_size
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    print_master(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

# Load and configure DeepSpeed settings
ds_config = load_deepspeed_config()
print_master(f"DeepSpeed config loaded with ZeRO stage {ds_config['zero_optimization']['stage']}")

# Initialize DeepSpeed engine
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    args=None,
    model=model,
    config=ds_config,
    dist_init_required=True
)

# Load DeepSpeed checkpoint if resuming
if init_from == 'resume':
    try:
        _, client_state = model_engine.load_checkpoint(deepspeed_out_dir)
        if client_state is not None:
            iter_num = client_state.get('iter_num', iter_num)
            best_val_loss = client_state.get('best_val_loss', best_val_loss)
            print_master(f"Loaded DeepSpeed checkpoint from {deepspeed_out_dir}")
        else:
            print_master(f"No DeepSpeed checkpoint found in {deepspeed_out_dir}, starting fresh")
    except Exception as e:
        print_master(f"Could not load DeepSpeed checkpoint: {e}, starting fresh")

# Logging setup
if wandb_log and master_process:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Debug: Print DeepSpeed configuration and GPU utilization info
print_master(f"DeepSpeed Configuration:")
print_master(f"- ZeRO Stage: {ds_config['zero_optimization']['stage']}")
print_master(f"- World Size: {world_size}")
print_master(f"- Local Rank: {local_rank}")
print_master(f"- Device: {device}")
print_master(f"- Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Check if model is properly sharded
if hasattr(model_engine, 'module'):
    print_master(f"- Model is wrapped in DDP-like container")
else:
    print_master(f"- Model is not wrapped")

# Training loop
print_master(f"Starting DeepSpeed training with ZeRO stage {ds_config['zero_optimization']['stage']}")
X, Y = get_batch('train', data_dir, block_size, batch_size, device_type, device)
print_master(f"split:train, data_dir:{data_dir}, block_size:{block_size}, batch_size:{batch_size}, device_type:{device_type}, device:{device}")

t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
checkpoint_failures = 0  # Track consecutive checkpoint failures

while True:
    # Evaluation and checkpointing
    if iter_num > 0 and iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(eval_iters, model_engine, None, data_dir, block_size, batch_size, device_type, device, use_deepspeed=True)
        print_master(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            # Get learning rate safely - fallback to optimizer LR if scheduler not ready
            try:
                current_lr = model_engine.get_lr()[0]
            except (IndexError, AttributeError):
                current_lr = model_engine.optimizer.param_groups[0]['lr']
            
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": current_lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                # Try saving with client state for better recovery
                client_state = {
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print_master("Calling save_checkpoint with client_state...")                            
                model_engine.save_checkpoint(deepspeed_out_dir)
                print_master("checkpoint save succeeded")
        
        if iter_num == 0 and eval_only:
            break
        
    # Forward pass - DeepSpeed handles mixed precision automatically
    logits, loss = model_engine(X, Y)
    # Backward pass - DeepSpeed handles gradient accumulation internally
    model_engine.backward(loss)

    # Fetch next batch while model is doing backward pass
    X, Y = get_batch('train', data_dir, block_size, batch_size, device_type, device)
    # Optimizer step - DeepSpeed handles zero_grad internally when gradient_accumulation_steps > 1
    model_engine.step()
    
    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()
        if local_iter_num >= 5:
            # For DeepSpeed, we need to get the raw model for MFU calculation
            raw_model = model_engine.module
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Get current learning rate for logging - handle scheduler not ready
        try:
            current_lr = model_engine.get_lr()[0]
        except (IndexError, AttributeError, RuntimeError):
            current_lr = model_engine.optimizer.param_groups[0]['lr']
        
        print_master(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr {current_lr:.2e}")
    
    iter_num += 1
    local_iter_num += 1
    
    if iter_num > max_iters:
        break
    
