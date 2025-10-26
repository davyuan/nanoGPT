# config for training GPT-2 (124M) with DeepSpeed on 2x RTX 4090s
# launch as: ./train_deepspeed.sh config/train_gpt2_deepspeed.py

# Enable DeepSpeed
use_deepspeed = True

# wandb logging
wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M-deepspeed'

# Model configuration - GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Training configuration optimized for 2x RTX 4090s (24GB each)
# With ZeRO Stage 3, we can handle larger effective batch sizes
batch_size = 16  # per GPU micro-batch size
block_size = 1024
gradient_accumulation_steps = 8  # 16 * 1024 * 8 * 2 GPUs = ~262K tokens per iteration

# Optimizer settings
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# Evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 1
eval_only = False
always_save_checkpoint = True

# System
device = 'cuda'
dtype = 'bfloat16'  # Use bfloat16 for RTX 4090s
compile = True

# Data
dataset = 'openwebtext'
