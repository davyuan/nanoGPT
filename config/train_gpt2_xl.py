# config for training GPT-2 XL with DeepSpeed low memory optimization
# Optimized for RTX 4090 GPUs with minimal memory usage
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ deepspeed --num_gpus=2 train_deepspeed.py config/train_gpt2_xl.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-xl-lowmem'

# Memory-optimized batch settings for low VRAM consumption
# Micro batch size is fixed to 1 in deepspeed_config_lowmem.json
# Gradient accumulation compensates for small micro batch
batch_size = 1  # This will be overridden by DeepSpeed config
block_size = 1024
gradient_accumulation_steps = 32  # Increased for better memory efficiency  

# eval stuff - less frequent for low memory training
eval_interval = 50  # Less frequent evaluation to save memory
eval_iters = 50
log_interval = 10

# DeepSpeed low memory config for minimal GPU memory usage
deepspeed_config_path = 'deepspeed_config_lowmem.json'
