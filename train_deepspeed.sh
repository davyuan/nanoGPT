#!/bin/bash

# Shell script to run nanoGPT training with DeepSpeed on 2 RTX 4090s
# This will shard the model across both GPUs using ZeRO Stage 3
#
# Usage:
#   ./train_deepspeed.sh                    # Use default config
#   ./train_deepspeed.sh config/train_gpt2.py  # Use specific config file

echo -e "\033[32mStarting DeepSpeed training with model sharding across 2 RTX 4090s...\033[0m"
echo ""

# Check if a config file was provided as argument
CONFIG_FILE=${1:-""}
if [ -n "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_FILE" ]; then
        echo -e "\033[36mUsing config file: $CONFIG_FILE\033[0m"
    else
        echo -e "\033[31mError: Config file '$CONFIG_FILE' not found!\033[0m"
        exit 1
    fi
else
    echo -e "\033[36mUsing default configuration\033[0m"
fi

# Check if DeepSpeed is installed
echo "Checking DeepSpeed installation..."
if python3 -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)" 2>/dev/null; then
    echo -e "\033[32mDeepSpeed is installed and ready.\033[0m"
else
    echo -e "\033[33mDeepSpeed not found. Installing DeepSpeed...\033[0m"
    pip3 install deepspeed
    if [ $? -ne 0 ]; then
        echo -e "\033[31mFailed to install DeepSpeed. Please install manually:\033[0m"
        echo -e "\033[33mpip3 install deepspeed\033[0m"
        exit 1
    fi
    echo -e "\033[32mDeepSpeed installed successfully.\033[0m"
fi

# Set environment variables for maximum stability on consumer GPUs
export NCCL_P2P_DISABLE=1           # Disable P2P for stability on consumer GPUs
export NCCL_IB_DISABLE=1            # Disable InfiniBand
export NCCL_SHM_DISABLE=1           # Disable shared memory transport
export NCCL_NET_GDR_LEVEL=0         # Disable GPU Direct RDMA
export NCCL_NET_GDR_READ=0          # Disable GPU Direct RDMA reads
export NCCL_NVLS_ENABLE=0           # Disable NVLS (not available on consumer GPUs)
export NCCL_ALGO=Ring               # Force ring algorithm (most stable)
export NCCL_PROTO=Simple            # Use simple protocol
export NCCL_DEBUG=WARN              # Reduce debug verbosity
export NCCL_TIMEOUT=3600             # Increase timeout to 1 hour
export TORCH_NCCL_BLOCKING_WAIT=1   # Use blocking wait for better error handling (updated variable name)
export TORCH_DISTRIBUTED_DEBUG=OFF  # Reduce distributed training debug output
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128  # Better memory management with fragmentation reduction
export CUDA_LAUNCH_BLOCKING=0       # Keep async for performance
export NCCL_MIN_NRINGS=1            # Minimize rings for better performance
export NCCL_MAX_NRINGS=1            # Maximize rings for better performance

echo ""
echo -e "\033[32mRunning DeepSpeed training...\033[0m"
echo -e "\033[36mConfiguration:\033[0m"
echo -e "\033[37m- ZeRO Stage: 3 (Model + Optimizer Sharding - Full Distribution)\033[0m"
echo -e "\033[37m- Compilation: Disabled (incompatible with DeepSpeed ZeRO)\033[0m"
echo -e "\033[37m- Checkpoints: ./out_deepspeed\033[0m"
echo -e "\033[37m- Script: train_deepspeed.py\033[0m"
echo ""

# Run with DeepSpeed on 2 GPUs (config loaded from deepspeed_config.json)
if [ -n "$CONFIG_FILE" ]; then
    # Run with config file
    deepspeed --num_gpus=2 train_deepspeed.py "$CONFIG_FILE"
else
    # Run with default parameters (memory optimized)
    deepspeed --num_gpus=2 train_deepspeed.py \
        --deepspeed_config_path=deepspeed_config_lowmem.json
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "\033[32mTraining completed successfully!\033[0m"
    echo -e "\033[36mCheck the 'out_deepspeed' directory for DeepSpeed checkpoints.\033[0m"
    echo -e "\033[36mRegular checkpoints (if any) are in the 'out' directory.\033[0m"
else
    echo ""
    echo -e "\033[31mTraining failed with exit code: $?\033[0m"
    echo -e "\033[33mCheck the output above for error details.\033[0m"
    exit 1
fi

echo ""
echo -e "\033[32mScript completed.\033[0m"
