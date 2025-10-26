# DeepSpeed Training for nanoGPT

This setup enables model sharding across multiple GPUs using DeepSpeed ZeRO Stage 3, allowing you to train larger models that wouldn't fit on a single GPU.

## Installation

First, install DeepSpeed:

```bash
pip install deepspeed
```

## Running DeepSpeed Training

### Option 1: Use the provided scripts

**Windows Batch Script:**
```cmd
train_deepspeed.bat
```

**PowerShell Script:**
```powershell
.\train_deepspeed.ps1
```

### Option 2: Manual command

```bash
deepspeed --num_gpus=2 train.py --use_deepspeed=True --batch_size=8 --gradient_accumulation_steps=5 --compile=False
```

## Configuration

DeepSpeed configuration is stored in `deepspeed_config.json`. The configuration uses ZeRO Stage 3 with the following features:

- **Model Sharding**: Distributes model parameters across GPUs
- **Optimizer Offloading**: Offloads optimizer states to CPU to save GPU memory
- **Parameter Offloading**: Offloads inactive parameters to CPU
- **Gradient Compression**: Reduces communication overhead
- **Mixed Precision**: Uses FP16/BF16 for faster training

## Key Parameters

- `--use_deepspeed=True`: Enables DeepSpeed training
- `--batch_size=8`: Micro-batch size per GPU (adjust based on GPU memory)
- `--gradient_accumulation_steps=5`: Number of steps to accumulate gradients
- `--compile=False`: Disables PyTorch compilation (for compatibility)

## Memory Usage

With ZeRO Stage 3 and offloading enabled, you can train much larger models:

- **Without DeepSpeed**: Limited by single GPU memory (~24GB for RTX 4090)
- **With DeepSpeed ZeRO-3**: Can handle models much larger than single GPU memory

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes (4-8) and increase if memory allows
2. **Gradient Accumulation**: Use higher values (5-10) to simulate larger batch sizes
3. **Mixed Precision**: BF16 is recommended for RTX 4090s if supported
4. **CPU Offloading**: Reduces GPU memory usage but may slow training slightly

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce `batch_size` or increase gradient accumulation
2. **Slow Training**: Disable CPU offloading in `deepspeed_config.json`
3. **Communication Errors**: Set environment variables:
   ```bash
   export NCCL_P2P_DISABLE=1
   export NCCL_IB_DISABLE=1
   ```

### Monitoring

- Use `nvidia-smi` to monitor GPU memory usage
- Check training logs for MFU (Model FLOPs Utilization)
- Monitor CPU memory usage if using offloading

## Configuration Files

- `deepspeed_config.json`: Main DeepSpeed configuration file - edit this to customize training
- `train.py`: Training script with DeepSpeed support (loads config from JSON)
- Model checkpoints are saved in DeepSpeed format in the `out/` directory

### Customizing Configuration

Edit `deepspeed_config.json` to customize DeepSpeed settings:

```json
{
  "zero_optimization": {
    "stage": 3,              // 1, 2, or 3 - higher stages use more memory optimization
    "offload_optimizer": {
      "device": "cpu"        // "cpu" or "none" - offload optimizer to CPU to save GPU memory
    }
  },
  "train_micro_batch_size_per_gpu": "auto",  // Will use --batch_size parameter
  "fp16": {
    "enabled": "auto"        // Will auto-detect based on --dtype parameter
  }
}
```

The configuration uses `"auto"` values that are automatically filled from command-line arguments.

## Comparison: DDP vs DeepSpeed

| Feature | DDP | DeepSpeed ZeRO-3 |
|---------|-----|------------------|
| Model Size | Limited by single GPU | Can exceed single GPU memory |
| Memory Usage | Full model on each GPU | Model sharded across GPUs |
| Communication | Gradients only | Parameters + gradients |
| Setup Complexity | Simple | More complex |
| Performance | Faster for small models | Better for large models |

For your 2x RTX 4090 setup, DeepSpeed allows you to train larger models than would fit on a single 24GB GPU.
