#!/usr/bin/env python3
"""
Memory management script for nanoGPT DeepSpeed training.
Run this before training to clear GPU memory and optimize allocation.
"""

import torch
import gc
import sys
import os

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    print("Clearing GPU memory...")

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Get memory info
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(".2f")

    print("GPU memory cleared.")

def optimize_memory_settings():
    """Set optimal memory management settings."""
    print("Setting memory optimization settings...")

    # Set garbage collection threshold
    gc.set_threshold(700, 10, 10)  # More aggressive GC

    # Disable gradient computation globally (will be re-enabled in training)
    torch.set_grad_enabled(False)

    print("Memory settings optimized.")

if __name__ == "__main__":
    print("nanoGPT Memory Manager")
    print("=" * 30)

    clear_gpu_memory()
    optimize_memory_settings()

    print("Ready for training!")
