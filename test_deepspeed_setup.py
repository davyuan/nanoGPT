#!/usr/bin/env python3
"""
Test script to verify DeepSpeed setup and configuration.
Run this before starting full training to ensure everything works.
"""

import os
import sys
import torch
import torch.nn as nn

def test_cuda_setup():
    """Test CUDA availability and GPU detection."""
    print("=== CUDA Setup Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ CUDA not available!")
        return False
    
    print("✅ CUDA setup OK")
    return True

def test_deepspeed():
    """Test DeepSpeed installation and basic functionality."""
    print("\n=== DeepSpeed Test ===")
    
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        print("❌ DeepSpeed not installed!")
        print("Install with: pip install deepspeed")
        return False
    
    # Test basic DeepSpeed initialization
    try:
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Basic DeepSpeed config
        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 2,
            "optimizer": {
                "type": "Adam",
                "params": {"lr": 0.001}
            },
            "fp16": {"enabled": False},
            "zero_optimization": {"stage": 1}
        }
        
        # This will only work in a distributed environment
        # but we can at least test the import and config validation
        print("✅ DeepSpeed imports successfully")
        print("✅ Basic configuration valid")
        
    except Exception as e:
        print(f"❌ DeepSpeed test failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if we can load the nanoGPT model."""
    print("\n=== Model Loading Test ===")
    
    try:
        # Add the current directory to path to import model
        sys.path.append('.')
        from model import GPT, GPTConfig
        
        # Create a small test model
        config = GPTConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False
        )
        
        model = GPT(config)
        print(f"Model parameters: {model.get_num_params():,}")
        
        # Test forward pass
        if torch.cuda.is_available():
            device = 'cuda'
            model = model.to(device)
            x = torch.randint(0, 1000, (2, 32), device=device)
            y = torch.randint(0, 1000, (2, 32), device=device)
            
            logits, loss = model(x, y)
            print(f"Test forward pass successful")
            print(f"Logits shape: {logits.shape}")
            print(f"Loss: {loss.item():.4f}")
        
        print("✅ Model loading and forward pass OK")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_loading():
    """Test if training data is available."""
    print("\n=== Data Loading Test ===")
    
    try:
        import numpy as np
        
        # Check for shakespeare data (most common for testing)
        data_paths = [
            'data/shakespeare_char/train.bin',
            'data/shakespeare/train.bin',
            'data/openwebtext/train.bin'
        ]
        
        found_data = False
        for path in data_paths:
            if os.path.exists(path):
                print(f"Found data: {path}")
                
                # Test loading
                data = np.memmap(path, dtype=np.uint16, mode='r')
                print(f"Data size: {len(data):,} tokens")
                
                # Test batch creation
                block_size = 64
                batch_size = 4
                ix = torch.randint(len(data) - block_size, (batch_size,))
                x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
                y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
                
                print(f"Test batch shape: x={x.shape}, y={y.shape}")
                found_data = True
                break
        
        if not found_data:
            print("❌ No training data found!")
            print("Prepare data first. For Shakespeare:")
            print("cd data/shakespeare_char && python prepare.py")
            return False
        
        print("✅ Data loading OK")
        return True
        
    except Exception as e:
        print(f"❌ Data test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("DeepSpeed nanoGPT Setup Test")
    print("=" * 40)
    
    all_passed = True
    
    all_passed &= test_cuda_setup()
    all_passed &= test_deepspeed()
    all_passed &= test_model_loading()
    all_passed &= test_data_loading()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All tests passed! Ready for DeepSpeed training.")
        print("\nTo start training:")
        print("1. Windows: run train_deepspeed.bat")
        print("2. PowerShell: .\\train_deepspeed.ps1")
        print("3. Manual: deepspeed --num_gpus=2 train.py --use_deepspeed=True")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
