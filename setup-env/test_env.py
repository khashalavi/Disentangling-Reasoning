#!/usr/bin/env python3
"""Test script to verify environment and GPU access."""

import subprocess
import sys

def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def test_imports():
    print_section("Importing key packages")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available – check GPU allocation and CUDA module")
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")

    for pkg in ["transformers", "peft", "datasets", "sentence_transformers"]:
        try:
            mod = __import__(pkg)
            print(f"✓ {pkg} version: {mod.__version__}")
        except ImportError as e:
            print(f"✗ Failed to import {pkg}: {e}")

def test_nvidia_smi():
    print_section("nvidia-smi output")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("GPU info:\n", result.stdout.strip())
        else:
            print("nvidia-smi failed or no GPU found")
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")

def test_custom_modules():
    print_section("Custom project modules (if any)")
    modules = ["model.my_trainer", "model.load_model", "load_data.supervised_dataset"]
    for mod_name in modules:
        try:
            __import__(mod_name)
            print(f"✓ {mod_name} can be imported")
        except ImportError as e:
            print(f"✗ {mod_name} not found: {e}")

def test_simple_torch_operation():
    print_section("Simple PyTorch GPU operation")
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(3, 3).cuda()
            y = x @ x.T
            print("✓ Tensor multiplication on GPU succeeded")
        else:
            print("Skipping GPU operation – CUDA not available")
    except Exception as e:
        print(f"✗ GPU operation failed: {e}")

if __name__ == "__main__":
    print_section("Environment Test")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    test_imports()
    test_nvidia_smi()
    test_custom_modules()
    test_simple_torch_operation()

    print_section("Test completed")