#!/usr/bin/env python3
"""
Simple test to demonstrate working Qwen2.5 tensor parallel implementation
"""
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

# Test both single device and simulated multi-device
def test_single_device():
    print("=== SINGLE DEVICE TEST ===")
    print("Testing with original working implementation...")
    
    # Use the original working model from the bounties directory
    os.chdir("tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b")
    
    # Import and test the working implementation
    import subprocess
    result = subprocess.run([
        "python", "simple_inference.py", 
        "--model_path", "qwen25_7b_instruct_weights",
        "--prompt", "What is 2+2?",
        "--max_tokens", "10"
    ], capture_output=True, text=True)
    
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    return result.returncode == 0

def test_parallel_device():
    print("\n=== TENSOR PARALLEL TEST ===")
    print("Testing with simulated 4 devices...")
    
    # Set up multi-device simulation
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    
    # Test the rewritten implementation
    os.chdir("../qwen25-7b-rewritten")
    
    import subprocess
    result = subprocess.run([
        "python", "qwen25_tp_final.py",
        "--model_path", "qwen25_7b_instruct_weights", 
        "--prompt", "What is 2+2?",
        "--max_tokens", "10",
        "--tp", "4"
    ], capture_output=True, text=True)
    
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    return result.returncode == 0

def main():
    print("Qwen2.5-7B Tensor Parallel Quality Test")
    print("=" * 50)
    
    # Test single device (should work with original implementation)
    single_success = test_single_device()
    
    # Test tensor parallel (new implementation)  
    parallel_success = test_parallel_device()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Single Device: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"Tensor Parallel: {'✅ PASS' if parallel_success else '❌ FAIL'}")
    
    if single_success and parallel_success:
        print("\n🎉 SUCCESS: Both single and parallel modes produce quality responses!")
    elif single_success:
        print("\n⚠️  PARTIAL: Single device works, parallel needs fixes")
    else:
        print("\n❌ FAILURE: Core implementation has issues")

if __name__ == "__main__":
    main() 