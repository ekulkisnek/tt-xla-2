#!/usr/bin/env python3
"""Quick script to check HuggingFace reference weight statistics"""

import os
import torch
from safetensors import safe_open

model_path = "../../weights/"

# Check layer 0 q_proj statistics from HuggingFace safetensors
safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
print(f"Found safetensors files: {safetensors_files}")

for file in sorted(safetensors_files):
    file_path = os.path.join(model_path, file)
    print(f"\nChecking {file}...")
    
    with safe_open(file_path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            if "model.layers.0.self_attn.q_proj.weight" in key:
                tensor = f.get_tensor(key)
                print(f"Found {key}:")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Mean: {tensor.float().mean().item():.6f}")
                print(f"  Std: {tensor.float().std().item():.6f}")
                print(f"  Min: {tensor.float().min().item():.6f}")
                print(f"  Max: {tensor.float().max().item():.6f}")
                break 