#!/usr/bin/env python3
"""
Verify that JAX weights match PyTorch weights exactly
"""
import sys
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer
import jax.numpy as jnp
import json
import os
import numpy as np
from safetensors import safe_open

def verify_weight_loading():
    """Check specific weights against safetensors files"""
    
    model_path = "../instruct_weights"
    
    print("🔍 VERIFYING WEIGHT LOADING")
    print("="*50)
    
    # Load our JAX model
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # First, let's see what our JAX params structure looks like
    print(f"\n🔹 JAX parameter structure:")
    def print_nested_keys(d, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
        for key in sorted(d.keys())[:5]:  # Show first 5 keys
            if isinstance(d[key], dict):
                print(f"  {prefix}{key}/")
                print_nested_keys(d[key], prefix + "  ", max_depth, current_depth + 1)
            else:
                shape = getattr(d[key], 'shape', 'no shape')
                print(f"  {prefix}{key}: {shape}")
        if len(d.keys()) > 5:
            print(f"  {prefix}... ({len(d.keys())} total keys)")
    
    print_nested_keys(params)
    
    # Check a few critical weights directly from safetensors
    safetensors_file = os.path.join(model_path, "model-00001-of-00004.safetensors")
    
    print(f"\n🔹 Checking weights from: {safetensors_file}")
    
    with safe_open(safetensors_file, framework="np") as f:
        # Check embedding weights
        if "model.embed_tokens.weight" in f.keys():
            pt_embedding = f.get_tensor("model.embed_tokens.weight")
            jax_embedding = np.array(params["params"]["embed_tokens"]["embedding"])
            
            print(f"\nEmbedding weights:")
            print(f"  PyTorch shape: {pt_embedding.shape}")
            print(f"  JAX shape: {jax_embedding.shape}")
            print(f"  PyTorch dtype: {pt_embedding.dtype}")
            print(f"  JAX dtype: {jax_embedding.dtype}")
            
            # Check if they match (accounting for transpose)
            if pt_embedding.shape != jax_embedding.shape:
                jax_embedding = jax_embedding.T
                print(f"  JAX transposed shape: {jax_embedding.shape}")
            
            diff = np.abs(pt_embedding.astype(np.float32) - jax_embedding.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Max difference: {max_diff}")
            print(f"  Mean difference: {mean_diff}")
            print(f"  Match: {'✅' if max_diff < 1e-5 else '❌'}")
        
        # Check first layer attention weights  
        if "model.layers.0.self_attn.q_proj.weight" in f.keys():
            pt_q_proj = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            jax_q_proj = np.array(params["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"])
            
            print(f"\nFirst layer Q projection:")
            print(f"  PyTorch shape: {pt_q_proj.shape}")
            print(f"  JAX shape: {jax_q_proj.shape}")
            
            # Account for transpose
            if pt_q_proj.shape != jax_q_proj.shape:
                jax_q_proj = jax_q_proj.T
                print(f"  JAX transposed shape: {jax_q_proj.shape}")
            
            diff = np.abs(pt_q_proj.astype(np.float32) - jax_q_proj.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Max difference: {max_diff}")
            print(f"  Mean difference: {mean_diff}")
            print(f"  Match: {'✅' if max_diff < 1e-5 else '❌'}")
    
    print(f"\n🔹 Available keys in safetensors file:")
    with safe_open(safetensors_file, framework="np") as f:
        keys = list(f.keys())[:10]  # Show first 10
        for key in keys:
            print(f"  {key}")
        print(f"  ... ({len(list(f.keys()))} total keys)")

if __name__ == "__main__":
    verify_weight_loading() 