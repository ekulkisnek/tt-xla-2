#!/usr/bin/env python3
"""
PHASE 1: Lightweight Weight Bug Proof (No Full Model Loading)
Compare specific QKV weights from safetensors vs our weight loading logic
"""
import numpy as np
import jax.numpy as jnp
from safetensors import safe_open
import json
import os

def get_param_path(key):
    """Replicate the param path mapping from our loader"""
    
    # This should match the logic in q25_jax_instruct.py load_params
    if "embed_tokens.weight" in key:
        return ["embed_tokens", "embedding"]
    
    if "layers." in key:
        parts = key.split(".")
        layer_num = parts[2]  # parts[2] is the layer number, not parts[1]
        layer_key = f"layers_{layer_num}"
        
        if "self_attn" in key:
            if "q_proj.weight" in key:
                return [layer_key, "self_attn", "q_proj", "kernel"]
            elif "k_proj.weight" in key:
                return [layer_key, "self_attn", "k_proj", "kernel"] 
            elif "v_proj.weight" in key:
                return [layer_key, "self_attn", "v_proj", "kernel"]
            elif "o_proj.weight" in key:
                return [layer_key, "self_attn", "o_proj", "kernel"]
        
        if "mlp" in key:
            if "gate_proj.weight" in key:
                return [layer_key, "mlp", "gate_proj", "kernel"]
            elif "up_proj.weight" in key:
                return [layer_key, "mlp", "up_proj", "kernel"]
            elif "down_proj.weight" in key:
                return [layer_key, "mlp", "down_proj", "kernel"]
    
    return None

def transpose_if_needed(param, key):
    """Replicate transpose logic"""
    # Linear layers (2D) get transposed
    if len(param.shape) == 2:
        return param.T
    return param

def diff_qkv_light():
    """Compare QKV weights without loading full model"""
    
    print("🔍 PHASE 1: Lightweight QKV Weight Bug Proof")
    print("="*60)
    
    layer = 0
    safetensors_file = "../instruct_weights/model-00001-of-00004.safetensors"
    
    print(f"📊 Comparing Layer {layer} QKV weights...")
    print(f"📁 Source: {safetensors_file}")
    
    with safe_open(safetensors_file, framework="np") as f:
        
        for proj in ["q", "k", "v"]:
            pt_key = f"model.layers.{layer}.self_attn.{proj}_proj.weight"
            
            if pt_key in f.keys():
                print(f"\n🔍 {proj.upper()}_proj:")
                
                # Get original safetensors weight
                safetensors_weight = f.get_tensor(pt_key).astype(np.float32)
                print(f"  Safetensors shape: {safetensors_weight.shape}")
                
                # Simulate our loading process
                param_path = get_param_path(pt_key)
                print(f"  JAX param path: {param_path}")
                print(f"  Debug: key='{pt_key}', parts={pt_key.split('.')}")
                
                # Apply our transpose logic
                jax_weight = transpose_if_needed(safetensors_weight, pt_key)
                print(f"  JAX processed shape: {jax_weight.shape}")
                
                # Compare shapes first
                if safetensors_weight.shape == jax_weight.shape:
                    # Same shape - test both with and without transpose
                    diff_direct = np.abs(safetensors_weight - jax_weight).max()
                    diff_transposed = np.abs(safetensors_weight - jax_weight.T).max()
                    print(f"  Difference (no transpose): {diff_direct:.6f}")
                    print(f"  Difference (with transpose): {diff_transposed:.6f}")
                    diff = min(diff_direct, diff_transposed)
                    if diff_transposed < diff_direct:
                        print(f"  🔧 Square matrix should NOT be transposed!")
                else:
                    # Different shapes - compare with transpose
                    diff = np.abs(safetensors_weight - jax_weight.T).max()
                    print(f"  Difference (with transpose): {diff:.6f}")
                
                if diff < 1e-6:
                    print(f"  ✅ PROCESSING CORRECT")
                else:
                    print(f"  ❌ PROCESSING BUG DETECTED")
                    print(f"     🔧 This suggests transpose/path mapping issue")
                
                # Additional debug info
                print(f"  Original weight range: [{safetensors_weight.min():.3f}, {safetensors_weight.max():.3f}]")
                print(f"  First few values: {safetensors_weight.flatten()[:5]}")
            else:
                print(f"❌ Key {pt_key} not found")
    
    print(f"\n{'='*60}")
    print("📋 DIAGNOSIS:")
    print("  ✅ All 'PROCESSING CORRECT' → Bug is elsewhere")
    print("  ❌ Any 'PROCESSING BUG' → Fix transpose/mapping logic")
    print("")
    print("🔄 Next: If processing is correct, compare with working base model")

if __name__ == "__main__":
    diff_qkv_light() 