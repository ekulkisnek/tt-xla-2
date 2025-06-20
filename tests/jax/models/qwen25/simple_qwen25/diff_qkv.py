#!/usr/bin/env python3
"""
PHASE 1: Prove the Weight Bug in QKV projections
Compare safetensors vs JAX weights for Q/K/V projections in first layer
"""
import numpy as np
import jax.numpy as jnp
import sys
import os
sys.path.append('.')

def diff_qkv_weights():
    """Compare safetensors vs JAX QKV weights"""
    
    print("🔍 PHASE 1: Proving Weight Bug in QKV Projections")
    print("="*60)
    print("📝 Using safetensors direct comparison (avoids memory issues)")
    
    # Use our local weights
    from safetensors import safe_open
    
    # Load JAX model
    print("📥 Loading JAX model...")
    from q25_jax_instruct import Qwen25ForCausalLM, load_params
    import json
    
    config_path = "../instruct_weights/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    params = load_params(model, "../instruct_weights", jnp.float32)
    
    print("📊 Comparing weights from safetensors vs JAX...")
    layer = 0
    
    # Compare with safetensors directly
    safetensors_file = "../instruct_weights/model-00001-of-00004.safetensors"
    
    print(f"\n🔍 Layer {layer} QKV comparison:")
    
    with safe_open(safetensors_file, framework="np") as f:
        for proj in ["q", "k", "v"]:
            pt_key = f"model.layers.{layer}.self_attn.{proj}_proj.weight"
            jax_key_path = ["params", f"layers_{layer}", "self_attn", f"{proj}_proj", "kernel"]
            
            if pt_key in f.keys():
                pt_w = f.get_tensor(pt_key).astype(np.float32)
                
                # Navigate JAX params
                jx_w = params
                for key in jax_key_path:
                    jx_w = jx_w[key]
                jx_w = np.array(jx_w, dtype=np.float32)
                
                print(f"\n  {proj.upper()}_proj:")
                print(f"    Safetensors shape: {pt_w.shape}")
                print(f"    JAX shape: {jx_w.shape}")
                
                # Check transpose (Flax uses transposed kernels)
                if pt_w.shape != jx_w.shape:
                    print(f"    JAX transposed shape: {jx_w.T.shape}")
                    diff = np.abs(pt_w - jx_w.T).max()
                    print(f"    Max diff (with transpose): {diff:.6f}")
                else:
                    diff = np.abs(pt_w - jx_w).max()
                    print(f"    Max diff (no transpose): {diff:.6f}")
                
                # Status
                if diff < 1e-5:
                    print(f"    ✅ MATCH")
                else:
                    print(f"    ❌ MISMATCH (diff: {diff:.6f})")
                    print(f"    🎯 Expected: ~0.55 for broken loader")
            else:
                print(f"    ❌ Key {pt_key} not found in safetensors")
    
    print(f"\n{'='*60}")
    print("📋 PHASE 1 Results:")
    print("   If all show 'MATCH' → weight loading is correct")
    print("   If any show 'MISMATCH' → proceed to PHASE 2 (fix loader)")

if __name__ == "__main__":
    diff_qkv_weights() 