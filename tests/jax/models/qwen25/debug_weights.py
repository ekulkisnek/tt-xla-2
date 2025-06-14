#!/usr/bin/env python3
"""
Debug weight loading by comparing embedding outputs directly.
"""
import torch
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoModelForCausalLM
import json
import sys
import os
sys.path.append('jax_scripts')
import simple_inference as si

def compare_embeddings():
    """Compare PyTorch and JAX embedding outputs."""
    print("=== EMBEDDING COMPARISON ===")
    
    # Test token
    token_id = 1
    
    # PyTorch embedding
    print("Loading PyTorch model...")
    pt_model = AutoModelForCausalLM.from_pretrained("weights", torch_dtype=torch.bfloat16, device_map="cpu")
    pt_embed = pt_model.model.embed_tokens(torch.tensor([[token_id]]))
    pt_embed_np = pt_embed.detach().cpu().float().numpy()
    
    print(f"PyTorch embedding shape: {pt_embed_np.shape}")
    print(f"PyTorch embedding std: {pt_embed_np.std():.6f}")
    print(f"PyTorch embedding mean: {pt_embed_np.mean():.6f}")
    print(f"PyTorch embedding sample: {pt_embed_np[0, 0, :5]}")
    
    del pt_model, pt_embed
    
    # JAX embedding
    print("\nLoading JAX model...")
    jax.config.update("jax_enable_x64", False)
    
    with open("weights/config.json", 'r') as f:
        config = json.load(f)
    
    model = si.Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    params = si.load_params(model, "weights", jnp.bfloat16)
    
    # Get embedding weight directly
    embed_weight = params['params']['embed_tokens']['embedding']
    print(f"JAX embedding weight shape: {embed_weight.shape}")
    print(f"JAX embedding weight std: {float(jnp.std(embed_weight)):.6f}")
    
    # Extract single embedding
    jax_embed = embed_weight[token_id:token_id+1, :]  # [1, hidden_size]
    jax_embed = jax_embed[None, :]  # [1, 1, hidden_size]
    
    print(f"JAX embedding shape: {jax_embed.shape}")
    print(f"JAX embedding std: {float(jnp.std(jax_embed)):.6f}")
    print(f"JAX embedding mean: {float(jnp.mean(jax_embed)):.6f}")
    print(f"JAX embedding sample: {[float(x) for x in jax_embed[0, 0, :5]]}")
    
    # Compare
    diff = np.abs(np.array(jax_embed) - pt_embed_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n=== COMPARISON ===")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✅ Embeddings match closely!")
        return True
    else:
        print("❌ Embeddings differ significantly!")
        return False

def compare_weight_stats():
    """Compare overall weight statistics."""
    print("\n=== WEIGHT STATISTICS COMPARISON ===")
    
    # PyTorch weights
    print("Loading PyTorch model...")
    pt_model = AutoModelForCausalLM.from_pretrained("weights", torch_dtype=torch.bfloat16, device_map="cpu")
    
    pt_embed_weight = pt_model.model.embed_tokens.weight.detach().cpu().float().numpy()
    pt_lm_head_weight = pt_model.lm_head.weight.detach().cpu().float().numpy()
    
    print(f"PyTorch embed weight: shape={pt_embed_weight.shape}, std={pt_embed_weight.std():.6f}")
    print(f"PyTorch lm_head weight: shape={pt_lm_head_weight.shape}, std={pt_lm_head_weight.std():.6f}")
    
    del pt_model
    
    # JAX weights
    print("\nLoading JAX weights...")
    jax.config.update("jax_enable_x64", False)
    
    with open("weights/config.json", 'r') as f:
        config = json.load(f)
    
    model = si.Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    params = si.load_params(model, "weights", jnp.bfloat16)
    
    jax_embed_weight = np.array(params['params']['embed_tokens']['embedding'])
    jax_lm_head_weight = np.array(params['params']['lm_head']['kernel'])
    
    print(f"JAX embed weight: shape={jax_embed_weight.shape}, std={float(jax_embed_weight.std()):.6f}")
    print(f"JAX lm_head weight: shape={jax_lm_head_weight.shape}, std={float(jax_lm_head_weight.std()):.6f}")
    
    # Compare embed weights
    if pt_embed_weight.shape == jax_embed_weight.shape:
        embed_diff = np.abs(pt_embed_weight - jax_embed_weight)
        print(f"Embedding weight max diff: {embed_diff.max():.2e}")
    else:
        print(f"⚠️ Embedding shape mismatch: PT={pt_embed_weight.shape} vs JAX={jax_embed_weight.shape}")
    
    # Compare LM head weights (accounting for transpose)
    if pt_lm_head_weight.shape == jax_lm_head_weight.T.shape:
        lm_head_diff = np.abs(pt_lm_head_weight - jax_lm_head_weight.T)
        print(f"LM head weight max diff (with transpose): {lm_head_diff.max():.2e}")
    elif pt_lm_head_weight.shape == jax_lm_head_weight.shape:
        lm_head_diff = np.abs(pt_lm_head_weight - jax_lm_head_weight)
        print(f"LM head weight max diff (no transpose): {lm_head_diff.max():.2e}")
    else:
        print(f"⚠️ LM head shape mismatch: PT={pt_lm_head_weight.shape} vs JAX={jax_lm_head_weight.shape}")

if __name__ == "__main__":
    print("Weight Loading Debug Script")
    print("="*50)
    
    try:
        embed_ok = compare_embeddings()
        compare_weight_stats()
        
        if embed_ok:
            print("\n✅ Weight loading appears correct")
        else:
            print("\n❌ Weight loading has issues")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 