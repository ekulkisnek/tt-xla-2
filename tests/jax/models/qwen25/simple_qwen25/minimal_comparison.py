#!/usr/bin/env python3
"""
Minimal test to identify exact source of differences.
Tests just the first few layers + embedding to isolate issues.
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params

def test_step_by_step():
    """Test layer by layer to isolate the issue."""
    model_path = "../weights"
    
    print("="*60)
    print("STEP-BY-STEP DEBUGGING")
    print("="*60)
    
    # Load config and tokenizer
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test input: same as PyTorch comparison
    test_ids = np.array([[1]], dtype=np.int32)
    print(f"Test input: {test_ids.tolist()}")
    
    # Load JAX model 
    print("\nLoading JAX model (bfloat16 - minimal memory)...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    jax_params = load_params(jax_model, model_path, jnp.bfloat16)
    
    print("✓ JAX model loaded")
    
    # Step 1: Full forward pass (simplified)
    print("\n--- Full Forward Pass (Single Token) ---")
    output = jax_model.apply(
        jax_params,
        input_ids=jnp.array(test_ids),
        return_dict=True
    )
    logits = output["logits"]
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
    print(f"First 5 logits: {[float(x) for x in logits[0,0,:5]]}")
    
    # Step 2: Compare against PyTorch reference  
    print("\n--- Comparison Analysis ---")
    
    # PyTorch reference for token ID 1
    pt_first5 = np.array([4.613307, 1.6183337, 3.1188283, 4.717752, 2.506084])
    jax_first5 = np.array([float(x) for x in logits[0,0,:5]])
    
    print(f"PyTorch first 5: {pt_first5}")
    print(f"JAX first 5: {jax_first5}")
    
    logits_diff = np.abs(jax_first5 - pt_first5)
    max_diff = np.max(logits_diff)
    
    print(f"Differences: {logits_diff}")
    print(f"Max difference: {max_diff:.6f}")
    
    # Analysis
    if max_diff < 0.1:
        print("✅ CLOSE: Differences are small, likely dtype effects")
    elif max_diff < 1.0:
        print("⚠️ MODERATE: Some implementation differences")
    else:
        print("❌ MAJOR: Significant implementation issues")
        
        # Additional debugging for major differences
        print("\nDEBUGGING MAJOR DIFFERENCES:")
        
        # Check weight tying
        embed_w = jax_params["params"]["embed_tokens"]["embedding"]
        lm_head_w = jax_params["params"]["lm_head"]["kernel"]
        
        if embed_w.shape == lm_head_w.T.shape:
            tie_diff = float(jnp.max(jnp.abs(embed_w - lm_head_w.T)))
            print(f"  Weight tying check: max_diff = {tie_diff:.2e}")
        
        # Check if logits have wrong scale
        logits_scale = float(jnp.std(logits))
        pt_scale = float(np.std(pt_first5))  # Rough estimate
        print(f"  Logits scale: JAX={logits_scale:.6f}, PT~{pt_scale:.6f}")
        
        # Check for initialization vs loaded weights
        embed_std = float(jnp.std(embed_w))
        print(f"  Embedding std: {embed_std:.6f} (should be ~0.014 if loaded correctly)")

if __name__ == "__main__":
    test_step_by_step() 