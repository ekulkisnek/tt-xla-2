#!/usr/bin/env python3
"""
Systematic parity comparison script for S-2 and S-3 testing.
Based on the finish-line checklist for proving numerical identical ports.
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_inference import Qwen25ForCausalLM, load_params

def torch_logits(pt, ids):
    """Get PyTorch logits for comparison."""
    with torch.no_grad():
        return pt(ids).logits.cpu().numpy()

def run_s2_single_token_test():
    """S-2: 1-token logits test - max Δ < 1e-6 in float32"""
    print("="*60)
    print("S-2: SINGLE TOKEN LOGITS TEST (float32)")
    print("="*60)
    
    # Load tokenizer and create test input
    tokenizer = AutoTokenizer.from_pretrained("../weights") 
    ids_np = np.array([[1]], dtype=np.int32)  # Single token
    ids_torch = torch.tensor(ids_np)
    
    print(f"Test input: {ids_np.tolist()}")
    
    # Load PyTorch model (float32)
    print("Loading PyTorch model (float32)...")
    pt = AutoModelForCausalLM.from_pretrained(
        "../weights", 
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Load JAX model (float32)  
    print("Loading JAX model (float32)...")
    with open("../weights/config.json", 'r') as f:
        config = json.load(f)
    
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    jax_params = load_params(jax_model, "../weights", jnp.float32)
    
    # Get logits from both models
    print("Computing PyTorch logits...")
    pt_logits = torch_logits(pt, ids_torch)
    
    print("Computing JAX logits...")
    jax_out = jax_model.apply(jax_params, ids_np, return_dict=True)
    jax_logits = np.array(jax_out["logits"])
    
    # Compare
    print(f"\nResults:")
    print(f"PyTorch logits shape: {pt_logits.shape}")
    print(f"JAX logits shape: {jax_logits.shape}")
    
    # Numerical comparison
    max_abs_diff = np.max(np.abs(pt_logits - jax_logits))
    mean_abs_diff = np.mean(np.abs(pt_logits - jax_logits))
    
    print(f"Max absolute Δ: {max_abs_diff:.2e}")
    print(f"Mean absolute Δ: {mean_abs_diff:.2e}")
    
    # First 5 logits comparison
    pt_first5 = pt_logits[0, 0, :5]
    jax_first5 = jax_logits[0, 0, :5]
    
    print(f"\nFirst 5 logits comparison:")
    print(f"PyTorch: {pt_first5}")
    print(f"JAX:     {jax_first5}")
    print(f"Diff:    {np.abs(pt_first5 - jax_first5)}")
    
    # Pass criterion: max Δ < 1e-6
    threshold = 1e-6
    if max_abs_diff < threshold:
        print(f"\n✅ S-2 PASSED: max Δ ({max_abs_diff:.2e}) < {threshold:.0e}")
        return True
    else:
        print(f"\n❌ S-2 FAILED: max Δ ({max_abs_diff:.2e}) >= {threshold:.0e}")
        return False

def run_s3_multi_token_test(): 
    """S-3: 16-token logits test (covers RoPE) - max Δ < 1e-4"""
    print("\n" + "="*60)
    print("S-3: MULTI-TOKEN LOGITS TEST (float32)")
    print("="*60)
    
    # Load tokenizer and create test input  
    tokenizer = AutoTokenizer.from_pretrained("../weights")
    ids_np = np.array([[1, 42, 600, 17, 5, 8, 9, 2, 1234, 56, 789, 101, 202, 303, 404, 505]], dtype=np.int32)  # 16 tokens
    ids_torch = torch.tensor(ids_np)
    
    print(f"Test input: {ids_np.tolist()}")
    print(f"Input length: {ids_np.shape[1]} tokens")
    
    # Load PyTorch model (float32)
    print("Loading PyTorch model (float32)...")
    pt = AutoModelForCausalLM.from_pretrained(
        "../weights", 
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Load JAX model (float32)
    print("Loading JAX model (float32)...")
    with open("../weights/config.json", 'r') as f:
        config = json.load(f)
    
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    jax_params = load_params(jax_model, "../weights", jnp.float32)
    
    # Get logits from both models
    print("Computing PyTorch logits...")
    pt_logits = torch_logits(pt, ids_torch)
    
    print("Computing JAX logits...")
    jax_out = jax_model.apply(jax_params, ids_np, return_dict=True)
    jax_logits = np.array(jax_out["logits"])
    
    # Compare
    print(f"\nResults:")
    print(f"PyTorch logits shape: {pt_logits.shape}")
    print(f"JAX logits shape: {jax_logits.shape}")
    
    # Numerical comparison
    max_abs_diff = np.max(np.abs(pt_logits - jax_logits))
    mean_abs_diff = np.mean(np.abs(pt_logits - jax_logits))
    
    print(f"Max absolute Δ: {max_abs_diff:.2e}")
    print(f"Mean absolute Δ: {mean_abs_diff:.2e}")
    
    # Last token logits comparison (most sensitive due to RoPE)
    pt_last = pt_logits[0, -1, :5]
    jax_last = jax_logits[0, -1, :5]
    
    print(f"\nLast token first 5 logits comparison:")
    print(f"PyTorch: {pt_last}")
    print(f"JAX:     {jax_last}")
    print(f"Diff:    {np.abs(pt_last - jax_last)}")
    
    # Pass criterion: max Δ < 1e-4
    threshold = 1e-4
    if max_abs_diff < threshold:
        print(f"\n✅ S-3 PASSED: max Δ ({max_abs_diff:.2e}) < {threshold:.0e}")
        return True
    else:
        print(f"\n❌ S-3 FAILED: max Δ ({max_abs_diff:.2e}) >= {threshold:.0e}")
        return False

def main():
    """Run S-2 and S-3 parity tests."""
    print("SYSTEMATIC PARITY TESTING")
    print("Testing numerical identical between PyTorch and JAX implementations")
    
    try:
        # Run S-2: Single token test
        s2_passed = run_s2_single_token_test()
        
        # Run S-3: Multi-token test  
        s3_passed = run_s3_multi_token_test()
        
        # Final assessment
        print("\n" + "="*60)
        print("FINAL ASSESSMENT")
        print("="*60)
        
        if s2_passed and s3_passed:
            print("🎉 BOTH TESTS PASSED: Numerical parity achieved!")
            print("   Your JAX port is numerically identical to PyTorch.")
        elif s2_passed:
            print("⚠️  S-2 passed but S-3 failed: RoPE or multi-token issues")
            print("   Single token works but multi-token has problems.")  
        else:
            print("❌ S-2 failed: Fundamental implementation differences")
            print("   Basic forward pass has numerical issues.")
            
        return s2_passed and s3_passed
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 