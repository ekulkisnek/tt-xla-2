#!/usr/bin/env python3
"""
Test JAX model with the same inputs as PyTorch for comparison.
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params

def main():
    model_path = "../weights"
    
    print("="*60)
    print("JAX MODEL TEST (for PyTorch comparison)")
    print("="*60)
    
    # Load config and tokenizer
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Same test inputs as PyTorch comparison
    test_ids = np.array([[1]], dtype=np.int32)
    prompt = "AI is"
    
    print(f"Test input: {test_ids.tolist()}")
    print(f"Test prompt: '{prompt}'")
    
    # Load JAX model (using bfloat16 to reduce memory)
    print("\nLoading JAX model (bfloat16)...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    jax_params = load_params(jax_model, model_path, jnp.bfloat16)
    
    print("✓ JAX model loaded successfully")
    
    # Single token test
    print("\n--- Single Token Test ---")
    jax_out = jax_model.apply(
        jax_params,
        input_ids=jnp.array(test_ids),
        return_dict=True
    )
    jax_logits = np.array(jax_out["logits"])
    
    print(f"JAX logits shape: {jax_logits.shape}")
    print(f"JAX logits range: [{float(np.min(jax_logits)):.3f}, {float(np.max(jax_logits)):.3f}]")
    print(f"JAX first 5 logits: {[float(x) for x in jax_logits[0,0,:5]]}")
    
    # Greedy generation test
    print("\n--- Greedy Generation Test ---")
    jax_inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jax_inputs["input_ids"]
    
    generated_tokens = []
    current_ids = input_ids
    past_key_values = None
    
    print(f"Starting generation from: '{prompt}'")
    
    for step in range(16):
        outputs = jax_model.apply(
            jax_params,
            input_ids=jnp.array(current_ids),
            past_key_values=past_key_values,
            return_dict=True
        )
        
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        next_token = jnp.argmax(logits[0, -1, :])
        generated_tokens.append(int(next_token))
        
        current_ids = np.array([[int(next_token)]], dtype=np.int32)
        
        if int(next_token) == tokenizer.eos_token_id:
            break
    
    jax_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"JAX generated: '{jax_text}'")
    
    # Compare to known PyTorch results
    print("\n" + "="*60)
    print("COMPARISON TO PYTORCH")
    print("="*60)
    
    # PyTorch reference data (from previous run)
    pt_logits_first5 = np.array([4.613307, 1.6183337, 3.1188283, 4.717752, 2.506084])
    pt_text = " a technology that is changing the world. It is a technology that is changing the"
    
    print("PyTorch reference:")
    print(f"  First 5 logits: {pt_logits_first5}")
    print(f"  Generated text: '{pt_text}'")
    
    print("\nJAX results:")
    print(f"  First 5 logits: {jax_logits[0,0,:5]}")
    print(f"  Generated text: '{jax_text}'")
    
    # Numerical comparison (approximate due to dtype difference)
    logits_diff = np.abs(jax_logits[0,0,:5] - pt_logits_first5)
    max_logits_diff = np.max(logits_diff)
    
    print(f"\nNumerical comparison:")
    print(f"  Max logits difference: {max_logits_diff:.3f}")
    print(f"  Per-element differences: {logits_diff}")
    
    # Text comparison
    print(f"\nText comparison:")
    if pt_text == jax_text:
        print("✓ TEXT MATCHES: Perfect generation parity!")
    else:
        print("❌ TEXT DIFFERS: Generation discrepancy")
        print(f"  Length: PT={len(pt_text)}, JAX={len(jax_text)}")
    
    # Assessment
    print(f"\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)
    
    if pt_text == jax_text:
        print("🎉 GENERATION PARITY: JAX produces identical text to PyTorch!")
    elif max_logits_diff < 1.0:
        print("✅ CLOSE PARITY: JAX is numerically close to PyTorch")
    else:
        print("⚠️  SIGNIFICANT DIFFERENCES: JAX diverges from PyTorch")
    
    print(f"\nNote: Comparison done with JAX in bfloat16 vs PyTorch float32")
    print(f"Some numerical differences are expected due to precision differences.")

if __name__ == "__main__":
    main() 