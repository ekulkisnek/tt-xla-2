#!/usr/bin/env python3
"""
Test JAX model in float32 for precise comparison with PyTorch.
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
    print("JAX MODEL TEST (FLOAT32 - PRECISE COMPARISON)")
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
    
    # Load JAX model in FLOAT32 for precise comparison
    print("\nLoading JAX model (FLOAT32)...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    jax_params = load_params(jax_model, model_path, jnp.float32)
    
    print("✓ JAX model loaded successfully in float32")
    
    # Single token test
    print("\n--- Single Token Test (Float32) ---")
    jax_out = jax_model.apply(
        jax_params,
        input_ids=jnp.array(test_ids),
        return_dict=True
    )
    jax_logits = np.array(jax_out["logits"])
    
    print(f"JAX logits shape: {jax_logits.shape}")
    print(f"JAX logits range: [{float(np.min(jax_logits)):.6f}, {float(np.max(jax_logits)):.6f}]")
    print(f"JAX first 5 logits: {[float(x) for x in jax_logits[0,0,:5]]}")
    
    # Greedy generation test (short)
    print("\n--- Greedy Generation Test (Float32) ---")
    jax_inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jax_inputs["input_ids"]
    
    generated_tokens = []
    current_ids = input_ids
    past_key_values = None
    
    print(f"Starting generation from: '{prompt}'")
    
    for step in range(10):  # Shorter test
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
    print("PRECISE COMPARISON TO PYTORCH (Float32)")
    print("="*60)
    
    # PyTorch reference data (from previous run)
    pt_logits_first5 = np.array([4.613307, 1.6183337, 3.1188283, 4.717752, 2.506084])
    pt_text = " a technology that is changing the world. It is a technology that is changing the"
    
    print("PyTorch reference (float32):")
    print(f"  First 5 logits: {pt_logits_first5}")
    print(f"  Generated text: '{pt_text}'")
    
    print("\nJAX results (float32):")
    print(f"  First 5 logits: {[float(x) for x in jax_logits[0,0,:5]]}")
    print(f"  Generated text: '{jax_text}'")
    
    # Precise numerical comparison
    jax_first5 = np.array([float(x) for x in jax_logits[0,0,:5]])
    logits_diff = np.abs(jax_first5 - pt_logits_first5)
    max_logits_diff = np.max(logits_diff)
    
    print(f"\nPrecise numerical comparison (both float32):")
    print(f"  Max logits difference: {max_logits_diff:.6f}")
    print(f"  Per-element differences: {logits_diff}")
    
    # Detailed analysis
    print(f"\nDetailed logits analysis:")
    for i, (pt_val, jax_val, diff) in enumerate(zip(pt_logits_first5, jax_first5, logits_diff)):
        print(f"  Logit {i}: PT={pt_val:.6f}, JAX={jax_val:.6f}, diff={diff:.6f}")
    
    # Text comparison
    print(f"\nText comparison:")
    if pt_text == jax_text:
        print("✓ TEXT MATCHES: Perfect generation parity!")
    else:
        print("❌ TEXT DIFFERS: Generation discrepancy")
        print(f"  Length: PT={len(pt_text)}, JAX={len(jax_text)}")
    
    # Assessment with precise thresholds
    print(f"\n" + "="*60)
    print("PRECISE ASSESSMENT (Float32 vs Float32)")
    print("="*60)
    
    if max_logits_diff < 1e-6:
        print("🎉 PERFECT PARITY: JAX matches PyTorch to machine precision!")
    elif max_logits_diff < 1e-4:
        print("✅ EXCELLENT PARITY: JAX is very close to PyTorch")
    elif max_logits_diff < 1e-2:
        print("✅ GOOD PARITY: JAX has acceptable differences")
    elif max_logits_diff < 1.0:
        print("⚠️  MODERATE DIFFERENCES: JAX needs some debugging")
    else:
        print("❌ MAJOR DIFFERENCES: JAX has significant implementation issues")
        print(f"   Max difference {max_logits_diff:.6f} suggests fundamental bugs")

if __name__ == "__main__":
    main() 