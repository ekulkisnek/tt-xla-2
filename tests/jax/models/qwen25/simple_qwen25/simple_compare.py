#!/usr/bin/env python3
"""
Simplified PyTorch vs JAX Qwen2.5 Comparison
Focus on key validation stages with memory efficiency.
"""

import os
import json
import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jax.numpy as jnp
from simple_inference import Qwen25ForCausalLM, load_params

def main():
    model_path = "../weights"
    
    print("="*80)
    print("SIMPLIFIED QWEN2.5 PYTORCH vs JAX COMPARISON")
    print("="*80)
    
    # Load config and tokenizer
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test input
    test_ids = np.array([[1]], dtype=np.int32)
    prompt = "AI is"
    
    print(f"Test input: {test_ids.tolist()}")
    print(f"Test prompt: '{prompt}'")
    
    # === PYTORCH REFERENCE ===
    print("\n" + "="*50)
    print("PYTORCH REFERENCE")
    print("="*50)
    
    print("Loading PyTorch model...")
    pt_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    pt_model.eval()
    
    # Single token test
    with torch.no_grad():
        pt_logits = pt_model(torch.tensor(test_ids)).logits.cpu().numpy()
    
    print(f"PyTorch logits shape: {pt_logits.shape}")
    print(f"PyTorch logits range: [{np.min(pt_logits):.3f}, {np.max(pt_logits):.3f}]")
    print(f"PyTorch first 5 logits: {pt_logits[0,0,:5]}")
    
    # Greedy generation test
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        pt_outputs = pt_model.generate(
            inputs.input_ids,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    pt_generated = pt_outputs[0][inputs.input_ids.shape[1]:]
    pt_text = tokenizer.decode(pt_generated, skip_special_tokens=True)
    print(f"PyTorch generated: '{pt_text}'")
    
    # Save PyTorch results
    torch.save({
        'logits': pt_logits,
        'generated_text': pt_text,
        'generated_tokens': pt_generated.tolist()
    }, 'pytorch_results.pt')
    
    # Cleanup PyTorch
    del pt_model, pt_outputs
    gc.collect()
    
    # === JAX IMPLEMENTATION ===
    print("\n" + "="*50)
    print("JAX IMPLEMENTATION")
    print("="*50)
    
    print("Loading JAX model...")
    jax_model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    jax_params = load_params(jax_model, model_path, jnp.float32)
    
    # Single token test
    jax_out = jax_model.apply(
        jax_params,
        input_ids=jnp.array(test_ids),
        return_dict=True
    )
    jax_logits = np.array(jax_out["logits"])
    
    print(f"JAX logits shape: {jax_logits.shape}")
    print(f"JAX logits range: [{np.min(jax_logits):.3f}, {np.max(jax_logits):.3f}]")
    print(f"JAX first 5 logits: {jax_logits[0,0,:5]}")
    
    # Greedy generation test
    jax_inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jax_inputs["input_ids"]
    
    generated_tokens = []
    current_ids = input_ids
    past_key_values = None
    
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
    
    # === COMPARISON ===
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    # Load PyTorch results
    pt_results = torch.load('pytorch_results.pt')
    pt_saved_logits = pt_results['logits']
    pt_saved_text = pt_results['generated_text']
    
    # Compare logits
    max_diff = np.max(np.abs(pt_saved_logits - jax_logits))
    mean_diff = np.mean(np.abs(pt_saved_logits - jax_logits))
    
    print(f"Logits max difference: {max_diff:.2e}")
    print(f"Logits mean difference: {mean_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✓ LOGITS MATCH: Perfect numerical precision!")
    elif max_diff < 1e-4:
        print("✓ LOGITS CLOSE: Good numerical precision")
    else:
        print("❌ LOGITS DIFFER: Significant numerical differences")
    
    # Compare generated text
    print(f"\nText comparison:")
    print(f"PyTorch: '{pt_saved_text}'")
    print(f"JAX:     '{jax_text}'")
    
    if pt_saved_text == jax_text:
        print("✓ TEXT MATCHES: Perfect generation parity!")
    else:
        print("❌ TEXT DIFFERS: Generation discrepancy")
    
    # Overall assessment
    print("\n" + "="*50)
    print("OVERALL ASSESSMENT")
    print("="*50)
    
    if max_diff < 1e-6 and pt_saved_text == jax_text:
        print("🎉 PERFECT PARITY: JAX implementation matches PyTorch exactly!")
    elif max_diff < 1e-4:
        print("✅ GOOD PARITY: JAX implementation is very close to PyTorch")
    else:
        print("⚠️  PARITY ISSUES: JAX implementation needs debugging")
    
    # Cleanup
    os.remove('pytorch_results.pt')

if __name__ == "__main__":
    main() 