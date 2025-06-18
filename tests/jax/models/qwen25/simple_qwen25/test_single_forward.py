#!/usr/bin/env python3
"""
Minimal test to compare single forward pass between PyTorch and JAX
"""
import numpy as np, jax, jax.numpy as jnp, json
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params

def test_jax_forward():
    model_path = "../weights"
    
    # Load tokenizer and create input
    tok = AutoTokenizer.from_pretrained(model_path)
    test_text = "Hello world"
    input_ids = tok(test_text, return_tensors="np")["input_ids"]
    
    print(f"Input: {test_text}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # Load JAX model with float32 for precise comparison
    print("Loading JAX model...")
    cfg = json.load(open(f"{model_path}/config.json"))
    print(f"Config check - rope_theta: {cfg['rope_theta']}")
    
    model = Qwen25ForCausalLM(cfg, dtype=jnp.float32)
    params = load_params(model, model_path, jnp.float32)
    
    # Single forward pass
    print("Running forward pass...")
    outputs = model.apply(params, input_ids=input_ids, return_dict=True)
    logits = outputs["logits"][0, -1]  # Last token logits
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{float(jnp.min(logits)):.3f}, {float(jnp.max(logits)):.3f}]")
    
    # Get top predictions
    top_5_indices = jnp.argsort(logits)[-5:][::-1]
    top_5_logits = logits[top_5_indices]
    
    print("Top 5 predictions:")
    for i, (idx, logit) in enumerate(zip(top_5_indices, top_5_logits)):
        token_text = tok.decode([int(idx)])
        print(f"  {i+1}. Token {int(idx)}: {repr(token_text)} (logit: {float(logit):.3f})")
    
    # Test next token generation
    next_token_id = int(jnp.argmax(logits))
    next_text = tok.decode([next_token_id])
    print(f"\nGreedy next token: {next_token_id} -> {repr(next_text)}")
    
    # Test second step with cache
    print("\nTesting second generation step...")
    next_input = jnp.array([[next_token_id]], dtype=jnp.int32)
    outputs_2 = model.apply(params, input_ids=next_input, 
                           past_key_values=outputs["past_key_values"], 
                           return_dict=True)
    logits_2 = outputs_2["logits"][0, -1]
    next_token_2 = int(jnp.argmax(logits_2))
    next_text_2 = tok.decode([next_token_2])
    print(f"Second token: {next_token_2} -> {repr(next_text_2)}")
    
    # Show full sequence
    full_sequence = input_ids[0].tolist() + [next_token_id, next_token_2]
    full_text = tok.decode(full_sequence)
    print(f"\nFull sequence: {repr(full_text)}")

if __name__ == "__main__":
    test_jax_forward() 