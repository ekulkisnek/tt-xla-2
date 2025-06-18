#!/usr/bin/env python3
"""
Debug script to understand generation issues
"""
import json, time
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params

def debug_generation():
    # Load model
    print("Loading model...")
    cfg = json.load(open('../weights/config.json'))
    model = Qwen25ForCausalLM(config=cfg, dtype=jnp.bfloat16)
    params = load_params(model, '../weights', jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('../weights')
    
    # Simple prompt
    prompt = "Q: What is 2+3?\n\nA: Let's think step by step.\n"
    print(f"Prompt: {repr(prompt)}")
    
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids[0]}")
    print(f"Decoded input: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    # Generation loop with debugging
    current_ids = input_ids
    past_key_values = None
    
    for step in range(5):
        print(f"\n=== STEP {step} ===")
        print(f"Current input: {current_ids[0]} -> '{tokenizer.decode(current_ids[0])}'")
        
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=current_ids,
            past_key_values=past_key_values,
            return_dict=True
        )
        
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        print(f"Logits shape: {logits.shape}")
        print(f"Past KV shape: {[kv.shape for kv in past_key_values[0][:2]] if past_key_values else 'None'}")
        
        # Get next token predictions
        next_logits = logits[0, -1, :]
        top_k = 10
        top_indices = jnp.argsort(next_logits)[-top_k:][::-1]
        top_probs = next_logits[top_indices]
        
        print("Top predictions:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            token_str = repr(tokenizer.decode([int(idx)]))
            print(f"  {i+1:2d}. {int(idx):5d} {token_str:15s} (logit: {float(prob):8.2f})")
        
        # Check for patterns
        next_token_id = int(top_indices[0])
        print(f"Selected: {next_token_id} -> {repr(tokenizer.decode([next_token_id]))}")
        
        # Update for next step
        current_ids = jnp.array([[next_token_id]], dtype=jnp.int32)
        
        # Check if we're in a loop
        if step > 0 and next_token_id == prev_token:
            print("⚠️  REPETITION DETECTED!")
        prev_token = next_token_id

if __name__ == "__main__":
    debug_generation() 