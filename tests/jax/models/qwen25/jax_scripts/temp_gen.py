#!/usr/bin/env python3
"""
Simple JAX generation for Gate 4 testing.
"""
import sys
import os
sys.path.append('.')
sys.path.append('../simple_qwen25')
import simple_inference as si
from transformers import AutoTokenizer
import json
import jax
import jax.numpy as jnp
import numpy as np

def main():
    # Parameters
    model_path = "../weights"
    prompt = "The meaning of life"
    max_tokens = 64
    dtype = jnp.bfloat16
    
    # Set JAX environment
    jax.config.update("jax_enable_x64", False)
    
    # Load config and tokenizer
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create model and load params
    model = si.Qwen25ForCausalLM(config=config, dtype=dtype)
    params = si.load_params(model, model_path, dtype)
    
    print("Starting JAX greedy generation...")
    
    # Tokenize initial prompt
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    
    # Simple generation loop
    generated_text = ""
    current_input = input_ids
    
    for i in range(max_tokens):
        # Forward pass
        outputs = model.apply(params, input_ids=current_input)
        
        # Handle dict vs direct logits output
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        
        # Get last token logits - JAX array slicing
        batch_size, seq_len, vocab_size = logits.shape
        last_logits = logits[0, seq_len-1, :]  # Avoid negative indexing
        
        # Greedy sampling
        next_token_id = int(jnp.argmax(last_logits))
        
        # Decode and add to text
        token_text = tokenizer.decode([next_token_id])
        generated_text += token_text
        print(token_text, end="", flush=True)
        
        # Stop if EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Append to input for next iteration (simple approach)
        new_token = jnp.array([[next_token_id]])
        current_input = jnp.concatenate([current_input, new_token], axis=1)
    
    print(f"\nGeneration completed!")
    
    # Save full text
    full_text = prompt + generated_text
    with open("../tmp/jax_g4.txt", "w") as f:
        f.write(full_text)
    
    print(f"Generated text length: {len(full_text)} chars")
    print(f"Full text: {full_text}")

if __name__ == "__main__":
    main() 