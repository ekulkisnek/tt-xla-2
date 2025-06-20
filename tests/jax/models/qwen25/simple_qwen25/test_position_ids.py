#!/usr/bin/env python3
"""
Debug position IDs during generation - might be causing number repetition
"""
import sys
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import json
import os

def debug_generation_step_by_step():
    """Debug each step of generation to see what's happening"""
    model_path = "../instruct_weights"
    
    print("🔹 Loading model...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Simple test
    prompt = "8 - 3 = "
    print(f"\n🧮 Testing: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids}")
    print(f"Input text: '{tokenizer.decode(input_ids[0])}'")
    
    batch_size, seq_length = input_ids.shape
    print(f"Initial seq_length: {seq_length}")
    
    # Test single forward pass first
    attention_mask = jnp.ones((batch_size, 1, 1, seq_length), dtype=jnp.int32)
    position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :]
    print(f"Initial position_ids: {position_ids}")
    
    print("\n🔍 Step 0 (Initial forward pass):")
    outputs = model.apply(
        params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        return_dict=True
    )
    
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    
    print(f"Logits shape: {logits.shape}")
    last_logits = logits[0, -1, :]
    top_k_logits, top_k_indices = jax.lax.top_k(last_logits, k=5)
    
    print("Top 5 predictions:")
    for i, (logit, token_id) in enumerate(zip(top_k_logits, top_k_indices)):
        token = tokenizer.decode([int(token_id)])
        print(f"  {i+1}. Token {int(token_id)}: '{token}' (logit: {float(logit):.3f})")
    
    # Pick the top token and continue
    next_token_id = int(top_k_indices[0])
    token = tokenizer.decode([next_token_id])
    print(f"\nSelected token: {next_token_id} ('{token}')")
    
    # Now test the NEXT step (this is where bugs often occur)
    print("\n🔍 Step 1 (Generation step):")
    
    # Update state for next step - CHECK THIS CAREFULLY
    new_input_ids = jnp.array([[next_token_id]], dtype=jnp.int32)
    new_attention_mask = jnp.ones((batch_size, 1, 1, 1), dtype=jnp.int32)
    new_position_ids = jnp.array([[seq_length]], dtype=jnp.int32)  # This should be seq_length (next position)
    
    print(f"New input_ids: {new_input_ids}")
    print(f"New position_ids: {new_position_ids}")
    print(f"Past key length: {past_key_values[0][0].shape[1] if past_key_values else 'None'}")
    
    outputs2 = model.apply(
        params,
        input_ids=new_input_ids,
        attention_mask=new_attention_mask,
        position_ids=new_position_ids,
        past_key_values=past_key_values,
        return_dict=True
    )
    
    logits2 = outputs2["logits"]
    print(f"Second logits shape: {logits2.shape}")
    
    last_logits2 = logits2[0, -1, :]
    top_k_logits2, top_k_indices2 = jax.lax.top_k(last_logits2, k=5)
    
    print("Top 5 predictions for step 2:")
    for i, (logit, token_id) in enumerate(zip(top_k_logits2, top_k_indices2)):
        token = tokenizer.decode([int(token_id)])
        print(f"  {i+1}. Token {int(token_id)}: '{token}' (logit: {float(logit):.3f})")

if __name__ == "__main__":
    debug_generation_step_by_step() 