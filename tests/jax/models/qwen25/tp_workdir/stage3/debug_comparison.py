#!/usr/bin/env python3
"""
Debug comparison between single device and tensor parallel modes
"""

import os
import sys
import json
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

# Add the current directory to path
sys.path.append('.')
from working_q25_with_tp import Qwen25ForCausalLM, load_params, create_mesh

def test_comparison():
    model_path = "../../weights"
    prompt = "Hello"
    
    # Load config and tokenizer
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dtype = jnp.bfloat16
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    print(f"Input IDs: {input_ids}")
    
    # Test 1: Single device
    print("\n=== SINGLE DEVICE ===")
    model_single = Qwen25ForCausalLM(config=config, dtype=dtype, use_tp=False)
    params_single = load_params(model_single, model_path, dtype)
    
    outputs_single = model_single.apply(
        params_single,
        input_ids=input_ids,
        return_dict=True
    )
    logits_single = outputs_single["logits"]
    next_token_single = jnp.argmax(logits_single[:, -1, :], axis=-1)
    print(f"Single device next token: {next_token_single}")
    print(f"Single device logits shape: {logits_single.shape}")
    print(f"Single device logits[:, -1, :5]: {logits_single[:, -1, :5]}")
    
    # Test 2: Tensor parallel
    print("\n=== TENSOR PARALLEL ===")
    mesh = create_mesh(model_parallel=2, data_parallel=1)
    with mesh:
        model_tp = Qwen25ForCausalLM(config=config, dtype=dtype, use_tp=True)
        params_tp = load_params(model_tp, model_path, dtype)
        
        outputs_tp = model_tp.apply(
            params_tp,
            input_ids=input_ids,
            return_dict=True
        )
        logits_tp = outputs_tp["logits"]
        next_token_tp = jnp.argmax(logits_tp[:, -1, :], axis=-1)
        print(f"TP next token: {next_token_tp}")
        print(f"TP logits shape: {logits_tp.shape}")
        print(f"TP logits[:, -1, :5]: {logits_tp[:, -1, :5]}")
    
    # Compare
    print(f"\n=== COMPARISON ===")
    print(f"Tokens match: {next_token_single[0] == next_token_tp[0]}")
    print(f"Logits diff (max): {jnp.max(jnp.abs(logits_single - logits_tp))}")
    print(f"Logits diff (mean): {jnp.mean(jnp.abs(logits_single - logits_tp))}")

if __name__ == "__main__":
    test_comparison() 