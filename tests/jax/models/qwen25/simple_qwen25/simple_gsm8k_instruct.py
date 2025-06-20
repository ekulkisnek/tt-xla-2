#!/usr/bin/env python3
"""
Simple GSM8K test using Qwen2.5-7B-Instruct with minimal prompting
"""
import os
import sys
import time
import json
import gc
import argparse
import logging
import re

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

# Import model classes from the instruct file
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_gsm8k")

def extract_number(text):
    """Extract the final number from the generated text."""
    # Look for various answer patterns
    patterns = [
        r"####\s*(-?\d+)",
        r"[Tt]he answer is\s*(-?\d+)",
        r"[Aa]nswer:\s*(-?\d+)",
        r"=\s*(-?\d+)(?:\s|$)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    
    # As fallback, look for the last number in the text
    numbers = re.findall(r'\b\d+\b', text)
    return numbers[-1] if numbers else None

def simple_greedy_generate(model, params, tokenizer, prompt, max_tokens=200):
    """Generate text using greedy decoding without chat template."""
    
    # Tokenize input directly (no chat template)
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create attention mask
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Position IDs
    position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
    
    # Initialize state
    state = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    
    generated_text = ""
    
    for i in range(max_tokens):
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=state["input_ids"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            past_key_values=state["past_key_values"],
            return_dict=True
        )
        
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        # Greedy selection
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        
        # Update state
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode token
        token = tokenizer.decode(next_token[0])
        generated_text += token
        
        # Stop on EOS or when we find an answer
        if next_token[0] == tokenizer.eos_token_id:
            break
        if "####" in generated_text and len(generated_text) > 50:
            break
    
    return generated_text

def test_math_problem():
    """Test a simple math problem with the instruct model."""
    
    # Load model
    model_path = "../instruct_weights"
    logger.info("Loading Qwen2.5-7B-Instruct model...")
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Simple test problem
    problem = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?"
    expected = "18"
    
    # Simple prompt without chat template
    prompt = f"Q: {problem}\n\nA: Let's solve this step by step.\n"
    
    print(f"Problem: {problem}")
    print(f"Expected answer: {expected}")
    print(f"\nPrompt being used:\n{prompt}")
    print("="*60)
    
    # Generate
    start_time = time.time()
    response = simple_greedy_generate(model, params, tokenizer, prompt, max_tokens=150)
    gen_time = time.time() - start_time
    
    print(f"Generated response:\n{response}")
    print("="*60)
    
    # Extract answer
    predicted = extract_number(response)
    is_correct = predicted == expected
    
    print(f"\nGeneration time: {gen_time:.1f}s")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")
    
    # Clean up
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return is_correct

if __name__ == "__main__":
    test_math_problem() 