#!/usr/bin/env python3
"""
Working GSM8K test using Qwen2.5-7B-Instruct with the same prompt format as base model
"""
import os
import sys
import time
import json
import gc
import re

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

# Import model classes
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params

def extract_number(text):
    """Extract the final numerical answer from generated text."""
    # Look for #### pattern first (GSM8K standard)
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1)
    
    # Look for "answer is" pattern
    match = re.search(r"(?:answer is|Answer:)\s*(-?\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # As fallback, get the last number that appears
    numbers = re.findall(r'\b\d+\b', text)
    return numbers[-1] if numbers else None

def greedy_generate(model, params, tokenizer, prompt, max_tokens=256):
    """Generate text using greedy decoding."""
    
    # Tokenize input
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
        
        # Stop conditions
        if next_token[0] == tokenizer.eos_token_id:
            break
        if "####" in generated_text and len(generated_text) > 100:
            break
    
    return generated_text

def test_gsm8k_problem():
    """Test the Janet's ducks problem from GSM8K."""
    
    # Load model
    model_path = "../instruct_weights"
    print("🔹 Loading Qwen2.5-7B-Instruct model...")
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Test problem (classic GSM8K)
    problem = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?"
    expected_answer = "18"
    
    # Use the exact same prompt format as gsm8k_simple.py
    prompt = f"Q: {problem}\n\nA: Let's think step by step.\n"
    
    print(f"📝 Problem: {problem}")
    print(f"🎯 Expected answer: {expected_answer}")
    print(f"\n📄 Using prompt:\n{prompt}")
    print("="*80)
    
    # Generate solution
    print("🔹 Generating solution...")
    start_time = time.time()
    generated_text = greedy_generate(model, params, tokenizer, prompt, max_tokens=300)
    gen_time = time.time() - start_time
    
    print(f"🤖 Generated response:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)
    
    # Extract answer
    predicted_answer = extract_number(generated_text)
    is_correct = predicted_answer == expected_answer
    
    # Results
    print(f"\n📊 Results:")
    print(f"⏱️  Generation time: {gen_time:.1f}s")
    print(f"✅ Expected: {expected_answer}")
    print(f"🔍 Predicted: {predicted_answer}")
    print(f"{'🎉 CORRECT!' if is_correct else '❌ INCORRECT'}")
    
    # Show reasoning if we can find it
    if "step" in generated_text.lower():
        reasoning_start = generated_text.lower().find("step")
        if reasoning_start > 0:
            reasoning = generated_text[reasoning_start:reasoning_start+200]
            print(f"💭 Reasoning snippet: ...{reasoning}...")
    
    # Clean up
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return is_correct

def test_simple_math():
    """Test with a very simple math problem first."""
    
    # Load model
    model_path = "../instruct_weights"
    print("🔹 Loading model for simple test...")
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Simple problem
    problem = "Sarah has 5 apples. She gives 2 apples to her friend. How many apples does Sarah have left?"
    expected = "3"
    
    prompt = f"Q: {problem}\n\nA: Let's think step by step.\n"
    
    print(f"📝 Simple problem: {problem}")
    print(f"🎯 Expected: {expected}")
    
    # Generate
    start_time = time.time()
    response = greedy_generate(model, params, tokenizer, prompt, max_tokens=100)
    gen_time = time.time() - start_time
    
    print(f"🤖 Response: {response}")
    
    predicted = extract_number(response)
    is_correct = predicted == expected
    
    print(f"⏱️  Time: {gen_time:.1f}s")
    print(f"🔍 Predicted: {predicted}")
    print(f"{'✅ CORRECT!' if is_correct else '❌ INCORRECT'}")
    
    # Clean up
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return is_correct

if __name__ == "__main__":
    print("🧮 Testing Qwen2.5-7B-Instruct on GSM8K")
    print("="*80)
    
    # Test simple math first
    print("\n🔹 PHASE 1: Simple Math Test")
    simple_success = test_simple_math()
    
    if simple_success:
        print("\n✅ Simple test passed! Moving to GSM8K...")
        print("\n🔹 PHASE 2: GSM8K Test")
        gsm8k_success = test_gsm8k_problem()
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Simple Math: {'✅' if simple_success else '❌'}")
        print(f"   GSM8K Problem: {'✅' if gsm8k_success else '❌'}")
    else:
        print("\n❌ Simple test failed. Skipping GSM8K test.")
    
    print("\n🔹 Test completed!") 