#!/usr/bin/env python3
"""
Math test to verify Qwen can solve 2+2=4 properly
Based on the working gsm8k_simple.py pattern
"""
import os
import json
import time
import gc
import re

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open
from transformers import AutoTokenizer

# Import working implementation components
from working_qwen import Qwen25ForCausalLM, load_params

def extract_answer(text):
    """Extract answer from text using proven patterns"""
    # Look for mathematical answers
    patterns = [
        r"####\s*(-?\d+)",
        r"[Tt]he answer is\s*(-?\d+)",
        r"[Aa]nswer:\s*(-?\d+)",
        r"=\s*(-?\d+)(?:\s|$)",
        r"(?:^|\n)(-?\d+)(?:\s|$)",  # Number on its own line
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    
    # Last resort: look for standalone numbers
    numbers = re.findall(r'\b\d+\b', text)
    return numbers[-1] if numbers else None

def simple_greedy_generate(model, params, tokenizer, prompt, max_tokens=100):
    """Generate using greedy decoding (temperature=0 equivalent)"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create proper 4D attention mask
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
        
        # GREEDY selection (argmax) - this is key!
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        
        # Update state for next iteration
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode token
        token = tokenizer.decode(next_token[0])
        generated_text += token
        print(token, end="", flush=True)
        
        # Stop conditions
        if next_token[0] == tokenizer.eos_token_id:
            break
        if "=" in generated_text and len(generated_text.split("=")) > 1:
            # Continue a bit more after = to get the answer
            if i > 20:  # Safety check
                break
    
    print()  # New line
    return generated_text

def test_math_problems():
    """Test basic math problems"""
    
    model_path = "qwen25_7b_instruct_weights"
    
    print("Loading Qwen2.5-7B model...")
    
    # Load config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model and tokenizer
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load parameters
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Test cases
    test_cases = [
        {"problem": "What is 2+2?", "expected": "4"},
        {"problem": "Calculate 3+5", "expected": "8"}, 
        {"problem": "2+2=", "expected": "4"},
        {"problem": "1+1", "expected": "2"},
    ]
    
    print("\n" + "="*60)
    print("MATH PROBLEM TESTS")
    print("="*60)
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        problem = test_case["problem"]
        expected = test_case["expected"]
        
        print(f"\n📝 Test {i+1}: {problem}")
        print(f"Expected: {expected}")
        print(f"Generated: ", end="")
        
        # Generate answer
        start_time = time.time()
        response = simple_greedy_generate(model, params, tokenizer, problem, max_tokens=50)
        gen_time = time.time() - start_time
        
        # Extract predicted answer
        predicted = extract_answer(response)
        is_correct = predicted == expected
        
        print(f"Extracted answer: {predicted}")
        print(f"Correct: {'✅ YES' if is_correct else '❌ NO'} ({gen_time:.1f}s)")
        
        results.append({
            "problem": problem,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "response": response,
            "time": gen_time
        })
    
    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Correct: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    
    if correct_count == total_count:
        print("🎉 SUCCESS: All math problems solved correctly!")
    else:
        print("❌ FAILURE: Some math problems incorrect")
        for r in results:
            if not r["correct"]:
                print(f"  Failed: {r['problem']} -> expected {r['expected']}, got {r['predicted']}")
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return correct_count == total_count

if __name__ == "__main__":
    success = test_math_problems()
    exit(0 if success else 1) 