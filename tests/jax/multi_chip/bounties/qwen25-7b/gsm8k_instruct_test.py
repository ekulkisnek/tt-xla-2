#!/usr/bin/env python3
"""
GSM8K test using Qwen2.5-7B-Instruct model with chat templates
"""
import os
import sys
import time
import json
import gc
import argparse
import logging
import re
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open

# Import the instruct model classes from our new file
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params, apply_chat_template

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gsm8k_instruct")

def extract_number_from_answer(text):
    """Extract the final numerical answer from generated text."""
    # Look for patterns like "The answer is X" or "#### X"
    patterns = [
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"[Tt]he answer is\s*(-?\d+(?:\.\d+)?)",
        r"[Aa]nswer:\s*(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)(?:\s|$)",
        r"(?:^|\s)(-?\d+(?:\.\d+)?)(?:\s|$)",  # Last number in the text
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                # Convert to int if it's a whole number, otherwise float
                num_str = matches[-1]  # Take the last match
                if '.' in num_str:
                    return str(int(float(num_str)))  # Convert to int for comparison
                else:
                    return num_str
            except:
                continue
    
    return None

def greedy_generate(model, params, tokenizer, prompt, max_tokens=300):
    """Generate text using greedy decoding (temperature=0)."""
    
    # Apply chat template for math problem
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = apply_chat_template(tokenizer, messages)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create attention mask - use 4D format for Qwen model
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Position IDs
    position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
    
    # Initialize generation state
    state = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    
    # Track generated text
    generated_text = ""
    
    logger.info(f"Starting generation with {input_ids.shape[1]} input tokens...")
    
    # Generate tokens
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
        
        # Get logits and past key values
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        # Greedy selection (argmax)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        
        # Update state
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode and collect token
        token = tokenizer.decode(next_token[0])
        generated_text += token
        
        # Check for end of sequence
        if next_token[0] == tokenizer.eos_token_id:
            logger.info(f"Generation stopped at EOS token after {i+1} tokens")
            break
            
        # Early stopping if we find a clear answer
        if "the answer is" in generated_text.lower() or "####" in generated_text:
            # Continue for a few more tokens to get the complete answer
            if i > len("the answer is") + 10:  # Give some buffer
                break
    
    return generated_text

def test_gsm8k_problem(model, params, tokenizer, problem, expected_answer):
    """Test a single GSM8K problem."""
    
    # Format the problem with step-by-step instruction
    prompt = f"Solve this math problem step by step:\n\n{problem}\n\nShow your work and provide the final numerical answer."
    
    logger.info(f"Problem: {problem[:100]}...")
    logger.info(f"Expected answer: {expected_answer}")
    
    # Generate solution
    start_time = time.time()
    generated_text = greedy_generate(model, params, tokenizer, prompt, max_tokens=400)
    gen_time = time.time() - start_time
    
    logger.info(f"Generated in {gen_time:.1f}s")
    print(f"\n🤖 Generated Solution:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    
    # Extract answer
    predicted_answer = extract_number_from_answer(generated_text)
    
    # Check if correct
    is_correct = predicted_answer == expected_answer
    
    print(f"\n📊 Results:")
    print(f"Expected: {expected_answer}")
    print(f"Predicted: {predicted_answer}")
    print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")
    
    return is_correct, generated_text, predicted_answer

def main():
    parser = argparse.ArgumentParser(description="GSM8K test with Qwen2.5-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="../instruct_weights", help="Path to instruct model weights")
    parser.add_argument("--problem", type=str, default=None, help="Custom problem to solve")
    parser.add_argument("--answer", type=str, default=None, help="Expected answer for custom problem")
    args = parser.parse_args()
    
    # Default GSM8K test problems if no custom problem provided
    test_problems = [
        {
            "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?",
            "answer": "18"
        },
        {
            "problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take?",
            "answer": "3"
        },
        {
            "problem": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "70000"
        }
    ]
    
    # Use custom problem if provided
    if args.problem and args.answer:
        test_problems = [{"problem": args.problem, "answer": args.answer}]
    
    # Load model
    logger.info("Loading Qwen2.5-7B-Instruct model...")
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load weights
    params = load_params(model, args.model_path, jnp.bfloat16)
    gc.collect(); jax.clear_caches()
    
    # Test problems
    correct_count = 0
    total_count = len(test_problems)
    
    for i, test_case in enumerate(test_problems):
        print(f"\n{'='*80}")
        print(f"🧮 GSM8K PROBLEM {i+1}/{total_count}")
        print(f"{'='*80}")
        
        is_correct, solution, predicted = test_gsm8k_problem(
            model, params, tokenizer, 
            test_case["problem"], 
            test_case["answer"]
        )
        
        if is_correct:
            correct_count += 1
    
    # Final results
    print(f"\n{'='*80}")
    print(f"🏆 FINAL RESULTS: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    print(f"{'='*80}")
    
    # Clean up
    del params; del model; gc.collect(); jax.clear_caches()

if __name__ == "__main__":
    main() 