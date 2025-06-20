#!/usr/bin/env python3
"""
Test fp32 vs bfloat16 precision for math problems
"""
import sys
sys.path.append('.')
from gsm8k_instruct_final import simple_greedy_generate, extract_answer
from q25_jax_instruct import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer
import jax.numpy as jnp
import json
import os

def test_precision():
    """Test different precision levels"""
    model_path = "../instruct_weights"
    
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test problem
    problem = "8 - 3 = "
    expected = "5"
    
    dtypes = [
        ("bfloat16", jnp.bfloat16),
        ("float32", jnp.float32)
    ]
    
    print(f"🧮 Testing precision for: {problem}")
    print(f"Expected: {expected}")
    print("="*60)
    
    for dtype_name, dtype in dtypes:
        print(f"\n🔢 Testing with {dtype_name}")
        print("-" * 30)
        
        # Load model with this precision
        print(f"Loading model in {dtype_name}...")
        model = Qwen25ForCausalLM(config=config, dtype=dtype)
        params = load_params(model, model_path, dtype)
        
        # Test generation
        response = simple_greedy_generate(model, params, tokenizer, problem, max_tokens=20)
        predicted = extract_answer(response)
        
        print(f"Generated: {response}")
        print(f"Extracted: {predicted}")
        print(f"Correct: {'✅' if predicted == expected else '❌'}")
        
        # Cleanup
        del model, params

if __name__ == "__main__":
    test_precision() 