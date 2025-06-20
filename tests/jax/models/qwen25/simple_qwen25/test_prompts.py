#!/usr/bin/env python3
"""
Test different prompt formats for better math reasoning
"""
import sys
sys.path.append('.')
from gsm8k_instruct_final import simple_greedy_generate, extract_answer
from q25_jax_instruct import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer
import jax.numpy as jnp
import json
import os

def test_prompt_formats():
    """Test different ways to format the same math problem"""
    model_path = "../instruct_weights"
    
    print("🔹 Loading model...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Same problem, different formats
    base_problem = "Tom has 8 apples. He eats 3. How many are left?"
    expected = "5"
    
    prompt_formats = [
        {
            "name": "Direct Q&A",
            "prompt": f"Q: {base_problem}\nA:"
        },
        {
            "name": "Step-by-step",
            "prompt": f"Q: {base_problem}\nA: Let me solve this step by step.\n"
        },
        {
            "name": "Math format", 
            "prompt": f"Problem: {base_problem}\nSolution: 8 - 3 = "
        },
        {
            "name": "Arithmetic only",
            "prompt": "8 - 3 = "
        },
        {
            "name": "GSM8K style",
            "prompt": f"Solve step-by-step then answer with ####:\n\n{base_problem}\n\n"
        },
        {
            "name": "Chat format (manual)",
            "prompt": f"<|im_start|>user\nSolve step-by-step: {base_problem}\n<|im_end|>\n<|im_start|>assistant\n"
        }
    ]
    
    print(f"\n🧮 Testing problem: {base_problem}")
    print(f"Expected answer: {expected}")
    print("="*80)
    
    results = []
    
    for fmt in prompt_formats:
        print(f"\n📝 Format: {fmt['name']}")
        print(f"Prompt: {repr(fmt['prompt'])}")
        print("-" * 40)
        
        response = simple_greedy_generate(model, params, tokenizer, fmt['prompt'], max_tokens=50)
        predicted = extract_answer(response)
        is_correct = predicted == expected
        results.append((fmt['name'], is_correct))
        
        print(f"Generated: {response}")
        print(f"Extracted: {predicted}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY:")
    for name, correct in results:
        print(f"  {name}: {'✅' if correct else '❌'}")
    
    correct_count = sum(1 for _, correct in results if correct)
    print(f"\nSuccess rate: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")

if __name__ == "__main__":
    test_prompt_formats() 