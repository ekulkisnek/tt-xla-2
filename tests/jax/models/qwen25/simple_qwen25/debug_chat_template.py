#!/usr/bin/env python3
"""
Debug chat template formatting differences between PyTorch and JAX
"""
import sys
sys.path.append('.')
from transformers import AutoTokenizer

def debug_prompt_formats():
    print("🔍 Debugging Prompt Format Differences")
    print("="*60)
    
    model_path = "../instruct_weights"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test the exact same problem
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    print(f"Original question: {question}")
    print()
    
    # Format 1: PyTorch chat template format (WORKING)
    messages = [{"role": "user", "content": f"Solve this math problem step by step:\n\n{question}\n\nShow your work and clearly state the final answer."}]
    pytorch_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("🤖 PYTORCH FORMAT (Working):")
    print("="*40)
    print(pytorch_prompt)
    print("="*40)
    print(f"Length: {len(pytorch_prompt)} chars")
    
    # Tokenize PyTorch format
    pytorch_tokens = tokenizer(pytorch_prompt, return_tensors="np")
    print(f"Token count: {pytorch_tokens['input_ids'].shape[1]}")
    print(f"First 10 tokens: {pytorch_tokens['input_ids'][0][:10].tolist()}")
    print()
    
    # Format 2: JAX plain format (BROKEN)
    jax_prompt = f"Solve this math problem step by step:\n\n{question}\n\nShow your work and clearly state the final answer."
    
    print("🔧 JAX FORMAT (Broken):")
    print("="*40)
    print(jax_prompt)
    print("="*40)
    print(f"Length: {len(jax_prompt)} chars")
    
    # Tokenize JAX format
    jax_tokens = tokenizer(jax_prompt, return_tensors="np")
    print(f"Token count: {jax_tokens['input_ids'].shape[1]}")
    print(f"First 10 tokens: {jax_tokens['input_ids'][0][:10].tolist()}")
    print()
    
    # Format 3: Test if JAX should use chat template too
    print("🧪 TESTING: JAX with Chat Template:")
    print("="*40)
    
    # Let's see what the difference is
    print("Key differences:")
    print(f"1. PyTorch length: {len(pytorch_prompt)} vs JAX length: {len(jax_prompt)}")
    print(f"2. PyTorch tokens: {pytorch_tokens['input_ids'].shape[1]} vs JAX tokens: {jax_tokens['input_ids'].shape[1]}")
    
    # Check if the token sequences are completely different
    pytorch_first_10 = pytorch_tokens['input_ids'][0][:10].tolist()
    jax_first_10 = jax_tokens['input_ids'][0][:10].tolist()
    
    print(f"3. First 10 tokens match: {'✅' if pytorch_first_10 == jax_first_10 else '❌'}")
    print(f"   PyTorch: {pytorch_first_10}")
    print(f"   JAX:     {jax_first_10}")
    
    # Decode the first few tokens to see the difference
    print("\n🔍 Token-by-token analysis:")
    for i in range(min(10, len(pytorch_first_10), len(jax_first_10))):
        if i < len(pytorch_first_10):
            pt_token = tokenizer.decode([pytorch_first_10[i]])
            print(f"  PyTorch token {i}: {pytorch_first_10[i]} = '{pt_token}'")
        if i < len(jax_first_10):
            jax_token = tokenizer.decode([jax_first_10[i]])
            print(f"  JAX token {i}:     {jax_first_10[i]} = '{jax_token}'")
        print()
    
    print("\n💡 HYPOTHESIS:")
    print("The issue might be that JAX is not using the proper chat template formatting")
    print("that the instruct model expects, while PyTorch is using it correctly.")

if __name__ == "__main__":
    debug_prompt_formats() 