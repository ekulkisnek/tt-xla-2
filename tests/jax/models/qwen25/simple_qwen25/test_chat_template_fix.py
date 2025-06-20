#!/usr/bin/env python3
"""
Test if using proper chat template fixes the JAX generation issue
"""
import sys
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params, sample_next_token
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import json
import os
import time

def test_simple_generation():
    print("🔍 Testing Chat Template Fix")
    print("="*50)
    
    model_path = "../instruct_weights"
    
    print("Loading minimal components...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Test with very simple prompt first
    question = "What is 2+2?"
    
    # Format 1: Plain text (should fail)
    plain_prompt = f"Q: {question}\nA:"
    print(f"\n🔧 PLAIN FORMAT: {plain_prompt}")
    
    # Format 2: Chat template (should work)
    messages = [{"role": "user", "content": question}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n🤖 CHAT FORMAT: {chat_prompt[:100]}...")
    
    # Test both formats with 1-2 tokens only
    for format_name, prompt in [("PLAIN", plain_prompt), ("CHAT", chat_prompt)]:
        print(f"\n{'='*30}")
        print(f"Testing {format_name} format:")
        print(f"{'='*30}")
        
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="np")
            input_ids = inputs["input_ids"]
            print(f"Input tokens: {input_ids.shape[1]}")
            
            # Single forward pass - just get next token
            outputs = model.apply(
                params,
                input_ids=input_ids,
                return_dict=True
            )
            
            logits = outputs["logits"]
            print(f"Logits shape: {logits.shape}")
            
            # Get top 5 predictions
            last_logits = logits[0, -1, :]  # Last position
            top_5_indices = jnp.argsort(last_logits)[-5:][::-1]
            
            print("Top 5 next token predictions:")
            for i, token_id in enumerate(top_5_indices):
                token_text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
                prob = float(jax.nn.softmax(last_logits)[token_id])
                print(f"  {i+1}. Token {int(token_id)}: '{token_text}' (prob: {prob:.4f})")
            
            # Generate just 1 token to see immediate result
            next_token_id = int(jnp.argmax(last_logits))
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(f"\nGenerated next token: '{next_token}'")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*50}")
    print("🎯 If CHAT format produces better predictions than PLAIN,")
    print("   then the root cause is confirmed to be chat template!")

if __name__ == "__main__":
    test_simple_generation() 