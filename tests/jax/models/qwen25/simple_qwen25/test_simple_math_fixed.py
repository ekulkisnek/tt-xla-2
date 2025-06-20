#!/usr/bin/env python3
"""
Test simple math with improved generation logic and chat templates
"""
import sys
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import json
import os
import time

def simple_generate_fixed(model, params, tokenizer, prompt, max_tokens=50):
    """Simplified generation with better state handling."""
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        print(f"Input shape: {input_ids.shape}")
        
        generated_tokens = []
        current_ids = input_ids
        past_key_values = None
        
        for i in range(max_tokens):
            # Forward pass
            outputs = model.apply(
                params,
                input_ids=current_ids,
                past_key_values=past_key_values,
                return_dict=True
            )
            
            logits = outputs["logits"]
            past_key_values = outputs.get("past_key_values")
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token_id = int(jnp.argmax(next_token_logits))
            
            # Stop on EOS
            if next_token_id == tokenizer.eos_token_id:
                print(f"Stopped at EOS token after {i+1} tokens")
                break
            
            generated_tokens.append(next_token_id)
            
            # Update for next iteration
            current_ids = jnp.array([[next_token_id]], dtype=jnp.int32)
            
            # Decode for debugging
            try:
                token_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                print(f"Token {i+1}: {next_token_id} = '{token_text}'")
            except:
                print(f"Token {i+1}: {next_token_id} = [decode error]")
            
            # Simple repetition check
            if len(generated_tokens) >= 3 and len(set(generated_tokens[-3:])) == 1:
                print(f"Stopped due to repetition after {i+1} tokens")
                break
        
        # Decode all generated tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def test_simple_math():
    print("🔍 Testing Simple Math with Chat Templates")
    print("="*60)
    
    model_path = "../instruct_weights"
    
    print("Loading model components...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Test simple math problems
    questions = [
        "What is 2+2?",
        "What is 5-3?",
    ]
    
    for i, question in enumerate(questions):
        print(f"\n{'='*40}")
        print(f"🧮 Problem {i+1}: {question}")
        print(f"{'='*40}")
        
        # Use proper chat template
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print(f"Chat template prompt length: {len(prompt)} chars")
        print(f"Prompt preview: {prompt[:100]}...")
        
        # Generate
        start_time = time.time()
        response = simple_generate_fixed(model, params, tokenizer, prompt, max_tokens=20)
        generation_time = time.time() - start_time
        
        print(f"\n🤖 Generated Response: '{response}'")
        print(f"⏱️  Generation time: {generation_time:.1f}s")
        
        # Check if response contains reasonable answer
        contains_number = any(char.isdigit() for char in response)
        print(f"Contains number: {'✅' if contains_number else '❌'}")

if __name__ == "__main__":
    test_simple_math() 