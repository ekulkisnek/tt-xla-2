#!/usr/bin/env python3
"""
Test JAX Qwen2.5-7B-Instruct on REAL GSM8K problems with full generation
Mirror of the PyTorch test but using JAX implementation
"""
import sys
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM, load_params, apply_chat_template
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import json
import os
import re
import time

def extract_answer(text):
    """Extract numerical answer from generated text"""
    # More comprehensive patterns for GSM8K
    patterns = [
        r"the answer is (\d+)",
        r"answer is (\d+)",
        r"answer: (\d+)", 
        r"####\s*(\d+)",
        r"therefore[,\s]*(?:the answer is\s*)?(\d+)",
        r"so[,\s]*(?:the answer is\s*)?(\d+)",
        r"result is (\d+)",
        r"equals? (\d+)",
        r"\$(\d+)",
        r"(\d+)\s*(?:dollars?|apples?|stickers?|candies?|books?|toys?|items?|people?|minutes?|hours?|days?)",
    ]
    
    # Try patterns in order of specificity
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[-1])  # Take the last match (usually the final answer)
    
    # Fallback: extract any number near the end
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return int(numbers[-1])
    
    return None

def jax_generate_with_chat_template(model, params, tokenizer, question, max_tokens=200):
    """Generate response using JAX model with chat template like PyTorch version"""
    try:
        # Use instruct chat template - same as PyTorch version
        messages = [{"role": "user", "content": f"Solve this math problem step by step:\n\n{question}\n\nShow your work and clearly state the final answer."}]
        
        # Apply chat template using the function from q25_jax_instruct
        formatted_prompt = apply_chat_template(tokenizer, messages)
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        batch_size, seq_length = input_ids.shape
        attention_mask = jnp.ones((batch_size, 1, 1, seq_length), dtype=jnp.int32)
        position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :]
        
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
            
            # Greedy sampling with numerical stability
            logits_stable = jnp.clip(logits[:, -1, :].astype(jnp.float32), -50.0, 50.0)
            next_token = jnp.argmax(logits_stable, axis=-1)
            token_id = int(next_token[0])
            
            # Stop on EOS
            if token_id == tokenizer.eos_token_id:
                break
            
            # Decode token
            try:
                token = tokenizer.decode([token_id], skip_special_tokens=True)
                generated_text += token
            except:
                # Skip problematic tokens
                pass
            
            # Update state for next iteration
            state["input_ids"] = jnp.array([[token_id]], dtype=jnp.int32)
            state["attention_mask"] = jnp.ones((batch_size, 1, 1, 1), dtype=jnp.int32)
            state["position_ids"] = jnp.array([[state["position_ids"][0, -1] + 1]], dtype=jnp.int32)
            state["past_key_values"] = past_key_values
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Generation error: {e}")
        return ""

def test_jax_real_gsm8k():
    print("🔥 Testing JAX Qwen2.5-7B-INSTRUCT on Real GSM8K")
    print("="*70)
    
    model_path = "../instruct_weights"
    
    print(f"📥 Loading JAX model from: {model_path}")
    
    # Load model
    try:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)  # Use float32 for better precision
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        params = load_params(model, model_path, jnp.float32)
        
        print("✅ JAX INSTRUCT model loaded")
        
    except Exception as e:
        print(f"❌ JAX loading failed: {e}")
        return
    
    # Same real GSM8K problems as PyTorch test - but only first 2
    problems = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": 72
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": 10
        }
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n{'='*70}")
        print(f"🧮 Problem {i+1}/2")
        print(f"{'='*70}")
        print(f"Question: {problem['question']}")
        print(f"Expected Answer: {problem['answer']}")
        
        # Generate with more tokens for complete reasoning
        start_time = time.time()
        response = jax_generate_with_chat_template(
            model, params, tokenizer, problem['question'], max_tokens=200
        )
        generation_time = time.time() - start_time
        
        print(f"\n🤖 Generated Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print(f"⏱️  Generation time: {generation_time:.1f}s")
        
        # Extract answer
        extracted = extract_answer(response)
        correct = extracted == problem['answer']
        
        print(f"\n📊 Analysis:")
        print(f"  Extracted Answer: {extracted}")
        print(f"  Expected Answer: {problem['answer']}")
        print(f"  Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        
        results.append({
            'problem': i+1,
            'question': problem['question'],
            'expected': problem['answer'],
            'generated': response,
            'extracted': extracted,
            'correct': correct,
            'time': generation_time
        })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"\n{'='*70}")
    print(f"🏆 JAX FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Score: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"Average generation time: {avg_time:.1f}s per problem")
    print(f"\nProblem-by-problem:")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"  {r['problem']}: {status} Expected {r['expected']}, Got {r['extracted']} ({r['time']:.1f}s)")
    
    # Save results for comparison with PyTorch
    with open('jax_real_gsm8k_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to jax_real_gsm8k_results.json")
    
    return results

if __name__ == "__main__":
    test_jax_real_gsm8k() 