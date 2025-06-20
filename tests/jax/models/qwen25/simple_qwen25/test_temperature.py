#!/usr/bin/env python3
"""
Test different temperature values to fix number repetition
"""
import sys
sys.path.append('.')
from gsm8k_instruct_final import simple_greedy_generate, extract_answer
from q25_jax_instruct import Qwen25ForCausalLM, load_params
from transformers import AutoTokenizer
import jax.numpy as jnp
import json
import os

def test_with_temperature(model, params, tokenizer, prompt, temperature=0.0):
    """Generate with a specific temperature"""
    # Modify the generation function to use temperature
    try:
        inputs = tokenizer(prompt, return_tensors="np")
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
        recent_tokens = []
        
        for i in range(30):  # Shorter for testing
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
            
            # Apply temperature and sample
            logits_stable = jnp.clip(logits[:, -1, :].astype(jnp.float32), -50.0, 50.0)
            
            if temperature < 1e-5:
                next_token = jnp.argmax(logits_stable, axis=-1)
            else:
                # Apply temperature
                logits_temp = logits_stable / temperature
                # Simple categorical sampling
                import jax
                rng_key = jax.random.PRNGKey(42)
                next_token = jax.random.categorical(rng_key, logits_temp, axis=-1)
            
            token_id = int(next_token[0])
            
            if token_id == tokenizer.eos_token_id:
                break
                
            # Track repetition
            recent_tokens.append(token_id)
            if len(recent_tokens) > 5:
                recent_tokens.pop(0)
                
            if len(recent_tokens) >= 3 and len(set(recent_tokens[-3:])) == 1:
                print(f"Stopping due to repetition at temp {temperature}")
                break
            
            # Decode token
            token = tokenizer.decode([token_id], skip_special_tokens=True)
            generated_text += token
            
            # Update state
            state["input_ids"] = jnp.array([[token_id]], dtype=jnp.int32)
            state["attention_mask"] = jnp.ones((batch_size, 1, 1, 1), dtype=jnp.int32)
            state["position_ids"] = jnp.array([[state["position_ids"][0, -1] + 1]], dtype=jnp.int32)
            state["past_key_values"] = past_key_values
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Generation error: {e}")
        return ""

def test_temperature_strategy():
    """Test different temperatures"""
    model_path = "../instruct_weights"
    
    print("🔹 Loading model...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    # Test problem that's currently failing
    problem = "Q: Tom has 8 apples. He eats 3. How many are left?\nA:"
    expected = "5"
    
    temperatures = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    print(f"\n🧮 Testing problem: {problem}")
    print(f"Expected answer: {expected}")
    print("="*60)
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp}")
        response = test_with_temperature(model, params, tokenizer, problem, temp)
        predicted = extract_answer(response)
        
        print(f"Generated: {response}")
        print(f"Extracted: {predicted}")
        print(f"Correct: {'✅' if predicted == expected else '❌'}")

if __name__ == "__main__":
    test_temperature_strategy() 