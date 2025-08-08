#!/usr/bin/env python3
"""
Multi-device generation script for Qwen2.5-7B TP in JAX.
"""

import argparse
import psutil
import gc
import time
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import os
import json
from model import Qwen25ForCausalLM, load_params, sample_next_token, mesh, setup_device_mesh

def generate_text(model, params, tokenizer, max_tokens, prompt):
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    peak_memory = initial_memory
    
    # Tokenize input with chat template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(formatted_text, return_tensors="jax")
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    past_key_values = None
    
    generated_tokens = []
    start_time = time.time()
    
    num_tokens_generated = 0
    
    for i in range(max_tokens):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids
        )
        
        outputs = model.apply(
            params, 
            **model_inputs, 
            return_dict=True
        )
        
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        
        update_inputs = model.update_inputs_for_generation(outputs, position_ids=model_inputs["position_ids"])
        past_key_values = update_inputs["past_key_values"]
        position_ids = update_inputs["position_ids"]
        
        input_ids = jnp.array([[next_token]])
        
        num_tokens_generated += 1
        
        current_mem = psutil.virtual_memory().used / (1024**3)
        peak_memory = max(peak_memory, current_mem)
        
        token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(f"Token {i+1}: '{token_text}'")
        
        if next_token == tokenizer.eos_token_id or "<|im_end|>" in token_text:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    gc.collect()
    
    print(f"Full output: {full_output}")
    print(f"Peak memory: {peak_memory:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")
    
    return full_output, peak_memory, avg_time_per_token

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    setup_device_mesh()
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    math_questions = [
        "Question: Sam scores 80 on the first test and 90 on the second. What score does he need on the third test to have an average of 85?",
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Question: Katy makes coffee using teaspoons of sugar and cups of water in the ratio of 7:13. If she used a total of 120 teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used.",
        "Question: A craft store makes a third of its sales in the fabric section, a quarter of its sales in the jewelry section, and the rest in the stationery section. They made 36 sales today. How many sales were in the stationery section?",
        "Question: A ticket costs $8. There are 5 friends going to the movie. They have a coupon for $10 off the total. How much do they pay in total?",
        "Question: In a class of 30 students, 40% are girls. How many boys are there?",
        "Question: A recipe requires 2 cups of flour for 12 cookies. How many cups are needed for 30 cookies?",
        "Question: Peter has $100. He buys a shirt for $25, pants for $35, and then finds $10. How much does he have left?",
        "Question: There are 4 apples and 5 oranges in a bowl. John adds 3 more apples and twice as many oranges as the original number of apples. How many fruits are there now?",
        "Question: A bus has 40 passengers. At the first stop, 1/5 get off and 8 get on. How many passengers are there now?"
    ]
    
    for idx, question in enumerate(math_questions, 1):
        print(f"Question {idx}: {question}")
        generate_text(model, params, tokenizer, 325, question)
        print("=" * 80)

if __name__ == "__main__":
    main()