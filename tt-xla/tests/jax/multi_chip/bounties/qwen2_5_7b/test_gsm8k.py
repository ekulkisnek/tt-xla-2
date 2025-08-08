#!/usr/bin/env python3
"""
GSM8K evaluation script for Qwen2.5-7B TP in JAX.
Loads dataset, runs inference on samples, extracts answers, computes accuracy.
Supports single-device mode for equivalence check.
Usage: python test_gsm8k.py --model_path weights --num_samples 100 --single_device
"""

import argparse
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer
import re
from model import Qwen25ForCausalLM, load_params, sample_next_token, make_causal_mask, setup_device_mesh, mesh  # Import from model.py
import os
import json

def extract_boxed_answer(text):
    """Extract the final boxed answer from generated text."""
    match = re.search(r'\boxed{([0-9]+)}', text)
    return int(match.group(1)) if match else None

def evaluate_gsm8k(model, params, tokenizer, num_samples=10, single_device=False):
    dataset = load_dataset("gsm8k", "main", split="test")[:num_samples]
    
    correct = 0
    for example in dataset:
        prompt = example["question"]
        target = int(example["answer"].split("#### ")[-1])
        
        # Generate response (using same logic as generate_text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(formatted_text, return_tensors="jax")
        batch, seq = input_ids.shape
        position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
        past_key_values = None
        generated_tokens = []
        
        for _ in range(500):  # Max tokens
            current_seq_len = input_ids.shape[1]
            key_len = current_seq_len if past_key_values is None else past_key_values[0][0].shape[1] + current_seq_len
            attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)
            
            outputs = model.apply(params, input_ids=input_ids, attention_mask=attention_mask, 
                                 position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            
            next_token = sample_next_token(logits[:, -1, :])
            generated_tokens.append(next_token)
            input_ids = jnp.array([[next_token]])
            position_ids = position_ids[:, -1:] + 1
            
            if next_token == tokenizer.eos_token_id:
                break
        
        output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predicted = extract_boxed_answer(output)
        
        if predicted == target:
            correct += 1
        print(f"Question: {prompt}\nPredicted: {predicted} | Target: {target}")
    
    accuracy = correct / num_samples * 100
    print(f"GSM8K Accuracy ({num_samples} samples): {accuracy:.2f}%")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="GSM8K Evaluation for Qwen2.5-7B TP")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--num_samples", type=int, default=10, help="Number of GSM8K samples to evaluate")
    parser.add_argument("--single_device", action="store_true", help="Run in single-device mode for equivalence check")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    if args.single_device:
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    
    global mesh
    mesh = setup_device_mesh()
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    accuracy = evaluate_gsm8k(model, params, tokenizer, args.num_samples, args.single_device)
    # For equivalence, run once with --single_device and compare to TP run

if __name__ == "__main__":
    main()