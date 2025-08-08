#!/usr/bin/env python3
"""
GSM8K evaluation script for Qwen2.5-7B TP in JAX.
"""

import argparse
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer
import re
from model import Qwen25ForCausalLM, load_params, sample_next_token, setup_device_mesh, mesh
import os
import json

def extract_boxed_answer(text):
    match = re.search(r'\boxed{([0-9]+)}', text)
    return int(match.group(1)) if match else None

def evaluate_gsm8k(model, params, tokenizer, num_samples=10):
    dataset = load_dataset("gsm8k", "main", split="test")[:num_samples]
    
    correct = 0
    for example in dataset:
        prompt = example["question"]
        target = int(example["answer"].split("#### ")[-1])
        
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
        
        for _ in range(325):
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
    parser.add_argument("--num_samples", type=int, default=10, help="Number of GSM8K samples")
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    setup_device_mesh()
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    evaluate_gsm8k(model, params, tokenizer, args.num_samples)

if __name__ == "__main__":
    main()