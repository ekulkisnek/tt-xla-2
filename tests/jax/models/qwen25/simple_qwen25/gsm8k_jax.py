#!/usr/bin/env python3
"""
GSM8K-specific JAX Qwen2.5-7B inference script for parity validation.
"""
import os
import json
import gc
import argparse
import logging
import re
import sys
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

# Force deterministic behavior
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["JAX_PLATFORMS"] = "cpu"
# Note: jax_deterministic_ops was removed in newer JAX versions
# Setting XLA_FLAGS should be sufficient for deterministic behavior

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gsm8k_jax")

# Import model from simple_inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_inference import Qwen25ForCausalLM, load_params

def extract_answer(text: str) -> str:
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""

def deterministic_generate(model, params, tokenizer, input_ids, max_new_tokens=256):
    """Deterministic greedy generation matching PyTorch"""
    if not isinstance(input_ids, jnp.ndarray):
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
    
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    
    generated_tokens = []
    current_ids = input_ids
    past_key_values = None
    
    for step in range(max_new_tokens):
        outputs = model.apply(
            params,
            input_ids=current_ids,
            past_key_values=past_key_values,
            return_dict=True
        )
        
        logits = outputs["logits"][:, -1, :]
        past_key_values = outputs["past_key_values"]
        
        # Greedy decoding
        next_token = jnp.argmax(logits, axis=-1)
        next_token_id = int(next_token[0])
        
        generated_tokens.append(next_token_id)
        
        if next_token_id == tokenizer.eos_token_id:
            break
        
        current_ids = next_token[:, None]
    
    return generated_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", default="jax_gsm8k.jsonl")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    
    # Load model
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    correct = 0
    total = 0
    
    with open(args.output_path, 'w') as f:
        for i, ex in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"[JAX] Processing {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%) - Question: {ex['question'][:50]}...")
                gc.collect()
            
            # Format prompt (exactly same as PyTorch)
            messages = [{"role": "user", "content": ex["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="np")
            if i % 50 == 0:
                logger.info(f"[JAX] Generating response for problem {i}...")
            generated_tokens = deterministic_generate(
                model, params, tokenizer, inputs.input_ids, max_new_tokens=256
            )
            
            # Decode
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract answers
            pred_answer = extract_answer(response)
            gt_answer = extract_answer(ex["answer"])
            
            is_correct = pred_answer == gt_answer
            if is_correct:
                correct += 1
            total += 1
            
            result = {
                "id": i,
                "predicted": pred_answer,
                "ground_truth": gt_answer,
                "correct": is_correct
            }
            f.write(json.dumps(result) + '\n')
    
    accuracy = correct / total * 100
    logger.info(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 