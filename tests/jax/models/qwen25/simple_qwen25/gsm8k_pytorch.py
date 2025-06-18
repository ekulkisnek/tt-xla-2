#!/usr/bin/env python3
"""
GSM8K-specific PyTorch Qwen2.5-7B inference script for parity validation.
"""
import os
import json
import gc
import argparse
import logging
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gsm8k_pytorch")

def extract_answer(text: str) -> str:
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", default="pytorch_gsm8k.jsonl")
    args = parser.parse_args()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    correct = 0
    total = 0
    
    with open(args.output_path, 'w') as f:
        for i, ex in enumerate(dataset):
            if i % 10 == 0:
                logger.info(f"[PyTorch] Processing {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%) - Question: {ex['question'][:50]}...")
            
            # Format prompt
            messages = [{"role": "user", "content": ex["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            if i % 50 == 0:
                logger.info(f"[PyTorch] Generating response for problem {i}...")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
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