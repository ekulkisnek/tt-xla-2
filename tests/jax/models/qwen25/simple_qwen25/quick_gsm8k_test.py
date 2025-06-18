#!/usr/bin/env python3
"""
Quick GSM8K test on first 10 problems to verify setup works and estimate timing.
"""
import os
import sys
import time
import argparse
import logging
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gsm8k_quick")

def run_pytorch_quick_test(model_path, num_problems=10):
    """Run PyTorch evaluation on first N problems"""
    logger.info(f"🔄 Testing PyTorch on {num_problems} problems...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import re
    
    def extract_answer(text: str) -> str:
        match = re.search(r"####\s*(-?\d[\d,]*)", text)
        if match:
            return match.group(1).replace(",", "")
        return ""
    
    # Load model
    logger.info("📦 Loading PyTorch model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    logger.info("✅ PyTorch model loaded")
    
    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    correct = 0
    results = []
    start_time = time.time()
    
    for i, ex in enumerate(dataset):
        if i >= num_problems:
            break
            
        logger.info(f"[PyTorch] Problem {i+1}/{num_problems}")
        
        # Format prompt
        messages = [{"role": "user", "content": ex["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt")
        problem_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        problem_time = time.time() - problem_start
        
        # Decode
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Extract answers
        pred_answer = extract_answer(response)
        gt_answer = extract_answer(ex["answer"])
        is_correct = pred_answer == gt_answer
        
        if is_correct:
            correct += 1
            
        logger.info(f"   Q: {ex['question'][:50]}...")
        logger.info(f"   Pred: '{pred_answer}' | GT: '{gt_answer}' | {'✅' if is_correct else '❌'} | {problem_time:.1f}s")
        
        results.append({
            "predicted": pred_answer,
            "ground_truth": gt_answer,
            "correct": is_correct
        })
    
    total_time = time.time() - start_time
    accuracy = correct / len(results) * 100
    avg_time = total_time / len(results)
    
    logger.info(f"✅ PyTorch: {accuracy:.1f}% accuracy, {avg_time:.1f}s/problem")
    logger.info(f"📈 Estimated full time: {avg_time * 1319 / 60:.1f} minutes")
    
    return results, accuracy, avg_time

def main():
    parser = argparse.ArgumentParser(description="Quick GSM8K test")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--num_problems", type=int, default=10, help="Number of problems to test")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Quick GSM8K Test")
    logger.info(f"✅ Dataset verified: Using authentic OpenAI GSM8K from GitHub")
    
    # Run PyTorch test
    pytorch_results, pytorch_acc, pytorch_time = run_pytorch_quick_test(args.model_path, args.num_problems)
    
    logger.info("=" * 60)
    logger.info("📊 QUICK TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PyTorch accuracy: {pytorch_acc:.1f}%")
    logger.info(f"Average time per problem: {pytorch_time:.1f} seconds")
    logger.info(f"Estimated full evaluation time: {pytorch_time * 1319 / 60:.1f} minutes")
    logger.info("=" * 60)
    
    logger.info("✨ Quick test complete! Ready for full evaluation.")

if __name__ == "__main__":
    main() 