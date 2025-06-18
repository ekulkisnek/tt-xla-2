#!/usr/bin/env python3
"""
Direct GSM8K parity validation with real-time progress indicators.
Runs evaluations directly instead of through subprocess to show progress.
"""
import os
import sys
import time
import argparse
import logging
import json
import gc
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gsm8k_direct")

def run_pytorch_evaluation(model_path, output_path):
    """Run PyTorch evaluation directly with progress indicators"""
    logger.info("🔄 Starting PyTorch GSM8K evaluation...")
    
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
    logger.info("📊 Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    logger.info(f"✅ Dataset loaded: {len(dataset)} problems")
    
    correct = 0
    total = 0
    
    logger.info("🚀 Starting PyTorch evaluation...")
    start_time = time.time()
    
    with open(output_path, 'w') as f:
        for i, ex in enumerate(dataset):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(dataset) - i) / rate if rate > 0 else 0
                logger.info(f"[PyTorch] {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%) | "
                           f"Rate: {rate:.1f} problems/sec | ETA: {eta/60:.1f}min")
            
            # Format prompt
            messages = [{"role": "user", "content": ex["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    do_sample=False,
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
    total_time = time.time() - start_time
    logger.info(f"✅ PyTorch evaluation complete!")
    logger.info(f"📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Direct PyTorch GSM8K evaluation with progress")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("gsm8k_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pytorch_output = output_dir / "pytorch_gsm8k.jsonl"
    
    logger.info("🚀 Starting Direct PyTorch GSM8K Evaluation")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Run evaluation
    pytorch_acc = run_pytorch_evaluation(args.model_path, pytorch_output)
    
    logger.info("✨ Evaluation complete!")

if __name__ == "__main__":
    main() 