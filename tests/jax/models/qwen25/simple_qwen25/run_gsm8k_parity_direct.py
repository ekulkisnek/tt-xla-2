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
logger = logging.getLogger("gsm8k_parity_direct")

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
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy

def run_jax_evaluation(model_path, output_path, dtype_str="float32"):
    """Run JAX evaluation directly with progress indicators"""
    logger.info("🔄 Starting JAX GSM8K evaluation...")
    
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from simple_inference import Qwen25ForCausalLM, load_params
    import re
    
    def extract_answer(text: str) -> str:
        match = re.search(r"####\s*(-?\d[\d,]*)", text)
        if match:
            return match.group(1).replace(",", "")
        return ""
    
    def deterministic_generate(model, params, tokenizer, input_ids, max_new_tokens=256):
        """Deterministic greedy generation"""
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
    
    dtype = jnp.float32 if dtype_str == "float32" else jnp.bfloat16
    
    # Load model
    logger.info("📦 Loading JAX model...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, dtype)
    logger.info("✅ JAX model loaded")
    
    # Load GSM8K
    logger.info("📊 Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    logger.info(f"✅ Dataset loaded: {len(dataset)} problems")
    
    correct = 0
    total = 0
    
    logger.info("🚀 Starting JAX evaluation...")
    start_time = time.time()
    
    with open(output_path, 'w') as f:
        for i, ex in enumerate(dataset):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(dataset) - i) / rate if rate > 0 else 0
                logger.info(f"[JAX] {i}/{len(dataset)} ({i/len(dataset)*100:.1f}%) | "
                           f"Rate: {rate:.1f} problems/sec | ETA: {eta/60:.1f}min")
                gc.collect()
            
            # Format prompt (exactly same as PyTorch)
            messages = [{"role": "user", "content": ex["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="np")
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
    total_time = time.time() - start_time
    logger.info(f"✅ JAX evaluation complete!")
    logger.info(f"📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return accuracy

def compare_results(jax_file, pytorch_file):
    """Compare results and show parity analysis"""
    logger.info("🔍 Comparing results...")
    
    # Load results
    jax_results = []
    with open(jax_file, 'r') as f:
        for line in f:
            jax_results.append(json.loads(line.strip()))
    
    pytorch_results = []
    with open(pytorch_file, 'r') as f:
        for line in f:
            pytorch_results.append(json.loads(line.strip()))
    
    if len(jax_results) != len(pytorch_results):
        logger.error(f"Mismatch in result count: JAX={len(jax_results)}, PyTorch={len(pytorch_results)}")
        return
    
    total = len(jax_results)
    jax_correct = sum(1 for r in jax_results if r["correct"])
    pytorch_correct = sum(1 for r in pytorch_results if r["correct"])
    
    both_correct = sum(1 for j, p in zip(jax_results, pytorch_results) if j["correct"] and p["correct"])
    both_wrong = sum(1 for j, p in zip(jax_results, pytorch_results) if not j["correct"] and not p["correct"])
    
    logger.info("=" * 50)
    logger.info("📊 PARITY ANALYSIS RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total problems: {total}")
    logger.info(f"JAX accuracy: {jax_correct/total*100:.2f}% ({jax_correct}/{total})")
    logger.info(f"PyTorch accuracy: {pytorch_correct/total*100:.2f}% ({pytorch_correct}/{total})")
    logger.info(f"Agreement rate: {(both_correct + both_wrong)/total*100:.2f}%")
    
    score_diff = abs(jax_correct - pytorch_correct) / total * 100
    logger.info(f"Score difference: {score_diff:.2f}%")
    
    if score_diff <= 0.1:
        logger.info("🎉 PARITY STATUS: PASS (≤0.1% difference)")
    elif score_diff <= 1.0:
        logger.info("⚠️  PARITY STATUS: CLOSE (≤1.0% difference)")
    else:
        logger.info("❌ PARITY STATUS: FAIL (>1.0% difference)")

def main():
    parser = argparse.ArgumentParser(description="Direct GSM8K parity validation")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--jax_dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch evaluation")
    parser.add_argument("--skip_jax", action="store_true", help="Skip JAX evaluation")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("gsm8k_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pytorch_output = output_dir / "pytorch_gsm8k.jsonl"
    jax_output = output_dir / "jax_gsm8k.jsonl"
    
    logger.info("🚀 Starting Direct GSM8K Parity Validation")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Run evaluations
    if not args.skip_pytorch:
        pytorch_acc = run_pytorch_evaluation(args.model_path, pytorch_output)
    
    if not args.skip_jax:
        jax_acc = run_jax_evaluation(args.model_path, jax_output, args.jax_dtype)
    
    # Compare if both were run
    if not args.skip_pytorch and not args.skip_jax:
        compare_results(jax_output, pytorch_output)
    
    logger.info("✨ Evaluation complete!")

if __name__ == "__main__":
    main() 