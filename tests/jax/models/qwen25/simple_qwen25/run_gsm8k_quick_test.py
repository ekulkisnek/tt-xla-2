#!/usr/bin/env python3
"""
Quick GSM8K test on first 20 problems to verify setup works.
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
logger = logging.getLogger("gsm8k_quick")

def run_pytorch_quick_test(model_path, output_path, num_problems=20):
    """Run PyTorch evaluation on first N problems"""
    logger.info(f"🔄 Starting PyTorch GSM8K quick test ({num_problems} problems)...")
    
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
    logger.info(f"✅ Dataset loaded: {len(dataset)} problems (testing first {num_problems})")
    
    correct = 0
    total = 0
    
    logger.info("🚀 Starting PyTorch evaluation...")
    start_time = time.time()
    
    with open(output_path, 'w') as f:
        for i, ex in enumerate(dataset):
            if i >= num_problems:
                break
                
            logger.info(f"[PyTorch] Problem {i+1}/{num_problems}: {ex['question'][:60]}...")
            
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
            total += 1
            
            logger.info(f"   Predicted: '{pred_answer}' | Ground Truth: '{gt_answer}' | "
                       f"{'✅ Correct' if is_correct else '❌ Wrong'} | Time: {problem_time:.1f}s")
            
            result = {
                "id": i,
                "question": ex["question"],
                "predicted": pred_answer,
                "ground_truth": gt_answer,
                "correct": is_correct,
                "response": response[:200] + "..." if len(response) > 200 else response
            }
            f.write(json.dumps(result) + '\n')
    
    accuracy = correct / total * 100
    total_time = time.time() - start_time
    avg_time = total_time / total
    
    logger.info("=" * 60)
    logger.info(f"✅ PyTorch quick test complete!")
    logger.info(f"📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"⚡ Average time per problem: {avg_time:.1f} seconds")
    logger.info(f"📈 Estimated full evaluation time: {avg_time * 1319 / 60:.1f} minutes")
    logger.info("=" * 60)
    
    return accuracy

def run_jax_quick_test(model_path, output_path, num_problems=20):
    """Run JAX evaluation on first N problems"""
    logger.info(f"🔄 Starting JAX GSM8K quick test ({num_problems} problems)...")
    
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
    
    # Load model
    logger.info("📦 Loading JAX model...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.float32)
    logger.info("✅ JAX model loaded")
    
    # Load GSM8K
    logger.info("📊 Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    logger.info(f"✅ Dataset loaded: {len(dataset)} problems (testing first {num_problems})")
    
    correct = 0
    total = 0
    
    logger.info("🚀 Starting JAX evaluation...")
    start_time = time.time()
    
    with open(output_path, 'w') as f:
        for i, ex in enumerate(dataset):
            if i >= num_problems:
                break
                
            logger.info(f"[JAX] Problem {i+1}/{num_problems}: {ex['question'][:60]}...")
            
            # Format prompt (exactly same as PyTorch)
            messages = [{"role": "user", "content": ex["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="np")
            problem_start = time.time()
            generated_tokens = deterministic_generate(
                model, params, tokenizer, inputs.input_ids, max_new_tokens=256
            )
            problem_time = time.time() - problem_start
            
            # Decode
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract answers
            pred_answer = extract_answer(response)
            gt_answer = extract_answer(ex["answer"])
            
            is_correct = pred_answer == gt_answer
            if is_correct:
                correct += 1
            total += 1
            
            logger.info(f"   Predicted: '{pred_answer}' | Ground Truth: '{gt_answer}' | "
                       f"{'✅ Correct' if is_correct else '❌ Wrong'} | Time: {problem_time:.1f}s")
            
            result = {
                "id": i,
                "question": ex["question"],
                "predicted": pred_answer,
                "ground_truth": gt_answer,
                "correct": is_correct,
                "response": response[:200] + "..." if len(response) > 200 else response
            }
            f.write(json.dumps(result) + '\n')
    
    accuracy = correct / total * 100
    total_time = time.time() - start_time
    avg_time = total_time / total
    
    logger.info("=" * 60)
    logger.info(f"✅ JAX quick test complete!")
    logger.info(f"📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"⚡ Average time per problem: {avg_time:.1f} seconds")
    logger.info(f"📈 Estimated full evaluation time: {avg_time * 1319 / 60:.1f} minutes")
    logger.info("=" * 60)
    
    return accuracy

def compare_quick_results(jax_file, pytorch_file):
    """Compare quick test results"""
    logger.info("🔍 Comparing quick test results...")
    
    # Load results
    jax_results = []
    with open(jax_file, 'r') as f:
        for line in f:
            jax_results.append(json.loads(line.strip()))
    
    pytorch_results = []
    with open(pytorch_file, 'r') as f:
        for line in f:
            pytorch_results.append(json.loads(line.strip()))
    
    logger.info("=" * 60)
    logger.info("📊 QUICK TEST PARITY ANALYSIS")
    logger.info("=" * 60)
    
    for i, (jax_r, pt_r) in enumerate(zip(jax_results, pytorch_results)):
        match = "✅ MATCH" if jax_r["predicted"] == pt_r["predicted"] else "❌ DIFFER"
        logger.info(f"Problem {i+1}: JAX='{jax_r['predicted']}' PyTorch='{pt_r['predicted']}' {match}")
    
    jax_acc = sum(1 for r in jax_results if r["correct"]) / len(jax_results) * 100
    pt_acc = sum(1 for r in pytorch_results if r["correct"]) / len(pytorch_results) * 100
    
    logger.info(f"\n📊 JAX accuracy: {jax_acc:.1f}%")
    logger.info(f"📊 PyTorch accuracy: {pt_acc:.1f}%")
    logger.info(f"📏 Difference: {abs(jax_acc - pt_acc):.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Quick GSM8K test")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--num_problems", type=int, default=20, help="Number of problems to test")
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch")
    parser.add_argument("--skip_jax", action="store_true", help="Skip JAX")
    
    args = parser.parse_args()
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("gsm8k_quick_test") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pytorch_output = output_dir / "pytorch_quick.jsonl"
    jax_output = output_dir / "jax_quick.jsonl"
    
    logger.info("🚀 Starting Quick GSM8K Test")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Run evaluations
    if not args.skip_pytorch:
        pytorch_acc = run_pytorch_quick_test(args.model_path, pytorch_output, args.num_problems)
    
    if not args.skip_jax:
        jax_acc = run_jax_quick_test(args.model_path, jax_output, args.num_problems)
    
    # Compare if both were run
    if not args.skip_pytorch and not args.skip_jax:
        compare_quick_results(jax_output, pytorch_output)
    
    logger.info("✨ Quick test complete!")

if __name__ == "__main__":
    main() 