#!/usr/bin/env python3
"""
Quick 5-problem GSM8K parity test for both JAX and PyTorch with bfloat16.
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
logger = logging.getLogger("quick_parity")

def extract_answer(text: str) -> str:
    import re
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""

def run_pytorch_test(model_path, num_problems=5):
    """Run PyTorch evaluation on first N problems with bfloat16"""
    logger.info(f"🔄 Testing PyTorch on {num_problems} problems (bfloat16)...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    # Load model with bfloat16
    logger.info("📦 Loading PyTorch model (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    logger.info("✅ PyTorch model loaded")
    
    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    correct = 0
    start_time = time.time()
    
    for i, ex in enumerate(dataset):
        if i >= num_problems:
            break
            
        logger.info(f"[PyTorch] Problem {i+1}/{num_problems}: {ex['question'][:50]}...")
        
        # Format prompt
        messages = [{"role": "user", "content": ex["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        problem_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
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
            
        logger.info(f"   Pred: '{pred_answer}' | GT: '{gt_answer}' | {'✅' if is_correct else '❌'} | {problem_time:.1f}s")
        
        results.append({
            "id": i,
            "predicted": pred_answer,
            "ground_truth": gt_answer,
            "correct": is_correct,
            "response": response[:100] + "..." if len(response) > 100 else response
        })
    
    total_time = time.time() - start_time
    accuracy = correct / len(results) * 100
    avg_time = total_time / len(results)
    
    logger.info(f"✅ PyTorch: {accuracy:.1f}% accuracy, {avg_time:.1f}s/problem")
    
    # Cleanup
    del model
    gc.collect()
    
    return results, accuracy, avg_time

def run_jax_test(model_path, num_problems=5):
    """Run JAX evaluation on first N problems with bfloat16"""
    logger.info(f"🔄 Testing JAX on {num_problems} problems (bfloat16)...")
    
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from simple_inference import Qwen25ForCausalLM, load_params
    
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
    
    # Load model with bfloat16
    logger.info("📦 Loading JAX model (bfloat16)...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    logger.info("✅ JAX model loaded")
    
    # Load GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    correct = 0
    start_time = time.time()
    
    for i, ex in enumerate(dataset):
        if i >= num_problems:
            break
            
        logger.info(f"[JAX] Problem {i+1}/{num_problems}: {ex['question'][:50]}...")
        
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
            
        logger.info(f"   Pred: '{pred_answer}' | GT: '{gt_answer}' | {'✅' if is_correct else '❌'} | {problem_time:.1f}s")
        
        results.append({
            "id": i,
            "predicted": pred_answer,
            "ground_truth": gt_answer,
            "correct": is_correct,
            "response": response[:100] + "..." if len(response) > 100 else response
        })
        
        gc.collect()
    
    total_time = time.time() - start_time
    accuracy = correct / len(results) * 100
    avg_time = total_time / len(results)
    
    logger.info(f"✅ JAX: {accuracy:.1f}% accuracy, {avg_time:.1f}s/problem")
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return results, accuracy, avg_time

def compare_results(jax_results, pytorch_results):
    """Compare JAX and PyTorch results"""
    logger.info("=" * 80)
    logger.info("📊 PARITY ANALYSIS (5-Problem Quick Test)")
    logger.info("=" * 80)
    
    matches = 0
    for i, (jax_r, pt_r) in enumerate(zip(jax_results, pytorch_results)):
        match = jax_r["predicted"] == pt_r["predicted"]
        if match:
            matches += 1
        status = "✅ MATCH" if match else "❌ DIFFER"
        logger.info(f"Problem {i+1}: JAX='{jax_r['predicted']}' PyTorch='{pt_r['predicted']}' {status}")
    
    jax_acc = sum(1 for r in jax_results if r["correct"]) / len(jax_results) * 100
    pt_acc = sum(1 for r in pytorch_results if r["correct"]) / len(pytorch_results) * 100
    agreement_rate = matches / len(jax_results) * 100
    
    logger.info(f"\n📊 JAX accuracy: {jax_acc:.1f}%")
    logger.info(f"📊 PyTorch accuracy: {pt_acc:.1f}%")
    logger.info(f"📏 Score difference: {abs(jax_acc - pt_acc):.1f}%")
    logger.info(f"🤝 Agreement rate: {agreement_rate:.1f}%")
    
    if abs(jax_acc - pt_acc) <= 0.1:
        logger.info("🎉 PARITY STATUS: PASS (≤0.1% difference)")
    elif abs(jax_acc - pt_acc) <= 1.0:
        logger.info("⚠️  PARITY STATUS: CLOSE (≤1.0% difference)")
    else:
        logger.info("❌ PARITY STATUS: FAIL (>1.0% difference)")

def main():
    parser = argparse.ArgumentParser(description="Quick 5-problem parity test")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--num_problems", type=int, default=5, help="Number of problems to test")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Quick GSM8K Parity Test (bfloat16)")
    logger.info(f"📊 Testing {args.num_problems} problems")
    logger.info("✅ Dataset: Authentic OpenAI GSM8K from GitHub")
    
    # Run PyTorch test
    pytorch_results, pytorch_acc, pytorch_time = run_pytorch_test(args.model_path, args.num_problems)
    
    # Run JAX test  
    jax_results, jax_acc, jax_time = run_jax_test(args.model_path, args.num_problems)
    
    # Compare results
    compare_results(jax_results, pytorch_results)
    
    logger.info("=" * 80)
    logger.info("⏱️  TIMING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"PyTorch avg time: {pytorch_time:.1f}s/problem")
    logger.info(f"JAX avg time: {jax_time:.1f}s/problem")
    logger.info(f"Speed ratio: {pytorch_time/jax_time:.1f}x (PyTorch/JAX)")
    logger.info("=" * 80)
    
    logger.info("✨ Quick parity test complete!")

if __name__ == "__main__":
    main() 