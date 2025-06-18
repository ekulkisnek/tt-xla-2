#!/usr/bin/env python3
"""
Quick 5-problem GSM8K parity test for both JAX and PyTorch with bfloat16.
"""
import os
import sys
import time
import logging
import json
import gc

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_5")

def extract_answer(text: str) -> str:
    import re
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    return ""

def run_pytorch_test(model_path, num_problems=5):
    """Run PyTorch evaluation"""
    logger.info(f"🔄 PyTorch test ({num_problems} problems, bfloat16)...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    start_time = time.time()
    
    for i, ex in enumerate(dataset):
        if i >= num_problems:
            break
            
        logger.info(f"[PyTorch] {i+1}/{num_problems}: {ex['question'][:40]}...")
        
        messages = [{"role": "user", "content": ex["question"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        pred_answer = extract_answer(response)
        gt_answer = extract_answer(ex["answer"])
        is_correct = pred_answer == gt_answer
        
        logger.info(f"   Pred: '{pred_answer}' | GT: '{gt_answer}' | {'✅' if is_correct else '❌'}")
        
        results.append({
            "predicted": pred_answer,
            "ground_truth": gt_answer,
            "correct": is_correct
        })
    
    total_time = time.time() - start_time
    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    
    logger.info(f"✅ PyTorch: {accuracy:.1f}% accuracy, {total_time/len(results):.1f}s/problem")
    
    del model
    gc.collect()
    
    return results

def run_jax_test(model_path, num_problems=5):
    """Run JAX evaluation"""
    logger.info(f"🔄 JAX test ({num_problems} problems, bfloat16)...")
    
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from simple_inference import Qwen25ForCausalLM, load_params
    
    def generate(model, params, tokenizer, input_ids, max_new_tokens=256):
        if not isinstance(input_ids, jnp.ndarray):
            input_ids = jnp.array(input_ids, dtype=jnp.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        
        generated_tokens = []
        current_ids = input_ids
        past_key_values = None
        
        for step in range(max_new_tokens):
            outputs = model.apply(params, input_ids=current_ids, past_key_values=past_key_values, return_dict=True)
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            next_token = jnp.argmax(logits, axis=-1)
            next_token_id = int(next_token[0])
            generated_tokens.append(next_token_id)
            
            if next_token_id == tokenizer.eos_token_id:
                break
            
            current_ids = next_token[:, None]
        
        return generated_tokens
    
    # Load model
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    params = load_params(model, model_path, jnp.bfloat16)
    
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    start_time = time.time()
    
    for i, ex in enumerate(dataset):
        if i >= num_problems:
            break
            
        logger.info(f"[JAX] {i+1}/{num_problems}: {ex['question'][:40]}...")
        
        messages = [{"role": "user", "content": ex["question"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="np")
        generated_tokens = generate(model, params, tokenizer, inputs.input_ids, max_new_tokens=256)
        
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        pred_answer = extract_answer(response)
        gt_answer = extract_answer(ex["answer"])
        is_correct = pred_answer == gt_answer
        
        logger.info(f"   Pred: '{pred_answer}' | GT: '{gt_answer}' | {'✅' if is_correct else '❌'}")
        
        results.append({
            "predicted": pred_answer,
            "ground_truth": gt_answer,
            "correct": is_correct
        })
        
        gc.collect()
    
    total_time = time.time() - start_time
    accuracy = sum(r["correct"] for r in results) / len(results) * 100
    
    logger.info(f"✅ JAX: {accuracy:.1f}% accuracy, {total_time/len(results):.1f}s/problem")
    
    del params, model
    gc.collect()
    jax.clear_caches()
    
    return results

def main():
    model_path = "../weights"
    num_problems = 5
    
    logger.info("🚀 Quick GSM8K Parity Test (5 problems, bfloat16)")
    logger.info("✅ Using authentic OpenAI GSM8K dataset")
    
    # Run tests
    pytorch_results = run_pytorch_test(model_path, num_problems)
    jax_results = run_jax_test(model_path, num_problems)
    
    # Compare
    logger.info("=" * 60)
    logger.info("📊 PARITY COMPARISON")
    logger.info("=" * 60)
    
    matches = 0
    for i, (jax_r, pt_r) in enumerate(zip(jax_results, pytorch_results)):
        match = jax_r["predicted"] == pt_r["predicted"]
        if match:
            matches += 1
        status = "✅ MATCH" if match else "❌ DIFFER"
        logger.info(f"Problem {i+1}: JAX='{jax_r['predicted']}' PyTorch='{pt_r['predicted']}' {status}")
    
    jax_acc = sum(r["correct"] for r in jax_results) / len(jax_results) * 100
    pt_acc = sum(r["correct"] for r in pytorch_results) / len(pytorch_results) * 100
    
    logger.info(f"\n📊 JAX accuracy: {jax_acc:.1f}%")
    logger.info(f"📊 PyTorch accuracy: {pt_acc:.1f}%")
    logger.info(f"📏 Score difference: {abs(jax_acc - pt_acc):.1f}%")
    logger.info(f"🤝 Agreement rate: {matches/len(jax_results)*100:.1f}%")
    
    if abs(jax_acc - pt_acc) <= 20:  # Relaxed for small sample
        logger.info("🎉 PARITY STATUS: GOOD (small sample)")
    else:
        logger.info("❌ PARITY STATUS: NEEDS INVESTIGATION")
    
    logger.info("✨ Quick test complete!")

if __name__ == "__main__":
    main() 