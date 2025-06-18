#!/usr/bin/env python3
"""
GSM8K-specific JAX Qwen2.5-7B inference script for parity validation.
Based on simple_inference.py but optimized for GSM8K evaluation with:
- Deterministic generation (temperature=0.0, greedy decoding)
- Correct answer extraction using #### pattern
- Memory efficient dataset processing
- Consistent prompt formatting with PyTorch version

Usage:
python gsm8k_jax_inference.py --model_path ../weights --dataset_path gsm8k_test.jsonl --output_path jax_predictions.jsonl --dtype float32
"""
import os
import sys
import time
import json
import gc
import argparse
import logging
import re
from typing import Dict, Any, Optional, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open
from datasets import load_dataset
from transformers import AutoTokenizer

# Force deterministic behavior
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
jax.config.update('jax_deterministic_ops', True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gsm8k_jax")

# Import model classes from simple_inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_inference import (
    QwenAttention, QwenMLP, QwenDecoderLayer, Qwen25ForCausalLM,
    get_param_path, transpose_if_needed, process_safetensors_file,
    merge_param_dicts, load_params
)

def extract_answer(text: str) -> str:
    """
    Extract numerical answer from GSM8K response using standard #### pattern.
    Returns the number as string, or empty string if not found.
    """
    # GSM8K standard pattern: #### followed by number (possibly with commas)
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        # Remove commas and return just the number
        return match.group(1).replace(",", "")
    return ""

def deterministic_generate(model, params, tokenizer, input_ids, max_new_tokens=256, repetition_penalty=1.1):
    """
    Deterministic greedy generation for GSM8K evaluation.
    Uses temperature=0.0, top_p=1.0, top_k=0 for pure greedy decoding.
    """
    # Deterministic PRNG key
    rng_key = jax.random.PRNGKey(0)
    
    # Convert to JAX arrays if needed
    if not isinstance(input_ids, jnp.ndarray):
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
    
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]  # Add batch dimension
    
    batch_size, seq_length = input_ids.shape
    generated_tokens = []
    current_ids = input_ids
    past_key_values = None
    
    # Track token frequencies for repetition penalty
    token_counts = {}
    for token_id in input_ids[0].tolist():
        token_counts[token_id] = token_counts.get(token_id, 0) + 1
    
    for step in range(max_new_tokens):
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=current_ids,
            past_key_values=past_key_values,
            return_dict=True
        )
        
        logits = outputs["logits"][:, -1, :]  # Get last token logits
        past_key_values = outputs["past_key_values"]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id, count in token_counts.items():
                if count > 0:
                    penalty = repetition_penalty ** count
                    if logits[0, token_id] > 0:
                        logits = logits.at[0, token_id].set(logits[0, token_id] / penalty)
                    else:
                        logits = logits.at[0, token_id].set(logits[0, token_id] * penalty)
        
        # Pure greedy decoding (temperature=0.0)
        next_token = jnp.argmax(logits, axis=-1)
        next_token_id = int(next_token[0])
        
        # Update token counts
        token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
        
        generated_tokens.append(next_token_id)
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Update for next iteration
        current_ids = next_token[:, None]
    
    return generated_tokens

def evaluate_gsm8k(model, params, tokenizer, dataset_path: str, output_path: str, dtype=jnp.bfloat16):
    """
    Evaluate model on GSM8K dataset with memory-efficient processing.
    """
    logger.info(f"Loading GSM8K dataset from {dataset_path}")
    
    # Load dataset
    if dataset_path == "hf" or dataset_path is None:
        # Load from HuggingFace datasets
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        problems = [{"question": ex["question"], "answer": ex["answer"]} for ex in dataset]
    else:
        # Load JSONL file
        problems = []
        with open(dataset_path, 'r') as f:
            for line in f:
                problems.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(problems)} problems")
    
    predictions = []
    correct = 0
    total = 0
    
    for i, problem in enumerate(problems):
        if i % 100 == 0:
            logger.info(f"Processing problem {i+1}/{len(problems)}")
            gc.collect()  # Memory cleanup every 100 problems
        
        question = problem["question"]
        
        # Apply chat template (same as PyTorch version)
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs.input_ids
        
        try:
            # Generate response
            generated_tokens = deterministic_generate(
                model, params, tokenizer, input_ids, 
                max_new_tokens=256, repetition_penalty=1.1
            )
            
            # Decode response
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract predicted answer
            predicted_answer = extract_answer(response)
            
            # Extract ground truth answer
            if "answer" in problem:
                # Extract from GSM8K format answer
                gt_answer = extract_answer(problem["answer"])
            else:
                gt_answer = str(problem.get("answer", ""))
            
            # Check correctness (compare numbers, ignoring commas)
            is_correct = predicted_answer == gt_answer
            if is_correct:
                correct += 1
            total += 1
            
            # Store prediction
            prediction = {
                "id": i,
                "question": question,
                "response": response,
                "predicted_answer": predicted_answer,
                "ground_truth": gt_answer,
                "correct": is_correct
            }
            predictions.append(prediction)
            
            # Log some examples
            if i < 5 or (i + 1) % 100 == 0:
                logger.info(f"Problem {i+1}: Predicted={predicted_answer}, GT={gt_answer}, Correct={is_correct}")
                logger.info(f"Response: {response[:200]}...")
        
        except Exception as e:
            logger.error(f"Error processing problem {i}: {e}")
            gt_answer = extract_answer(problem["answer"]) if "answer" in problem else ""
            prediction = {
                "id": i,
                "question": question,
                "response": f"ERROR: {str(e)}",
                "predicted_answer": "",
                "ground_truth": gt_answer,
                "correct": False
            }
            predictions.append(prediction)
            total += 1
    
    # Calculate final accuracy
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    logger.info(f"Final Results:")
    logger.info(f"Total problems: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    
    # Save predictions
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    # Save summary
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "model": "JAX Qwen2.5-7B",
        "dtype": str(dtype),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = output_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Summary saved to {summary_path}")
    
    return accuracy, predictions

def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation with JAX Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--dataset_path", type=str, default="hf", help="Path to GSM8K dataset (jsonl) or 'hf' for HuggingFace")
    parser.add_argument("--output_path", type=str, default="jax_gsm8k_predictions.jsonl", help="Output predictions file")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Model dtype")
    
    args = parser.parse_args()
    
    # Set dtype
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    
    logger.info(f"Starting GSM8K evaluation with dtype={dtype}")
    
    # Load model config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load weights
    logger.info("Loading model weights...")
    params = load_params(model, args.model_path, dtype)
    
    # Memory cleanup
    gc.collect()
    jax.clear_caches()
    
    # Run evaluation
    logger.info("Starting GSM8K evaluation...")
    accuracy, predictions = evaluate_gsm8k(
        model, params, tokenizer, args.dataset_path, args.output_path, dtype
    )
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()
    
    logger.info(f"GSM8K evaluation completed. Final accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 