#!/usr/bin/env python3
"""
GSM8K-specific PyTorch Qwen2.5-7B inference script for parity validation.
Based on pytorch_q25.py but optimized for GSM8K evaluation with:
- Deterministic generation (temperature=0.0, greedy decoding)
- Correct answer extraction using #### pattern
- Memory efficient dataset processing
- Consistent prompt formatting with JAX version

Usage:
python gsm8k_pytorch_inference.py --model_path ../weights --dataset_path gsm8k_test.jsonl --output_path pytorch_predictions.jsonl
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import psutil

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gsm8k_pytorch")

def check_memory():
    """Check available system memory"""
    memory = psutil.virtual_memory()
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    return memory.available / (1024**3)

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

def load_model_optimized(model_path: str):
    """
    Load Qwen2.5 model with memory optimizations for GSM8K evaluation.
    """
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    logger.info("Loading model with memory optimizations...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        device_map="cpu",  # Force CPU to match memory constraints
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_compile=False  # Disable compilation for deterministic behavior
    )
    
    logger.info("Model loaded successfully")
    check_memory()
    
    return model, tokenizer

def deterministic_generate(model, tokenizer, input_ids, max_new_tokens=256, repetition_penalty=1.1):
    """
    Deterministic greedy generation for GSM8K evaluation.
    Uses temperature=0.0, top_p=1.0, top_k=0 for pure greedy decoding.
    """
    # Set model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Generate with deterministic parameters matching JAX version
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (temperature=0.0)
            temperature=None,  # Disable temperature when do_sample=False
            top_p=1.0,
            top_k=0,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False
        )
    
    # Extract only the generated tokens (remove input)
    generated_tokens = generated_ids[0][len(input_ids[0]):]
    
    return generated_tokens.tolist()

def evaluate_gsm8k(model, tokenizer, dataset_path: str, output_path: str):
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
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        question = problem["question"]
        
        # Apply chat template (same as JAX version)
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        try:
            # Generate response
            generated_tokens = deterministic_generate(
                model, tokenizer, input_ids,
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
        "model": "PyTorch Qwen2.5-7B",
        "dtype": "float16",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    summary_path = output_path.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Summary saved to {summary_path}")
    
    return accuracy, predictions

def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation with PyTorch Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--dataset_path", type=str, default="hf", help="Path to GSM8K dataset (jsonl) or 'hf' for HuggingFace")
    parser.add_argument("--output_path", type=str, default="pytorch_gsm8k_predictions.jsonl", help="Output predictions file")
    
    args = parser.parse_args()
    
    logger.info("Starting GSM8K evaluation with PyTorch")
    check_memory()
    
    # Load model and tokenizer
    model, tokenizer = load_model_optimized(args.model_path)
    
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Run evaluation
    logger.info("Starting GSM8K evaluation...")
    accuracy, predictions = evaluate_gsm8k(
        model, tokenizer, args.dataset_path, args.output_path
    )
    
    # Final cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info(f"GSM8K evaluation completed. Final accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 