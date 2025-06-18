#!/usr/bin/env python3
"""
Quick test script to verify GSM8K setup works on first 5 problems.
Useful for debugging before running full evaluation.

Usage:
python test_gsm8k_setup.py --model_path ../weights
"""
import os
import sys
import json
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gsm8k_test")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_extract_answer():
    """Test answer extraction function"""
    logger.info("Testing answer extraction...")
    
    # Import from both scripts
    from gsm8k_jax import extract_answer as jax_extract
    from gsm8k_pytorch import extract_answer as pt_extract
    
    test_cases = [
        ("The answer is #### 42", "42"),
        ("Total cost: #### 1,234", "1234"), 
        ("No clear answer here", ""),
        ("#### -5", "-5"),
        ("Multiple #### 10 and #### 20", "10"),  # Should get first
    ]
    
    for text, expected in test_cases:
        jax_result = jax_extract(text)
        pt_result = pt_extract(text)
        
        if jax_result == pt_result == expected:
            logger.info(f"✅ '{text}' -> '{jax_result}'")
        else:
            logger.error(f"❌ '{text}' -> JAX: '{jax_result}', PT: '{pt_result}', Expected: '{expected}'")
            return False
    
    return True

def test_dataset_loading():
    """Test GSM8K dataset loading"""
    logger.info("Testing dataset loading...")
    
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        logger.info(f"✅ Loaded {len(dataset)} test problems")
        
        # Show first problem
        first_problem = dataset[0]
        logger.info(f"Sample question: {first_problem['question'][:100]}...")
        logger.info(f"Sample answer: {first_problem['answer'][:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return False

def test_tokenizer(model_path):
    """Test tokenizer and chat template"""
    logger.info("Testing tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test chat template
        test_question = "What is 2 + 2?"
        messages = [{"role": "user", "content": test_question}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        logger.info(f"✅ Tokenizer loaded")
        logger.info(f"Sample prompt: {prompt[:200]}...")
        
        # Test tokenization
        inputs = tokenizer(prompt, return_tensors="np")
        logger.info(f"Input length: {inputs.input_ids.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Tokenizer test failed: {e}")
        return False

def test_model_loading(model_path, test_jax=True, test_pytorch=True):
    """Test model loading for both frameworks"""
    results = {}
    
    if test_pytorch:
        logger.info("Testing PyTorch model loading...")
        try:
            import torch
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info("✅ PyTorch model loaded successfully")
            results["pytorch"] = True
            
            # Cleanup
            del model
            
        except Exception as e:
            logger.error(f"❌ PyTorch model loading failed: {e}")
            results["pytorch"] = False
    
    if test_jax:
        logger.info("Testing JAX model loading...")
        try:
            import jax.numpy as jnp
            from simple_inference import Qwen25ForCausalLM, load_params
            
            # Load config
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create model
            model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
            
            # Load a small subset of params for testing
            logger.info("Loading JAX model weights (this may take a while)...")
            params = load_params(model, model_path, jnp.float32)
            
            logger.info("✅ JAX model loaded successfully")
            results["jax"] = True
            
            # Cleanup
            del params, model
            
        except Exception as e:
            logger.error(f"❌ JAX model loading failed: {e}")
            results["jax"] = False
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test GSM8K setup")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--skip_models", action="store_true", help="Skip model loading tests")
    
    args = parser.parse_args()
    
    logger.info("🧪 Testing GSM8K Setup")
    logger.info(f"Model path: {args.model_path}")
    
    all_passed = True
    
    # Test 1: Answer extraction
    if not test_extract_answer():
        all_passed = False
    
    # Test 2: Dataset loading  
    if not test_dataset_loading():
        all_passed = False
    
    # Test 3: Tokenizer
    if not test_tokenizer(args.model_path):
        all_passed = False
    
    # Test 4: Model loading (optional, slow)
    if not args.skip_models:
        model_results = test_model_loading(args.model_path)
        if not all(model_results.values()):
            all_passed = False
            logger.warning("Some model loading tests failed - check dependencies")
    else:
        logger.info("⏭️  Skipping model loading tests")
    
    # Summary
    if all_passed:
        logger.info("🎉 All tests passed! Setup is ready for GSM8K evaluation.")
        logger.info("Run: python run_gsm8k_parity.py --model_path ../weights")
    else:
        logger.error("❌ Some tests failed. Please fix issues before running evaluation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 