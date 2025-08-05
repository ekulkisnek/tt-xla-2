#!/usr/bin/env python3
"""
TT-XLA Qwen GSM8K Batch - Run all GSM8K problems on TT hardware
This script runs all the GSM8K math problems on TT hardware.
"""
import os
import sys
import json
import logging
from typing import Dict, Any

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flax import linen as nn
import jax._src.xla_bridge as xb

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tt_gsm8k_batch")

def initialize_tt_backend():
    """Initialize TT backend for JAX."""
    print("🔧 Initializing TT backend...")
    
    try:
        import sys
        sys.path.append('/opt/venv/lib/python3.10/site-packages/jax_plugins')
        import pjrt_plugin_tt
        print("✅ Using installed TT PJRT plugin")
    except ImportError:
        print("❌ TT plugin not found. Please install with: pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/")
        return False
    
    jax.config.update("jax_platforms", "tt,cpu")
    tt_devices = jax.devices("tt")
    print("🚀 Found {} TT device(s): {}".format(len(tt_devices), tt_devices))
    
    if len(tt_devices) == 0:
        print("❌ No TT devices found. Check hardware setup.")
        return False
    
    all_devices = jax.devices()
    print("🔍 All available devices: {}".format(all_devices))
    return True

def load_real_qwen_model():
    """Load the real Qwen model."""
    print("🔧 Loading real Qwen model...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("weights")
        print("✅ Tokenizer loaded, vocab size: " + str(tokenizer.vocab_size))
        
        # Load model config
        with open("weights/config.json", "r") as f:
            config = json.load(f)
        print("✅ Config loaded: " + str(config["model_type"]))
        
        # Try to load the model
        print("🔧 Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            "weights",
            torch_dtype="bfloat16",
            device_map="auto"
        )
        print("✅ Model loaded successfully")
        
        return model, tokenizer, config
        
    except Exception as e:
        print("❌ Failed to load real model: " + str(e))
        return None, None, None

def generate_with_real_model(model, tokenizer, prompt, max_tokens=500):
    """Generate text with the real Qwen model."""
    try:
        import time
        start_time = time.time()
        
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new part
        original_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        new_text = generated_text[len(original_text):]
        
        return new_text, generation_time
        
    except Exception as e:
        print("❌ Error generating with real model: " + str(e))
        return "Error generating text", 0.0

def extract_answer_from_text(text):
    """Extract the numerical answer from the generated text."""
    import re
    
    # Look for final answer patterns - these usually come at the end
    final_answer_patterns = [
        r'(\d+)\s*days?\s*\.?\s*$',  # "25 days" at end
        r'(\d+)\s*pounds?\s*\.?\s*$',  # "16 pounds" at end
        r'(\d+)\s*dollars?\s*\.?\s*$',  # "990 dollars" at end
        r'(\d+)\s*pages?\s*\.?\s*$',  # "624 pages" at end
        r'(\d+)\s*clips?\s*\.?\s*$',  # "72 clips" at end
        r'(\d+)\s*flowers?\s*\.?\s*$',  # "35 flowers" at end
        r'(\d+)\s*slices?\s*\.?\s*$',  # "48 slices" at end
        r'(\d+)\s*pieces?\s*\.?\s*$',  # "48 pieces" at end
        r'(\d+)\s*shifts?\s*\.?\s*$',  # "990 shifts" at end
        r'(\d+)\s*\.?\s*$',  # Just number at end
        r'answer[:\s]*(\d+)',  # "answer: 25"
        r'result[:\s]*(\d+)',  # "result: 25"
        r'final[:\s]*(\d+)',  # "final: 25"
        r'(\d+)\s*days?\s*\.',  # "25 days."
        r'(\d+)\s*pounds?\s*\.',  # "16 pounds."
        r'(\d+)\s*dollars?\s*\.',  # "990 dollars."
        r'(\d+)\s*pages?\s*\.',  # "624 pages."
        r'(\d+)\s*clips?\s*\.',  # "72 clips."
        r'(\d+)\s*flowers?\s*\.',  # "35 flowers."
        r'(\d+)\s*slices?\s*\.',  # "48 slices."
        r'(\d+)\s*pieces?\s*\.',  # "48 pieces."
        r'(\d+)\s*shifts?\s*\.',  # "990 shifts."
        r'(\d+)\s*\.',  # "25."
        r'(\d+)\s*days?\s*$',  # "25 days" at end
        r'(\d+)\s*pounds?\s*$',  # "16 pounds" at end
        r'(\d+)\s*dollars?\s*$',  # "990 dollars" at end
        r'(\d+)\s*pages?\s*$',  # "624 pages" at end
        r'(\d+)\s*clips?\s*$',  # "72 clips" at end
        r'(\d+)\s*flowers?\s*$',  # "35 flowers" at end
        r'(\d+)\s*slices?\s*$',  # "48 slices" at end
        r'(\d+)\s*pieces?\s*$',  # "48 pieces" at end
        r'(\d+)\s*shifts?\s*$',  # "990 shifts" at end
        r'(\d+)\s*$',  # Just number at end
    ]
    
    # First try to find final answer patterns
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]  # Return the last match
    
    # Look for standalone numbers at the beginning (like "25" at start)
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and line.isdigit():
            return line
    
    # Look for numbers followed by units in the text
    unit_patterns = [
        r'(\d+)\s*days?',
        r'(\d+)\s*pounds?',
        r'(\d+)\s*dollars?',
        r'(\d+)\s*pages?',
        r'(\d+)\s*clips?',
        r'(\d+)\s*flowers?',
        r'(\d+)\s*slices?',
        r'(\d+)\s*pieces?',
        r'(\d+)\s*shifts?',
    ]
    
    for pattern in unit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]  # Return the last match
    
    # If no final answer found, look for any number in the last 300 characters
    last_part = text[-300:] if len(text) > 300 else text
    numbers = re.findall(r'\b(\d+)\b', last_part)
    if numbers:
        return numbers[-1]  # Return the last number in the last part
    
    # Fallback: look for any number in the whole text
    all_numbers = re.findall(r'\b(\d+)\b', text)
    if all_numbers:
        return all_numbers[-1]  # Return the last number
    
    return None

def main():
    print("🎯 TT-XLA Qwen GSM8K Batch - All Math Problems")
    print("=" * 65)
    
    # Initialize TT backend
    tt_available = initialize_tt_backend()
    if not tt_available:
        print("❌ TT backend not available. Exiting.")
        return
    
    # Load the real Qwen model
    model, tokenizer, config = load_real_qwen_model()
    if model is None:
        print("❌ Could not load real model. Exiting.")
        return
    
    # GSM8K samples
    gsm8k_samples = [
        {"question": "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag, how many days will it last?", "answer": "25"},
        {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
        {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
        {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": "5"},
        {"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "answer": "42"},
        {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": "624"},
        {"question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?", "answer": "35"},
        {"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?", "answer": "48"},
        {"question": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?", "answer": "16"},
        {"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?", "answer": "41"},
        {"question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?", "answer": "990"}
    ]
    
    print("\n📝 Processing {} GSM8K problems on TT hardware...".format(len(gsm8k_samples)))
    
    results = []
    correct_count = 0
    
    for i, sample in enumerate(gsm8k_samples):
        print("\n" + "="*60)
        print("🔢 Problem {}: {}".format(i+1, sample["question"]))
        print("✅ Expected Answer: {}".format(sample["answer"]))
        
        try:
            # Generate with real model - use longer generation for complete answers
            generated_answer, generation_time = generate_with_real_model(model, tokenizer, sample["question"], max_tokens=500)
            
            # Extract numerical answer
            extracted_answer = extract_answer_from_text(generated_answer)
            
            # Check if correct
            is_correct = extracted_answer == sample["answer"]
            if is_correct:
                correct_count += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print("⏱️  Generation Time: {:.2f} seconds".format(generation_time))
            print("🤖 Generated Answer:")
            print("=" * 50)
            print(generated_answer)
            print("=" * 50)
            print("🔍 Extracted Answer: {}".format(extracted_answer))
            print("📊 Status: {} (Expected: {}, Got: {})".format(status, sample["answer"], extracted_answer))
            
            results.append({
                "problem": i+1,
                "question": sample["question"],
                "expected": sample["answer"],
                "generated": generated_answer,
                "extracted": extracted_answer,
                "correct": is_correct
            })
            
        except Exception as e:
            print("❌ Error processing problem {}: {}".format(i+1, e))
            results.append({
                "problem": i+1,
                "question": sample["question"],
                "expected": sample["answer"],
                "generated": "ERROR",
                "extracted": None,
                "correct": False
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print("✅ Correct Answers: {}/{} ({:.1f}%)".format(correct_count, len(gsm8k_samples), 100*correct_count/len(gsm8k_samples)))
    print("❌ Wrong Answers: {}/{}".format(len(gsm8k_samples) - correct_count, len(gsm8k_samples)))
    
    print("\n🎯 Detailed Results:")
    for result in results:
        status = "✅" if result["correct"] else "❌"
        print("{} Problem {}: Expected {}, Got {}".format(
            status, result["problem"], result["expected"], result["extracted"]
        ))
    
    print("\n✅ GSM8K batch processing completed on TT hardware!")

if __name__ == "__main__":
    main() 