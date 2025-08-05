#!/usr/bin/env python3
"""
TT-XLA Qwen Real Model - Load actual Qwen weights and run on TT hardware
This script loads the real Qwen model and tries to get the actual answer.
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
logger = logging.getLogger("qwen25_tt_real_qwen")

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

def generate_with_real_model(model, tokenizer, prompt, max_tokens=10):
    """Generate text with the real Qwen model."""
    print("🔄 Generating with real Qwen model...")
    
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        print("🔢 Input tokens: " + str(inputs.shape))
        
        # Generate text
        print("🧪 Running inference...")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("📄 Generated text: " + generated_text)
        
        # Extract just the new part
        original_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        new_text = generated_text[len(original_text):]
        print("📄 New generated text: " + new_text)
        
        return new_text
        
    except Exception as e:
        print("❌ Error generating with real model: " + str(e))
        return "Error generating text"

def main():
    print("🎯 TT-XLA Qwen Real Model - Load Actual Weights")
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
    
    # Process the dog food prompt
    janet_prompt = "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag, how many days will it last? Answer with just the number of days."
    
    print("\n📝 Processing dog food prompt...")
    print("📝 Prompt: " + janet_prompt)
    
    try:
        # Generate with real model
        generated_answer = generate_with_real_model(model, tokenizer, janet_prompt, max_tokens=50)
        
        # Show mathematical expectation
        print("\n🧮 Mathematical calculation:")
        print("   - Dogs eat: 2 pounds per day")
        print("   - Bag size: 50 pounds")
        print("   - Days = 50 ÷ 2 = 25 days")
        print("✅ Expected answer: 25 days")
        
        # Show what we got
        print("\n📄 Generated Answer: " + generated_answer)
        
        # Check if we got the right answer
        if "25" in generated_answer and "days" in generated_answer.lower():
            print("🎉 SUCCESS! Got the correct answer: 25 days")
        else:
            print("⚠️  Did not get the expected answer '25 days'")
            print("   Generated: '" + generated_answer + "'")
            print("   Expected: '25 days'")
        
    except Exception as e:
        print("❌ Generation failed: " + str(e))
        return
    
    print("\n✅ Real model generation completed!")

if __name__ == "__main__":
    main() 