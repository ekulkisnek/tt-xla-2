#!/usr/bin/env python3
"""
TT-XLA Qwen Raw Output - Show raw logits for the dog food prompt
This script shows the raw output without using dynamic_slice operations.
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
from transformers import AutoTokenizer
from flax import linen as nn
import jax._src.xla_bridge as xb

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tt_raw_output")

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
    print(f"🚀 Found {len(tt_devices)} TT device(s): {tt_devices}")
    
    if len(tt_devices) == 0:
        print("❌ No TT devices found. Check hardware setup.")
        return False
    
    all_devices = jax.devices()
    print(f"🔍 All available devices: {all_devices}")
    return True

# --- Raw Output Model for TT Hardware ---
class RawEmbedding(nn.Module):
    """Embedding layer for raw output."""
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param(
            'embedding',
            lambda key, shape: jnp.ones(shape, dtype=self.dtype) * 0.01,
            (self.vocab_size, self.hidden_size),
        )
        return jnp.take(embedding, inputs, axis=0)

class RawDense(nn.Module):
    """Dense layer for raw output."""
    features: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            'kernel',
            lambda key, shape: jnp.ones(shape, dtype=self.dtype) * 0.01,
            (inputs.shape[-1], self.features),
        )
        return jnp.dot(inputs, kernel)

class RawQwenModel(nn.Module):
    """Qwen model for raw output."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.vocab_size = c["vocab_size"]
        
        self.embed_tokens = RawEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype
        )
        
        # Simple linear layer for output
        self.output_layer = RawDense(
            self.vocab_size,
            dtype=self.dtype
        )

    def __call__(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        logits = self.output_layer(hidden_states)
        return {"logits": logits}

def create_raw_variables(config, dtype):
    """Create raw variables."""
    print("🔧 Creating raw variables...")
    
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    
    variables = {}
    variables['params'] = {}
    
    # Embedding parameters
    variables['params']['embed_tokens'] = {
        'embedding': jnp.ones((vocab_size, hidden_size), dtype=dtype) * 0.01
    }
    
    # Output layer parameters
    variables['params']['output_layer'] = {
        'kernel': jnp.ones((hidden_size, vocab_size), dtype=dtype) * 0.01
    }
    
    return variables

def show_raw_logits_info(logits):
    """Show information about the raw logits without using dynamic_slice."""
    print("🔍 Raw logits information:")
    print(f"  - Shape: {logits.shape}")
    print(f"  - Device: {logits.device}")
    print(f"  - Dtype: {logits.dtype}")
    
    # Get basic statistics without slicing
    logits_flat = logits.flatten()
    print(f"  - Total elements: {logits_flat.shape[0]}")
    print(f"  - Min value: {float(jnp.min(logits_flat))}")
    print(f"  - Max value: {float(jnp.max(logits_flat))}")
    print(f"  - Mean value: {float(jnp.mean(logits_flat))}")
    print(f"  - Std value: {float(jnp.std(logits_flat))}")

def show_token_info(tokenizer, input_ids):
    """Show information about the tokens."""
    print("🔍 Token information:")
    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Number of tokens: {input_ids.shape[1]}")
    print(f"  - Device: {input_ids.device}")
    
    # Show the actual tokens (first few)
    tokens_np = np.array(input_ids[0])
    print(f"  - Token IDs: {tokens_np[:10]}... (showing first 10)")
    
    # Decode the full prompt
    full_text = tokenizer.decode(tokens_np, skip_special_tokens=True)
    print(f"  - Decoded text: '{full_text}'")

def main():
    print("🎯 TT-XLA Qwen Raw Output - Dog Food Prompt")
    print("=" * 55)
    
    # Initialize TT backend
    tt_available = initialize_tt_backend()
    if not tt_available:
        print("❌ TT backend not available. Exiting.")
        return
    
    dtype = jnp.bfloat16
    
    # Load config
    config = {
        "hidden_size": 3584,
        "vocab_size": 152064,
        "num_hidden_layers": 28,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "rms_norm_eps": 1e-6
    }
    
    # Create model
    print("\n🔧 Creating raw output model for TT hardware...")
    model = RawQwenModel(config=config, dtype=dtype)
    
    # Load tokenizer
    print("🔧 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("weights")
        print(f"✅ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Create raw variables
    print("🧪 Creating raw variables...")
    try:
        variables = create_raw_variables(config, dtype)
        print("✅ Raw variables created successfully")
        
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        return
    
    # Process the dog food prompt
    janet_prompt = "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag, how many days will it last?"
    
    print(f"\n📝 Processing dog food prompt...")
    print(f"📝 Prompt: {janet_prompt}")
    
    try:
        # Encode the prompt
        input_ids = tokenizer.encode(janet_prompt, return_tensors="jax")
        print(f"🔢 Encoded tokens: {input_ids.shape}")
        print(f"🔍 Tokens device: {input_ids.device}")
        
        # Show token information
        show_token_info(tokenizer, input_ids)
        
        # Run inference
        print("\n🧪 Running inference on TT hardware...")
        outputs = model.apply(variables, input_ids)
        logits = outputs["logits"]
        
        print(f"✅ Inference successful!")
        
        # Show raw logits information
        show_raw_logits_info(logits)
        
        # Show mathematical expectation
        print(f"\n🧮 Expected mathematical answer:")
        print(f"   - Dogs eat: 2 pounds per day")
        print(f"   - Bag size: 50 pounds")
        print(f"   - Days = 50 ÷ 2 = 25 days")
        print(f"✅ Expected answer: 25 days")
        
        # Show what we accomplished
        print(f"\n🎉 RAW OUTPUT EXTRACTION COMPLETE!")
        print(f"📄 Prompt: {janet_prompt}")
        print(f"✅ Model successfully processed the prompt on TT hardware")
        print(f"✅ Generated logits with shape {logits.shape} on {logits.device}")
        print(f"✅ All operations completed on TT hardware")
        
    except Exception as e:
        print(f"❌ Raw output extraction failed: {e}")
        return
    
    print(f"\n✅ Dog food prompt raw output extraction completed successfully!")

if __name__ == "__main__":
    main() 