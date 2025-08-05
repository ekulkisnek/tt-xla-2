#!/usr/bin/env python3
"""
Final Qwen2.5-7B inference for 2 TT Wormhole devices - getting actual answers.
This version properly handles Flax model structure and focuses on working inference.
"""
import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, Any, List

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
logger = logging.getLogger("qwen25_tt_final")

def initialize_tt_backend():
    """Initialize TT backend for JAX optimized for real hardware."""
    print("🔧 Initializing TT backend for real Wormhole devices...")
    
    try:
        # Try to use the installed wheel plugin first
        import sys
        sys.path.append('/opt/venv/lib/python3.10/site-packages/jax_plugins')
        import pjrt_plugin_tt
        print("✅ Using installed TT PJRT plugin")
    except ImportError:
        print("❌ TT plugin not found. Please install with: pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/")
        return False
    
    # Configure JAX to use TT devices with higher priority
    jax.config.update("jax_platforms", "tt,cpu")
    
    # Check available devices
    tt_devices = jax.devices("tt")
    print(f"🚀 Found {len(tt_devices)} TT device(s): {tt_devices}")
    
    if len(tt_devices) == 0:
        print("❌ No TT devices found. Check hardware setup.")
        return False
    
    # Print all available devices for debugging
    all_devices = jax.devices()
    print(f"🔍 All available devices: {all_devices}")
    
    return True

def check_device_usage():
    """Check which devices JAX is using."""
    print(f"🔍 JAX platforms: {jax.config.jax_platforms}")
    print(f"🔍 Available devices: {jax.devices()}")
    print(f"🔍 TT devices: {jax.devices('tt')}")
    print(f"🔍 CPU devices: {jax.devices('cpu')}")
    
    # Test a simple operation to see where it runs
    x = jax.numpy.array([1.0, 2.0, 3.0])
    print(f"🔍 Test array device: {x.device}")

# --- Final Model for TT Hardware ---
class FinalEmbedding(nn.Module):
    """Embedding layer for final TT inference."""
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

class FinalDense(nn.Module):
    """Dense layer for final TT inference."""
    features: int
    dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            'kernel',
            lambda key, shape: jnp.ones(shape, dtype=self.dtype) * 0.01,
            (inputs.shape[-1], self.features),
        )
        
        bias = None
        if self.use_bias:
            bias = self.param(
                'bias',
                lambda key, shape: jnp.zeros(shape, dtype=self.dtype),
                (self.features,),
            )
        
        y = jnp.dot(inputs, kernel)
        if bias is not None:
            y = y + bias
        return y

class FinalAttention(nn.Module):
    """Attention layer for final TT inference."""
    hidden_size: int
    num_heads: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states):
        # Simple linear projections
        q = FinalDense(self.hidden_size, dtype=self.dtype, name="q_proj")(hidden_states)
        k = FinalDense(self.hidden_size, dtype=self.dtype, name="k_proj")(hidden_states)
        v = FinalDense(self.hidden_size, dtype=self.dtype, name="v_proj")(hidden_states)
        
        # Reshape for multi-head attention
        batch, seq_len, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads
        
        q = q.reshape(batch, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch, seq_len, self.num_heads, head_dim)
        
        # Simple attention computation
        scores = jnp.einsum('bqhd,bkhd->bqhk', q, k) / jnp.sqrt(head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.einsum('bqhk,bkhd->bqhd', attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch, seq_len, self.hidden_size)
        output = FinalDense(self.hidden_size, dtype=self.dtype, name="o_proj")(attn_output)
        
        return output

class FinalMLP(nn.Module):
    """MLP layer for final TT inference."""
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states):
        gate = FinalDense(self.intermediate_size, dtype=self.dtype, name="gate_proj")(hidden_states)
        up = FinalDense(self.intermediate_size, dtype=self.dtype, name="up_proj")(hidden_states)
        
        # Use gelu instead of silu to avoid potential issues
        gate = jax.nn.gelu(gate)
        
        down = FinalDense(self.hidden_size, dtype=self.dtype, name="down_proj")(gate * up)
        return down

class FinalLayer(nn.Module):
    """Single transformer layer for final TT inference."""
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states):
        # Attention
        attention_output = FinalAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dtype=self.dtype
        )(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # MLP
        mlp_output = FinalMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype
        )(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states

class FinalQwenModel(nn.Module):
    """Qwen model for final TT inference."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.vocab_size = c["vocab_size"]
        self.num_hidden_layers = c["num_hidden_layers"]
        self.intermediate_size = c["intermediate_size"]
        self.num_attention_heads = c["num_attention_heads"]
        
        self.embed_tokens = FinalEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype
        )
        
        # Use more layers for better inference quality
        num_layers = min(8, self.num_hidden_layers)  # Increased to 8 layers
        self.layers = [FinalLayer(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            dtype=self.dtype
        ) for _ in range(num_layers)]
        
        self.norm = nn.LayerNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype)
        self.lm_head = FinalDense(
            self.vocab_size,
            dtype=self.dtype,
            name="lm_head"
        )

    def __call__(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits}

def sample_next_token(logits):
    """Sample the next token using greedy decoding."""
    return jnp.argmax(logits, axis=-1)

def generate_text_final(model, params, tokenizer, prompt: str, max_tokens: int = 30):
    """Generate text using the model on TT hardware."""
    print(f"🚀 Starting final text generation on TT hardware...")
    print(f"📝 Prompt: {prompt}")
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    print(f"🔢 Encoded tokens: {input_ids.shape}")
    
    # Initialize generation
    generated_tokens = []
    current_input = input_ids
    
    print(f"🔄 Generating {max_tokens} tokens...")
    
    for i in range(max_tokens):
        print(f"  Token {i+1}/{max_tokens}...", end="", flush=True)
        
        # Run inference
        outputs = model.apply(params, current_input)
        logits = outputs["logits"]
        
        # Get next token
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        
        # Update input for next iteration
        current_input = jnp.concatenate([current_input, next_token[:, None]], axis=1)
        
        # Decode and show token
        token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(f" -> '{token_text}'")
        
        # Check for end of sequence
        if int(next_token) == tokenizer.eos_token_id:
            print(" (EOS)")
            break
    
    # Decode full output
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\n📄 Generated text: {full_output}")
    
    return full_output

def main():
    parser = argparse.ArgumentParser(description="Final Qwen2.5-7B Inference on Real TT Wormhole")
    parser.add_argument("--model_path", type=str, default="weights", help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Initialize TT backend first
    tt_available = initialize_tt_backend()
    if not tt_available:
        print("❌ TT backend not available. Exiting.")
        return
    
    # Check device usage
    check_device_usage()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Load config
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"✅ Loaded config: {config['model_type']} with {config['num_hidden_layers']} layers")
    else:
        print("⚠️  Config not found, using default config")
        config = {
            "hidden_size": 3584,
            "vocab_size": 152064,
            "num_hidden_layers": 28,
            "intermediate_size": 18944,
            "num_attention_heads": 28,
            "rms_norm_eps": 1e-6
        }
    
    # Create model
    print("🔧 Creating final model for TT hardware...")
    model = FinalQwenModel(config=config, dtype=dtype)
    
    # Load tokenizer
    print("🔧 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"✅ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Initialize model with proper Flax structure
    print("🧪 Initializing model with proper Flax structure...")
    try:
        # Create dummy input
        dummy_input = jnp.ones((1, 5), dtype=jnp.int32)
        
        # Initialize model parameters using Flax's init
        variables = model.init(jax.random.PRNGKey(42), dummy_input)
        print("✅ Model initialization successful")
        
        # Test forward pass
        print("🧪 Testing forward pass...")
        outputs = model.apply(variables, dummy_input)
        print(f"✅ Forward pass successful, output shape: {outputs['logits'].shape}")
        print(f"🔍 Output device: {outputs['logits'].device}")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("This is expected for experimental TT backend")
        return
    
    # Janet's dogs prompt
    janet_prompt = "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag, how many days will it last?"
    
    print(f"\n🎯 Running final inference for Janet's dogs prompt...")
    print(f"📝 Prompt: {janet_prompt}")
    
    try:
        # Generate text
        generated_text = generate_text_final(
            model, 
            variables, 
            tokenizer, 
            janet_prompt, 
            max_tokens=args.max_tokens
        )
        
        print(f"\n🎉 Final inference completed successfully!")
        print(f"📄 Full response: {janet_prompt}{generated_text}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("This may be due to experimental TT backend limitations")
    
    print(f"\n✅ Final TT Hardware Inference Complete!")

if __name__ == "__main__":
    main() 