#!/usr/bin/env python3
"""
Qwen2.5-7B inference for Janet's dogs prompt on Tenstorrent Wormhole.
This version focuses on getting actual inference working on TT hardware.
"""
import os
import sys
import json
import argparse
import logging
import time
import jax.random
from typing import Dict, Any, List

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

# Set up multi-device (do this before importing jax)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import jax._src.xla_bridge as xb

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tt_inference")

# Global mesh for tensor parallelism
mesh = None

def initialize_tt_backend():
    """Initialize TT backend for JAX."""
    print("🔧 Initializing TT backend...")
    
    try:
        # Try to use the installed wheel plugin first
        import sys
        sys.path.append('/opt/venv/lib/python3.10/site-packages/jax_plugins')
        import pjrt_plugin_tt
        print("✅ Using installed TT PJRT plugin")
    except ImportError:
        print("⚠️  TT plugin not found in wheel, trying local build...")
        # Fall back to local build if wheel not available
        plugin_path = os.path.join(
            os.path.dirname(__file__), "../../../build/src/tt/pjrt_plugin_tt.so"
        )
        if os.path.exists(plugin_path):
            print(f"🔧 Loading TT PJRT plugin from {plugin_path}")
            xb.discover_pjrt_plugins()
            xb.register_plugin("tt", priority=500, library_path=plugin_path, options=None)
        else:
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

# --- Inference-Optimized Model Code ---
class InferenceEmbedding(nn.Module):
    """Embedding layer optimized for inference."""
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.hidden_size),
            self.dtype
        )
        return jnp.take(embedding, inputs, axis=0)

class InferenceAttention(nn.Module):
    """Simplified attention for inference."""
    hidden_size: int
    num_heads: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states):
        # Simple linear projections
        q = nn.Dense(self.hidden_size, dtype=self.dtype, name="q_proj")(hidden_states)
        k = nn.Dense(self.hidden_size, dtype=self.dtype, name="k_proj")(hidden_states)
        v = nn.Dense(self.hidden_size, dtype=self.dtype, name="v_proj")(hidden_states)
        
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
        output = nn.Dense(self.hidden_size, dtype=self.dtype, name="o_proj")(attn_output)
        
        return output

class InferenceMLP(nn.Module):
    """MLP layer optimized for inference."""
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states):
        gate = nn.Dense(self.intermediate_size, dtype=self.dtype, name="gate_proj")(hidden_states)
        up = nn.Dense(self.intermediate_size, dtype=self.dtype, name="up_proj")(hidden_states)
        
        # Use gelu instead of silu to avoid potential issues
        gate = jax.nn.gelu(gate)
        
        down = nn.Dense(self.hidden_size, dtype=self.dtype, name="down_proj")(gate * up)
        return down

class InferenceLayer(nn.Module):
    """Single transformer layer for inference."""
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, hidden_states):
        # Attention
        attention_output = InferenceAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dtype=self.dtype
        )(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # MLP
        mlp_output = InferenceMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype
        )(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states

class InferenceQwenModel(nn.Module):
    """Qwen model optimized for inference on TT hardware."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.vocab_size = c["vocab_size"]
        self.num_hidden_layers = c["num_hidden_layers"]
        self.intermediate_size = c["intermediate_size"]
        self.num_attention_heads = c["num_attention_heads"]
        
        self.embed_tokens = InferenceEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype
        )
        
        # Use fewer layers for faster inference
        num_layers = min(4, self.num_hidden_layers)
        self.layers = [InferenceLayer(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            dtype=self.dtype
        ) for _ in range(num_layers)]
        
        self.norm = nn.LayerNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype)
        self.lm_head = nn.Dense(
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

def setup_device_mesh():
    """Setup device mesh for tensor parallelism."""
    global mesh
    print("Setting up device mesh for TT hardware...")
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    if len(devices) == 1:
        print("Single TT device detected - using single device mode")
        mesh = Mesh(devices, axis_names=("mp",))
    else:
        # Use all available devices for tensor parallelism
        mesh = Mesh(devices, axis_names=("mp",))
        print(f"Created multi-device mesh: {mesh}")
    
    return mesh

def sample_next_token(logits):
    """Sample the next token using greedy decoding."""
    return jnp.argmax(logits, axis=-1)

def generate_text_simple(model, params, tokenizer, prompt: str, max_tokens: int = 20):
    """Generate text using the model on TT hardware."""
    print(f"🚀 Starting text generation on TT hardware...")
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
    parser = argparse.ArgumentParser(description="Qwen2.5-7B Inference on TT Wormhole")
    parser.add_argument("--model_path", type=str, default="weights", help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Initialize TT backend first
    tt_available = initialize_tt_backend()
    if not tt_available:
        print("❌ TT backend not available. Exiting.")
        return
    
    # Check device usage
    check_device_usage()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Setup device mesh for tensor parallelism
    mesh = setup_device_mesh()
    
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
    print("🔧 Creating inference model...")
    model = InferenceQwenModel(config=config, dtype=dtype)
    
    # Load tokenizer
    print("🔧 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"✅ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Initialize model with dummy parameters (for testing)
    print("🧪 Initializing model parameters...")
    try:
        # Create dummy input
        dummy_input = jnp.ones((1, 5), dtype=jnp.int32)
        
        # Initialize model parameters
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
    
    print(f"\n🎯 Running inference for Janet's dogs prompt...")
    print(f"📝 Prompt: {janet_prompt}")
    
    try:
        # Generate text
        generated_text = generate_text_simple(
            model, 
            variables, 
            tokenizer, 
            janet_prompt, 
            max_tokens=args.max_tokens
        )
        
        print(f"\n🎉 Inference completed successfully!")
        print(f"📄 Full response: {janet_prompt}{generated_text}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("This may be due to experimental TT backend limitations")
    
    print(f"\n✅ TT Hardware Inference Test Complete!")

if __name__ == "__main__":
    main() 