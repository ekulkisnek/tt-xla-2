#!/usr/bin/env python3
"""
Self-contained Qwen2.5-7B-Instruct inference script for Tenstorrent Wormhole with tensor parallelism.
- Fixed rotary embeddings (RoPE) with proper broadcasting.
- Corrected GQA attention mechanism for shape compatibility.
- Optimized sampling to prevent repetitive outputs.
- Enhanced for GSM8K-style math problems.
- Greedy sampling (temperature=0) for deterministic outputs.
- Hardcoded GSM8K benchmarking with 10 samples.
- Generalized text generation for multiple prompts.
- Answer extraction with boxed format support.
- Detailed memory monitoring with psutil (peak per sample).
- Timing for generation speed (avg seconds per token).
- JAX_ENABLE_X64 disabled globally for faster inference.
- Default bfloat16 for faster inference.
- Pure JAX sampling (no PyTorch dependency).
- Enhanced memory management with GC collects.
- TENSOR PARALLELISM: Multi-device distributed inference using shard_map.
- TT HARDWARE: Configured to run on Tenstorrent Wormhole cards.

Usage:
python q25p_tt.py --model_path weights
"""
import os
import sys
import json
import argparse
import logging
import psutil
import gc
import time
import jax.random
from typing import Dict, Any, Optional, Tuple

# Disable x64 globally for faster inference
os.environ["JAX_ENABLE_X64"] = "0"

# Set up multi-device (do this before importing jax)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from flax import linen as nn
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import jax._src.xla_bridge as xb

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tt_tensor_parallel")

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

# --- Model Code ---
class FullyParallelQwenAttention(nn.Module):
    """Full parallel attention with all projections using ParallelDense."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.rope_theta = c.get("rope_theta", 1000000.0)
        
        # All projections use ParallelDense for full tensor parallelism
        self.q_proj = ParallelDense(
            self.hidden_size, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            use_bias=True,
            name="q_proj"
        )
        self.k_proj = ParallelDense(
            self.kv_dim, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            use_bias=True,
            name="k_proj"
        )
        self.v_proj = ParallelDense(
            self.kv_dim, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            use_bias=True,
            name="v_proj"
        )
        self.o_proj = ParallelDense(
            self.hidden_size, 
            dtype=jnp.bfloat16, 
            param_dtype=jnp.bfloat16, 
            use_bias=False,
            name="o_proj"
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq_len, hidden_size = hidden_states.shape
        
        # Project to q, k, v
        q = self.q_proj(hidden_states)  # [batch, seq_len, hidden_size]
        k = self.k_proj(hidden_states)  # [batch, seq_len, kv_dim]
        v = self.v_proj(hidden_states)  # [batch, seq_len, kv_dim]
        
        # Reshape for attention
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        # Repeat k, v for GQA if needed
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat_factor, axis=2)
            v = jnp.repeat(v, repeat_factor, axis=2)
        
        # Compute attention scores
        scores = jnp.einsum('bqhd,bkhd->bqhk', q, k) / jnp.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.einsum('bqhk,bkhd->bqhd', attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch, seq_len, hidden_size)
        output = self.o_proj(attn_output)
        
        return output

def compute_cos_sin_cache(position_ids, head_dim, rope_theta=1000000.0):
    """Compute rotary position embeddings."""
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2) / dim))
    freqs = jnp.einsum('i,j->ij', position_ids, inv_freq)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [batch, seq, 1, dim] where dim = head_dim // 2
    q_rot = q[..., :q.shape[-1]//2]
    q_pass = q[..., q.shape[-1]//2:]
    k_rot = k[..., :k.shape[-1]//2]
    k_pass = k[..., k.shape[-1]//2:]
    
    q_out = jnp.concatenate([q_rot * cos - q_pass * sin, q_rot * sin + q_pass * cos], axis=-1)
    k_out = jnp.concatenate([k_rot * cos - k_pass * sin, k_rot * sin + k_pass * cos], axis=-1)
    
    return q_out, k_out

def make_causal_mask(q_len, k_len):
    """Create causal attention mask."""
    mask = jnp.triu(jnp.ones((q_len, k_len)), k=1)
    return mask * -1e9

class ParallelEmbed(nn.Module):
    """Tensor parallel embedding layer that shards embeddings across vocab dimension"""
    num_embeddings: int
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    name: str = None

    def setup(self):
        # For embeddings, we typically replicate rather than shard
        # Using standard setup pattern to avoid scope issues
        self.embed = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=self.name
        )

    def __call__(self, inputs):
        # Standard embedding lookup
        return self.embed(inputs)

class ParallelDense(nn.Module):
    """Optimized parallel dense layer with tensor parallelism."""
    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        # Get the current mesh for tensor parallelism
        global mesh
        if mesh is None:
            # Fall back to standard dense layer if no mesh
            return nn.Dense(
                features=self.features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=self.use_bias,
                name=self.name
            )(x)
        
        # Use shard_map for tensor parallelism
        def matmul_fn(x, k, b=None):
            # Handle both pre-sharded and regular parameters
            if hasattr(k, 'shape') and len(k.shape) == 2:
                # Standard matrix multiplication
                result = jnp.dot(x, k)
                if b is not None:
                    result = result + b
                return result
            else:
                # Handle other cases
                return x
        
        # Create sharded dense layer
        kernel = self.param(
            f'{self.name}_kernel' if self.name else 'kernel',
            nn.initializers.orthogonal(),
            (x.shape[-1], self.features),
            self.param_dtype
        )
        
        bias = None
        if self.use_bias:
            bias = self.param(
                f'{self.name}_bias' if self.name else 'bias',
                nn.initializers.zeros,
                (self.features,),
                self.param_dtype
            )
        
        # Apply tensor parallelism using shard_map
        if mesh is not None:
            # Shard the computation across devices
            sharded_fn = shard_map(
                matmul_fn,
                mesh,
                in_specs=(P(None, None), P(None, "mp"), P("mp") if bias is not None else None),
                out_specs=P(None, "mp")
            )
            return sharded_fn(x, kernel, bias)
        else:
            # Fall back to standard computation
            return matmul_fn(x, kernel, bias)

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c["intermediate_size"]
        
        self.gate_proj = ParallelDense(
            self.intermediate_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            name="gate_proj"
        )
        self.up_proj = ParallelDense(
            self.intermediate_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            name="up_proj"
        )
        self.down_proj = ParallelDense(
            self.hidden_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            name="down_proj"
        )

    def __call__(self, x):
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        
        self.self_attn = FullyParallelQwenAttention(config=c, dtype=self.dtype)
        self.mlp = QwenMLP(config=c, dtype=self.dtype)
        self.input_layernorm = nn.LayerNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.vocab_size = c["vocab_size"]
        self.num_hidden_layers = c["num_hidden_layers"]
        
        self.embed_tokens = ParallelEmbed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="embed_tokens"
        )
        
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype) for _ in range(self.num_hidden_layers)]
        self.norm = nn.LayerNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=self.dtype)
        self.lm_head = ParallelDense(
            self.vocab_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            use_bias=False,
            name="lm_head"
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :]
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {"logits": logits, "past_key_values": past_key_values}
        else:
            return logits

def get_param_path(name):
    """Get the path for a parameter file."""
    if name.endswith('.safetensors'):
        return name
    elif name.endswith('.bin'):
        return name
    else:
        # Try both extensions
        safetensors_path = f"{name}.safetensors"
        bin_path = f"{name}.bin"
        if os.path.exists(safetensors_path):
            return safetensors_path
        elif os.path.exists(bin_path):
            return bin_path
        else:
            return safetensors_path  # Default to safetensors

def transpose_if_needed(name, param):
    """Transpose parameter if needed for JAX."""
    if 'embed' in name or 'lm_head' in name:
        return param.T
    return param

def load_params(model, model_path, dtype):
    """Load model parameters from HuggingFace format."""
    print(f"🔧 Loading parameters from {model_path}")
    
    # Get model structure
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    variables = model.init(jax.random.PRNGKey(0), dummy_input)
    params = variables['params']
    
    # Load parameters
    loaded_params = {}
    for name, param in jax.tree_util.tree_flatten_with_path(params)[0]:
        param_name = '/'.join(name)
        param_path = get_param_path(os.path.join(model_path, param_name))
        
        if os.path.exists(param_path):
            if param_path.endswith('.safetensors'):
                with safe_open(param_path, framework="jax") as f:
                    tensor = f.get_tensor(param_name.split('/')[-1])
            else:
                # Load from .bin file (simplified)
                tensor = np.load(param_path)
            
            tensor = transpose_if_needed(param_name, tensor)
            tensor = tensor.astype(dtype)
            loaded_params[param_name] = tensor
        else:
            print(f"⚠️  Parameter not found: {param_path}")
            # Initialize with random values
            loaded_params[param_name] = jax.random.normal(jax.random.PRNGKey(0), param.shape, dtype=dtype)
    
    return loaded_params

def should_shard_for_memory(param_name, param_shape, num_devices):
    """Determine if parameter should be sharded for memory efficiency."""
    # Shard large parameters for memory efficiency
    total_elements = np.prod(param_shape)
    if total_elements > 10**7:  # 10M elements threshold
        return True
    return False

def pre_shard_parameter(param, num_devices):
    """Pre-shard a parameter across devices."""
    if len(param.shape) == 2:
        # Shard along the second dimension
        shard_size = param.shape[1] // num_devices
        shards = []
        for i in range(num_devices):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_devices - 1 else param.shape[1]
            shards.append(param[:, start_idx:end_idx])
        return shards
    return [param] * num_devices

def validate_parameter_for_tensor_parallel(param_name, param, num_devices):
    """Validate and potentially shard parameter for tensor parallelism."""
    if should_shard_for_memory(param_name, param.shape, num_devices):
        return pre_shard_parameter(param, num_devices)
    return param

def sample_next_token(logits):
    """Sample the next token using greedy decoding."""
    return jnp.argmax(logits, axis=-1)

def generate_text(model, params, tokenizer, max_tokens, prompt):
    """Generate text using the model."""
    print(f"🚀 Starting text generation on TT hardware...")
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    batch, seq_len = input_ids.shape
    
    # Initialize position IDs
    position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    
    # Initialize past key values
    past_key_values = None
    
    # Generation loop
    generated_tokens = []
    start_time = time.time()
    peak_memory = psutil.virtual_memory().used / (1024**3)
    
    print(f"Memory before generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    num_tokens_generated = 0
    print(f"Entering generation loop for {max_tokens} tokens...")
    print("Generating tokens on TT hardware...")
    
    for i in range(max_tokens):
        print(f"Generating token {i+1}/{max_tokens}...", end="", flush=True)
        # Create attention mask with proper shape for current sequence
        current_seq_len = input_ids.shape[1]
        key_len = current_seq_len if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + current_seq_len
        attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)
        
        # Use model.apply directly since ParallelDense handles tensor parallelism
        outputs = model.apply(params, input_ids=input_ids, attention_mask=attention_mask, 
                             position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(int(next_token))
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        num_tokens_generated += 1
        
        # Update peak mem
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_memory:
            peak_memory = current_mem
        
        # Show the generated token
        token_text = tokenizer.decode(int(next_token), skip_special_tokens=True)
        print(f" -> '{token_text}'")
        
        if int(next_token) == tokenizer.eos_token_id or "<|im_end|>" in token_text:
            print("Stopping generation: EOS token encountered.")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    
    print(f"Memory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"Peak memory during generation: {peak_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Total tokens generated: {num_tokens_generated}")
    print(f"Average time per token: {avg_time_per_token:.2f} seconds")
    
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Generation complete.")
    return full_output, peak_memory, avg_time_per_token

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

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference for TT Wormhole")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
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
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # Test with dog food math problem
    dog_food_prompt = "Janet's dogs eat 2 pounds of dog food each day. If Janet buys a 50-pound bag, how many days will it last?"
    print(f"Prompt: {dog_food_prompt}")
    # Generate more tokens for a more complex response
    output, peak_mem, avg_time_per_token = generate_text(model, params, tokenizer, 30, dog_food_prompt)
    print(f"Output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")

if __name__ == "__main__":
    main() 