#!/usr/bin/env python3
"""
Real Tensor Parallelism for Qwen 2.5-7B using JAX multi-device simulation
Based on Flax GSPMD approach: https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html#setup
"""

import os
# CRITICAL: Set XLA_FLAGS to create multiple JAX devices BEFORE importing JAX
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import sys
import time
import json
import gc
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open

# JAX sharding imports
from jax.sharding import NamedSharding, PartitionSpec, Mesh
from jax.experimental import mesh_utils

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("real_tp_qwen25")

def setup_devices():
    """Setup and display JAX devices for tensor parallelism"""
    devices = jax.devices()
    print(f'🔥 JAX Devices Available: {len(devices)}')
    for i, device in enumerate(devices):
        print(f'  Device {i}: {device}')
    
    # Create a mesh for tensor parallelism (e.g., 2x4 = 8 devices)
    if len(devices) >= 8:
        mesh_devices = np.array(devices[:8]).reshape(2, 4)
        mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
    elif len(devices) >= 4:
        mesh_devices = np.array(devices[:4]).reshape(1, 4)
        mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
    elif len(devices) >= 2:
        mesh_devices = np.array(devices[:2]).reshape(1, 2)
        mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
    else:
        # Single device fallback
        mesh_devices = np.array([devices[0]]).reshape(1, 1)
        mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
    
    print(f'🚀 Mesh Configuration: {mesh}')
    return mesh

class TensorParallelDense(nn.Module):
    """Real tensor parallel dense layer with proper sharding"""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    shard_axes: Tuple[Optional[str], Optional[str]] = (None, None)  # (input_shard, output_shard)
    reduce_scatter: bool = False
    
    @nn.compact
    def __call__(self, inputs):
        # Initialize kernel with proper sharding annotation
        kernel_init = nn.initializers.lecun_normal()
        kernel = self.param('kernel', kernel_init, (inputs.shape[-1], self.features), self.param_dtype)
        
        # Apply sharding constraint to kernel
        if self.shard_axes != (None, None):
            input_shard, output_shard = self.shard_axes
            kernel_spec = PartitionSpec(input_shard, output_shard)
            kernel = jax.lax.with_sharding_constraint(kernel, kernel_spec)
        
        # Perform matrix multiplication
        y = jnp.dot(inputs, kernel)
        
        # Handle bias
        if self.use_bias:
            bias_init = nn.initializers.zeros
            bias = self.param('bias', bias_init, (self.features,), self.param_dtype)
            if self.shard_axes[1] is not None:  # Shard bias same as output
                bias_spec = PartitionSpec(self.shard_axes[1])
                bias = jax.lax.with_sharding_constraint(bias, bias_spec)
            y = y + bias
        
        # Apply reduce-scatter (all-reduce across model parallel dimension)
        if self.reduce_scatter:
            # Only perform psum if we're in a pmapped/mesh context with the 'model' axis
            try:
                y = jax.lax.psum(y, axis_name='model')
            except (NameError, ValueError):
                # During init, the mesh context might not be available
                # In this case, just pass through (the actual computation will happen later)
                pass
        
        return y

# === Copy model structure from working q25_jax.py ===

def make_causal_mask(q_len, k_len):
    """Create causal mask for different query and key lengths."""
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return (i < j - (k_len - q_len)) * -1e9

def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    """Compute RoPE cos/sin cache"""
    pos = np.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, dtype=np.float32) / dim))
    freqs = np.einsum('bi,j->bij', pos, inv_freq)
    
    cos = jnp.array(np.cos(freqs))
    sin = jnp.array(np.sin(freqs))
    cos = jnp.repeat(cos, 2, axis=-1)
    sin = jnp.repeat(sin, 2, axis=-1)
    
    return cos, sin

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings"""
    def _rope(x, cos, sin):
        cos = cos[..., None, :]
        sin = sin[..., None, :]
        return (x * cos) + (rotate_half(x) * sin)
    return _rope(q, cos, sin), _rope(k, cos, sin)

class QwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = c.get("head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        # Use tensor parallel layers with proper sharding
        self.q_proj = TensorParallelDense(
            features=self.hidden_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model")  # Shard output features across model parallel
        )
        self.k_proj = TensorParallelDense(
            features=self.kv_dim, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model")  # Shard output features across model parallel
        )
        self.v_proj = TensorParallelDense(
            features=self.kv_dim, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model")  # Shard output features across model parallel
        )
        self.o_proj = TensorParallelDense(
            features=self.hidden_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=("model", None),  # Shard input features across model parallel
            reduce_scatter=True  # All-reduce the output
        )
        
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.max_position_embeddings = c.get("max_position_embeddings", 4096)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        
        # Project current hidden states (these will be sharded)
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            if cos is None or sin is None:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Handle past key-value cache (same as working version)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            
            # Ensure past cache is in the same format as current k,v: [batch, seq, num_kv_heads, head_dim]
            if past_k.shape[1] == self.num_kv_heads and past_k.shape[2] != self.num_kv_heads:
                # Past cache is in transposed format [batch, num_kv_heads, seq, head_dim], transpose it back
                past_k = jnp.transpose(past_k, (0,2,1,3))
                past_v = jnp.transpose(past_v, (0,2,1,3))
            
            # Convert past cache to KV head format if needed
            if past_k.shape[2] == self.num_heads:  # If past cache is in query head format
                # Reshape to group query heads into KV heads
                past_k = past_k.reshape(batch, -1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                past_k = jnp.mean(past_k, axis=3)  # Average over query heads per KV head
                past_v = past_v.reshape(batch, -1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                past_v = jnp.mean(past_v, axis=3)
            elif past_k.shape[0] == 0 or past_k.shape[1] == 0:  # Empty cache
                past_k = jnp.zeros((batch, 0, self.num_kv_heads, self.head_dim), dtype=past_k.dtype)
                past_v = jnp.zeros((batch, 0, self.num_kv_heads, self.head_dim), dtype=past_v.dtype)
            elif past_k.shape[2] != self.num_kv_heads:  # Unexpected head count
                raise ValueError(f"Past cache has unexpected number of heads: {past_k.shape[2]}, expected {self.num_kv_heads}")
            
            # Concatenate along sequence dimension
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        
        # Store cache before repeating (should be in KV head format)
        cache_k = k
        cache_v = v
        
        # GQA: repeat k/v to match query heads for attention computation
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Transpose for attention: [b, h, s, d]
        q = jnp.transpose(q, (0,2,1,3))  # [batch, num_heads, seq, head_dim]
        k = jnp.transpose(k, (0,2,1,3))  # [batch, num_heads, seq, head_dim]
        v = jnp.transpose(v, (0,2,1,3))  # [batch, num_heads, seq, head_dim]
        
        # Attention computation
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0,2,1,3)).reshape(batch, seq, self.hidden_size)
        
        # Output projection with tensor parallelism (includes all-reduce)
        output = self.o_proj(attn_out)
        
        return output, (cache_k, cache_v)

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c.get("intermediate_size", 4 * self.hidden_size)
        
        # Use tensor parallel layers with proper sharding
        self.gate_proj = TensorParallelDense(
            features=self.intermediate_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model")  # Shard output features across model parallel
        )
        self.up_proj = TensorParallelDense(
            features=self.intermediate_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model")  # Shard output features across model parallel
        )
        self.down_proj = TensorParallelDense(
            features=self.hidden_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=("model", None),  # Shard input features across model parallel
            reduce_scatter=True  # All-reduce the output
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
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), dtype=self.dtype, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), dtype=self.dtype, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        batch, seq, _ = hidden_states.shape
        if position_ids is None:
            position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
        cos, sin = compute_cos_sin_cache(position_ids, self.self_attn.head_dim, self.self_attn.rope_theta)
        
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos=cos,
            sin=sin
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, past_key_value

class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.vocab_size = c["vocab_size"]
        self.hidden_size = c["hidden_size"]
        self.num_layers = c["num_hidden_layers"]
        self.embed_tokens = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size, dtype=self.dtype, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") for i in range(self.num_layers)]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), dtype=self.dtype, name="norm")
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype, use_bias=False, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        
        # Determine key length (for cache handling)
        if past_key_values is not None and past_key_values[0] is not None:
            past_k, _ = past_key_values[0]
            key_len = past_k.shape[1] + seq  # past + current
        else:
            key_len = seq
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, 1, seq), dtype=self.dtype)
        
        # Create proper causal mask for variable lengths
        causal_mask = make_causal_mask(seq, key_len)
        causal_mask = causal_mask[None, None, :, :]  # Add batch and head dims
        
        # Convert attention_mask to bias: 0 -> -1e9, 1 -> 0
        attention_bias = (1.0 - attention_mask) * -1e9
        
        # For generation, we need to extend attention bias to match key length
        if key_len > seq:
            # Pad attention bias to match key length
            pad_len = key_len - seq
            pad_bias = jnp.zeros((batch, 1, 1, pad_len), dtype=self.dtype)
            attention_bias = jnp.concatenate([pad_bias, attention_bias], axis=-1)
        
        # Combine attention bias and causal mask
        attention_bias = attention_bias + causal_mask
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        
        # Process each layer and collect new key-value caches
        new_key_values = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, new_key_value = layer(
                hidden_states, 
                attention_mask=attention_bias, 
                position_ids=position_ids,
                past_key_value=past_key_value
            )
            new_key_values.append(new_key_value)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": new_key_values
            }
        return logits

# === Weight loading (copy from working version) ===

def get_param_path(name):
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    if name in direct_mapping:
        return direct_mapping[name]
    import re
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        layer_name = f"layers_{layer_idx}"
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (layer_name, "self_attn", proj_name, param_name)
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        layer_name = f"layers_{layer_idx}"
        proj_name = f"{proj_type}_proj"
        return (layer_name, "mlp", proj_name, "kernel")
    rotary_match = re.match(rotary_pattern, name)
    if rotary_match:
        return None
    return None

def transpose_if_needed(name, param):
    if "embed_tokens.weight" in name:
        return param
    if "layernorm.weight" in name or "norm.weight" in name:
        return param  # Don't transpose 1D layer norm weights
    if "weight" in name and ("proj" in name or "lm_head" in name):
        return jnp.transpose(param)
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    flax_params = {"params": {}}
    unmapped_keys = []
    
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            param_path = get_param_path(key)
            if param_path is None:
                unmapped_keys.append(key)
                continue
                
            param = f.get_tensor(key)
            original_dtype = param.dtype
            original_shape = param.shape
            
            # Handle dtype conversion safely
            if original_dtype == np.float16 and dtype == jnp.bfloat16:
                param = param.astype(np.float32)  # safer than direct bf16 conversion
            
            param = jnp.array(param, dtype=dtype)
            param = transpose_if_needed(key, param)
            
            current_dict = flax_params["params"]
            for path_part in param_path[:-1]:
                if path_part not in current_dict:
                    current_dict[path_part] = {}
                current_dict = current_dict[path_part]
            current_dict[param_path[-1]] = param
            
            logger.debug(f"Loaded {key} -> {'/'.join(param_path)}: {original_shape} {original_dtype} -> {param.shape} {param.dtype}")
            del param
            gc.collect()
    
    if unmapped_keys:
        logger.info(f"Unmapped keys in {os.path.basename(file_path)}: {unmapped_keys}")
    
    return flax_params

def merge_param_dicts(base_dict, new_dict):
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict) and isinstance(base_dict[key], dict):
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files."""
    logger.info("Loading weights...")
    
    # 1. Initialize full param tree with dummy input (inside mesh context)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    # Initialize model params inside the mesh context to access the 'model' axis
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 2. Load weights from safetensors files
    param_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            logger.info(f"Loading {file}")
            file_params = process_safetensors_file(file_path, dtype)
            param_dict = merge_param_dicts(param_dict, file_params)
    
    # 3. Map weights to model structure
    def map_params(params, param_dict):
        if isinstance(params, dict):
            out = {}
            for k, v in params.items():
                if k in param_dict:  # <- use loaded value if present
                    out[k] = map_params(v, param_dict[k])
                else:
                    out[k] = v
            return out
        else:  # leaf – replace if we have it
            return param_dict if isinstance(param_dict, (jnp.ndarray, np.ndarray)) else params
    
    # 4. Update initialized params with loaded weights
    params = map_params(init_params, param_dict)
    
    logger.info("✅ Weights loaded successfully (no weight tying)")
    return params

# === Generation ===

def sample_next_token(logits, temperature=0.7):
    """Simple sampling from logits with temperature."""
    if temperature < 1e-5:
        return jnp.argmax(logits, axis=-1)
    else:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        return jax.random.categorical(rng_key, logits / temperature, axis=-1)

def generate_text(model, params, tokenizer, prompt, max_tokens, temperature=0.7):
    """Generate text using the model with tensor parallelism."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create attention mask - use 4D format for Qwen model
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Position IDs
    position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
    
    # Initialize generation state
    state = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    
    # Track generated text
    generated_text = ""
    
    # Generate tokens
    for _ in range(max_tokens):
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=state["input_ids"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            past_key_values=state["past_key_values"],
            return_dict=True
        )
        
        # Get logits and past key values
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        # Sample next token
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature)
        
        # Update state
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode and print token
        token = tokenizer.decode(next_token[0])
        generated_text += token
        print(token, end="", flush=True)
        
        # Check for end of sequence
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    print()  # New line at end
    return generated_text

def main():
    # Setup multi-device environment
    print("🔥 Real Tensor Parallelism for Qwen 2.5-7B")
    print("=" * 60)
    mesh = setup_devices()
    
    parser = argparse.ArgumentParser(description="Real Tensor Parallel Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Load config and tokenizer
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"🚀 Using mesh: {mesh}")
    print(f"🔥 Model parallel size: {mesh.shape['model']}")
    print(f"📊 Data parallel size: {mesh.shape['data']}")
    
    # Create and run model with tensor parallelism
    with mesh:
        model = Qwen25ForCausalLM(config=config, dtype=dtype)
        params = load_params(model, args.model_path, dtype)
        gc.collect(); jax.clear_caches()
        
        print(f"\n🎯 Generating with prompt: '{args.prompt}'")
        generated_text = generate_text(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    # Clean up
    del params; del model; gc.collect(); jax.clear_caches()
    print(f"\n✅ Generation complete!")

if __name__ == "__main__":
    main()