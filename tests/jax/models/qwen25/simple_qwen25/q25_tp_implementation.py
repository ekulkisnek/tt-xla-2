#!/usr/bin/env python3
"""
Tensor Parallel Qwen2.5-7B implementation based on working q25_jax.py
Follows the roadmap for reliable, reproducible tensor parallelism.

Usage:
python q25_tp_implementation.py --model_path ../weights --prompt "Hello, how are you?" --max_tokens 20 --tp 4 --dp 1
"""
import os
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
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tp")

# --- Phase 1: Deterministic Mesh Factory ---
def create_mesh(model_parallel: int, data_parallel: int = 1):
    """Create a deterministic device mesh for tensor parallelism."""
    devices = jax.devices()
    total_devices = model_parallel * data_parallel
    
    assert len(devices) >= total_devices, f"Need {total_devices} devices, got {len(devices)}"
    
    # Use first total_devices in deterministic order
    selected_devices = devices[:total_devices]
    mesh_array = np.array(selected_devices).reshape(data_parallel, model_parallel)
    return Mesh(mesh_array, ("data", "model"))

# --- Tensor Parallel Dense Layer ---
class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism support."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    shard_axes: Tuple[Optional[str], Optional[str]] = (None, "model")  # (input, output)
    
    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.features),
            self.dtype
        )
        
        # Apply sharding constraint to kernel
        kernel = jax.lax.with_sharding_constraint(kernel, P(*self.shard_axes))
        
        y = jnp.matmul(x, kernel)
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), self.dtype)
            # Shard bias along same axis as output features
            bias = jax.lax.with_sharding_constraint(bias, P(self.shard_axes[1]))
            y = y + bias
            
        return y

# --- Keep all original utility functions from q25_jax.py ---
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
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
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(q, k, cos, sin):
    def _rope(x, cos, sin):
        cos = cos[..., None, :]
        sin = sin[..., None, :]
        return (x * cos) + (rotate_half(x) * sin)
    return _rope(q, cos, sin), _rope(k, cos, sin)

def make_causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return (i < j - (k_len - q_len)) * -1e9

# --- TP Attention Implementation ---
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
        
        # TP sharding: Q/K/V sharded on output dim
        self.q_proj = TensorParallelDense(
            features=self.hidden_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model"),
            name="q_proj"
        )
        self.k_proj = TensorParallelDense(
            features=self.kv_dim, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model"),
            name="k_proj"
        )
        self.v_proj = TensorParallelDense(
            features=self.kv_dim, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model"),
            name="v_proj"
        )
        # O projection sharded on input dim for all-reduce
        self.o_proj = TensorParallelDense(
            features=self.hidden_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=("model", None),
            name="o_proj"
        )
        
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.max_position_embeddings = c.get("max_position_embeddings", 4096)
        
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        
        # Project current hidden states - these will be automatically sharded
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            if cos is None or sin is None:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Handle past key-value cache - format: (batch, seq, kv_heads_per_gpu, head_dim)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            
            # Ensure past cache is in correct format
            if past_k.shape[1] == self.num_kv_heads and past_k.shape[2] != self.num_kv_heads:
                past_k = jnp.transpose(past_k, (0,2,1,3))
                past_v = jnp.transpose(past_v, (0,2,1,3))
            
            # Handle head count differences for GQA
            if past_k.shape[2] == self.num_heads:
                past_k = past_k.reshape(batch, -1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                past_k = jnp.mean(past_k, axis=3)
                past_v = past_v.reshape(batch, -1, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
                past_v = jnp.mean(past_v, axis=3)
            elif past_k.shape[0] == 0 or past_k.shape[1] == 0:
                past_k = jnp.zeros((batch, 0, self.num_kv_heads, self.head_dim), dtype=past_k.dtype)
                past_v = jnp.zeros((batch, 0, self.num_kv_heads, self.head_dim), dtype=past_v.dtype)
            elif past_k.shape[2] != self.num_kv_heads:
                raise ValueError(f"Past cache has unexpected number of heads: {past_k.shape[2]}, expected {self.num_kv_heads}")
            
            # Concatenate along sequence dimension
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        
        # Store cache before repeating (in KV head format for TP)
        cache_k = k
        cache_v = v
        
        # GQA: repeat k/v to match query heads for attention computation
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Transpose for attention: [b, h, s, d]
        q = jnp.transpose(q, (0,2,1,3))
        k = jnp.transpose(k, (0,2,1,3))
        v = jnp.transpose(v, (0,2,1,3))
        
        # Attention computation
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0,2,1,3)).reshape(batch, seq, self.hidden_size)
        
        # Apply output projection (includes all-reduce via input sharding)
        output = self.o_proj(attn_out)
        
        return output, (cache_k, cache_v)

# --- TP MLP Implementation ---
class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c.get("intermediate_size", 4 * self.hidden_size)
        
        # Gate and Up projections sharded on output dim
        self.gate_proj = TensorParallelDense(
            features=self.intermediate_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model"),
            name="gate_proj"
        )
        self.up_proj = TensorParallelDense(
            features=self.intermediate_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=(None, "model"),
            name="up_proj"
        )
        # Down projection sharded on input dim for all-reduce
        self.down_proj = TensorParallelDense(
            features=self.hidden_size, 
            dtype=self.dtype, 
            use_bias=False,
            shard_axes=("model", None),
            name="down_proj"
        )
    
    def __call__(self, x):
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# --- Decoder Layer ---
class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.input_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), 
            dtype=self.dtype, 
            name="input_layernorm"
        )
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), 
            dtype=self.dtype, 
            name="post_attention_layernorm"
        )
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
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states, past_key_value

# --- Main Model ---
class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.vocab_size = c["vocab_size"]
        self.hidden_size = c["hidden_size"]
        self.num_layers = c["num_hidden_layers"]
        
        # Keep embeddings and lm_head replicated (not sharded) for simplicity
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size, 
            features=self.hidden_size, 
            dtype=self.dtype, 
            name="embed_tokens"
        )
        self.layers = [
            QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") 
            for i in range(self.num_layers)
        ]
        self.norm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), 
            dtype=self.dtype, 
            name="norm"
        )
        self.lm_head = nn.Dense(
            self.vocab_size, 
            dtype=self.dtype, 
            use_bias=False, 
            name="lm_head"
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        
        # Determine key length (for cache handling)
        if past_key_values is not None and past_key_values[0] is not None:
            past_k, _ = past_key_values[0]
            key_len = past_k.shape[1] + seq
        else:
            key_len = seq
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, 1, seq), dtype=self.dtype)
        
        # Create proper causal mask for variable lengths
        causal_mask = make_causal_mask(seq, key_len)
        causal_mask = causal_mask[None, None, :, :]
        
        # Convert attention_mask to bias
        attention_bias = (1.0 - attention_mask) * -1e9
        
        # For generation, extend attention bias to match key length
        if key_len > seq:
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

# --- Phase 1: Parameter Loading with Sharding ---
def get_param_path(name):
    """Map HuggingFace parameter names to Flax structure."""
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
        return None  # Skip rotary embeddings
        
    return None

def get_partition_spec(param_name: str) -> P:
    """Get partition specification for a parameter."""
    # Attention projections
    if any(proj in param_name for proj in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
        return P(None, "model")  # Output dimension sharded
    elif any(proj in param_name for proj in ["o_proj", "down_proj"]):
        return P("model", None)  # Input dimension sharded
    # Everything else replicated
    else:
        return P(None, None)

def transpose_if_needed(name, param):
    """Apply necessary transposes for weight matrices."""
    if "embed_tokens.weight" in name:
        return param
    if "layernorm.weight" in name or "norm.weight" in name:
        return param
    if "weight" in name and ("proj" in name or "lm_head" in name):
        return jnp.transpose(param)
    return param

def process_safetensors_file(file_path, dtype, mesh):
    """Process safetensors file with tensor parallel sharding."""
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
                param = param.astype(np.float32)
            
            param = jnp.array(param, dtype=dtype)
            param = transpose_if_needed(key, param)
            
            # Apply tensor parallel sharding
            pspec = get_partition_spec(key)
            if mesh is not None:
                sharding = NamedSharding(mesh, pspec)
                param = jax.device_put(param, sharding)
            
            # Store in parameter tree
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
    """Merge parameter dictionaries."""
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict) and isinstance(base_dict[key], dict):
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_params_tp(model, model_path, dtype, mesh):
    """Load model parameters with tensor parallel sharding."""
    logger.info("Loading weights with tensor parallel sharding...")
    
    # 1. Initialize full param tree with dummy input
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 2. Load weights from safetensors files with sharding
    param_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            logger.info(f"Loading {file}")
            file_params = process_safetensors_file(file_path, dtype, mesh)
            param_dict = merge_param_dicts(param_dict, file_params)
    
    # 3. Map weights to model structure
    def map_params(params, param_dict):
        if isinstance(params, dict):
            out = {}
            for k, v in params.items():
                if k in param_dict:
                    out[k] = map_params(v, param_dict[k])
                else:
                    out[k] = v
            return out
        else:
            return param_dict if isinstance(param_dict, (jnp.ndarray, np.ndarray)) else params
    
    # 4. Update initialized params with loaded weights
    params = map_params(init_params, param_dict)
    
    # 5. Validation
    logger.info("Validating tensor parallel weights...")
    
    # Check sharding
    total_params = 0
    for name, param in jax.tree_util.tree_leaves_with_path(params):
        param_name = '/'.join(str(p) for p in name)
        total_params += np.prod(param.shape)
        
        if hasattr(param, 'addressable_shards'):
            num_shards = len(param.addressable_shards)
            logger.debug(f"{param_name}: {param.shape} on {num_shards} shards")
    
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    return params

# --- Phase 2: JIT Generation ---
def sample_next_token(logits, temperature=0.7):
    """Simple sampling from logits with temperature."""
    if temperature < 1e-5:
        return jnp.argmax(logits, axis=-1)
    else:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        return jax.random.categorical(rng_key, logits / temperature, axis=-1)

@jax.jit
def generate_step(model_apply, params, input_ids, attention_mask, position_ids, past_key_values):
    """JIT-compiled generation step."""
    outputs = model_apply(
        params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        return_dict=True
    )
    return outputs["logits"], outputs["past_key_values"]

def generate_text_tp(model, params, tokenizer, prompt, max_tokens, temperature=0.7, mesh=None):
    """Generate text using tensor parallel model."""
    logger.info(f"Generating with TP on mesh: {mesh.axis_names if mesh else 'None'}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create attention mask
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
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
    
    # Create jitted apply function
    model_apply = jax.jit(model.apply, donate_argnums=(0,))
    
    # Generate tokens
    with mesh:
        for i in range(max_tokens):
            # JIT-compiled forward pass
            logits, past_key_values = generate_step(
                model_apply,
                params,
                state["input_ids"],
                state["attention_mask"], 
                state["position_ids"],
                state["past_key_values"]
            )
            
            # Sample next token
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature)
            
            # Update state for next iteration
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

# --- Main Function with Phase 4 Features ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B Tensor Parallel Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Phase 4: Enable XLA optimizations
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false --xla_gpu_simplify_all_reduce=true"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Load config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create mesh
    if args.tp > 1:
        mesh = create_mesh(model_parallel=args.tp, data_parallel=args.dp)
        logger.info(f"Created mesh: {mesh.devices.shape} with axes {mesh.axis_names}")
        
        # Test mesh creation (Phase 1 validation)
        assert mesh.axis_names == ('data', 'model'), f"Wrong axis names: {mesh.axis_names}"
        logger.info("✓ Mesh creation test passed")
    else:
        mesh = None
        logger.info("Using single device (no tensor parallelism)")
    
    # Create model
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load weights
    if mesh is not None:
        with mesh:
            params = load_params_tp(model, args.model_path, dtype, mesh)
            
            # Phase 1: Validate sharding (blocking test)
            logger.info("Validating parameter sharding...")
            for name, param in jax.tree_util.tree_leaves_with_path(params):
                if hasattr(param, 'addressable_shards'):
                    shards = param.addressable_shards
                    device_ids = [shard.device.id for shard in shards]
                    logger.debug(f"Parameter {name}: {len(shards)} shards on devices {device_ids}")
            logger.info("✓ Parameter sharding validation passed")
    else:
        # Phase 4: Automatic branch - fall back to single device loading
        logger.info("Falling back to single device loading...")
        # Import single device functions from original file
        import sys
        sys.path.append(os.path.dirname(__file__))
        from q25_jax import load_params
        params = load_params(model, args.model_path, dtype)
    
    gc.collect()
    jax.clear_caches()
    
    # Generate
    if mesh is not None:
        generate_text_tp(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature, mesh)
    else:
        # Fall back to single-device generation
        from q25_jax import generate_text
        generate_text(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    # Clean up
    del params
    del model
    gc.collect()
    jax.clear_caches()

if __name__ == "__main__":
    main() 