#!/usr/bin/env python3
# STAGE 1: Single device refactor parity - TP disabled
"""
Self-contained, real Qwen2.5-7B inference script for single-device JAX.
- Uses real model code and weight mapping from run_inference.py/model.py
- Prints output to terminal only
- Allows dtype selection (bfloat16/float32)
- Cleans up memory after each run
- No file output, no simplification, no external local imports

Usage 
python simple_inference.py --model_path ../weights --prompt "Hello, how are you?" --max_tokens 20 --temperature 0.7 --top_p 0.9 --top_k 50 --dtype bfloat16
python q25_jax.py --model_path ../weights --prompt "The capital of France is" --max_tokens 10 --temperature 0.1
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

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_simple")

# === STAGE 2: MESH CREATION ===
def create_mesh(model_parallel: int, data_parallel: int = 1):
    """Create a deterministic device mesh for tensor parallelism."""
    devices = jax.devices()
    if len(devices) < model_parallel * data_parallel:
        raise RuntimeError(f"Need {model_parallel * data_parallel} devices, have {len(devices)}")
    mesh = np.array(devices[:model_parallel * data_parallel]).reshape(data_parallel, model_parallel)
    return Mesh(mesh, ("data", "model"))

# === STAGE 3: TENSOR PARALLEL DENSE LAYER ===
class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism support - Stage 3: real split-K matmul."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    shard_axes: Tuple[Optional[str], Optional[str]] = (None, "model")
    reduce_scatter: bool = False  # For o_proj and down_proj
    
    @nn.compact
    def __call__(self, x):
        # Check if we're in mesh context
        try:
            # Try to detect mesh context
            mesh_context = len(jax.devices()) > 1 and hasattr(jax.sharding, 'current_mesh') 
        except:
            mesh_context = False
        
        # Stage 3: Implement real split-K matmul with sharding constraints ONLY if mesh exists
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.features),
            self.dtype,
        )
        
        if mesh_context:
            # Apply sharding constraints according to your checklist
            kernel = jax.lax.with_sharding_constraint(kernel, P(None, "model"))
            x = jax.lax.with_sharding_constraint(x, P("data", None))
            
            # Each GPU produces its slice
            y_part = jnp.matmul(x, kernel)
            
            # For layers that need all-reduce (attention o_proj, MLP down_proj)
            if self.reduce_scatter:
                y = jax.lax.psum(y_part, axis_name="model")
            else:
                y = y_part  # Leave split for now (testing)
        else:
            # Stage 1: Just do normal Dense computation (no sharding)
            y = jnp.matmul(x, kernel)
        
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.dtype)
            if mesh_context:
                bias = jax.lax.with_sharding_constraint(bias, P("model",))
            y = y + bias
        
        return y

# --- Model code (copied from your real model.py, single-device only) ---
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
        # Stage 3: Use TensorParallelDense for all projections
        self.q_proj = TensorParallelDense(features=self.hidden_size, dtype=self.dtype, 
                                        use_bias=False, reduce_scatter=False)
        self.k_proj = TensorParallelDense(features=self.kv_dim, dtype=self.dtype, 
                                        use_bias=False, reduce_scatter=False)
        self.v_proj = TensorParallelDense(features=self.kv_dim, dtype=self.dtype, 
                                        use_bias=False, reduce_scatter=False)
        self.o_proj = TensorParallelDense(features=self.hidden_size, dtype=self.dtype, 
                                        use_bias=False, reduce_scatter=False)  # Stage 3 Step 2: Disable reduce first
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.max_position_embeddings = c.get("max_position_embeddings", 4096)
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        
        # Project current hidden states
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            if cos is None or sin is None:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Handle past key-value cache
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
        
        # Attention
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0,2,1,3)).reshape(batch, seq, self.hidden_size)
        
        # Return both output and updated cache (in same format as input: [batch, seq, num_kv_heads, head_dim])
        return self.o_proj(attn_out), (cache_k, cache_v)

def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    # position_ids: [batch, seq]
    # Returns cos, sin: [batch, seq, head_dim]
    pos = np.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, dtype=np.float32) / dim))
    freqs = np.einsum('bi,j->bij', pos, inv_freq)
    
    # Create cos and sin for full head_dim (not head_dim//2)
    cos = jnp.array(np.cos(freqs))
    sin = jnp.array(np.sin(freqs))
    
    # Repeat to match head_dim
    cos = jnp.repeat(cos, 2, axis=-1)
    sin = jnp.repeat(sin, 2, axis=-1)
    
    return cos, sin

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(q, k, cos, sin):
    # q, k: [batch, seq, n_heads, head_dim]
    # cos, sin: [batch, seq, head_dim]
    def _rope(x, cos, sin):
        # Reshape cos/sin to match x's dimensions
        cos = cos[..., None, :]  # [batch, seq, 1, head_dim]
        sin = sin[..., None, :]  # [batch, seq, 1, head_dim]
        return (x * cos) + (rotate_half(x) * sin)
    return _rope(q, cos, sin), _rope(k, cos, sin)

def make_causal_mask(q_len, k_len):
    """Create causal mask for different query and key lengths."""
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return (i < j - (k_len - q_len)) * -1e9

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c.get("intermediate_size", 4 * self.hidden_size)
        # Stage 3: Use TensorParallelDense for MLP projections
        self.gate_proj = TensorParallelDense(features=self.intermediate_size, dtype=self.dtype, 
                                           use_bias=False, reduce_scatter=False)
        self.up_proj = TensorParallelDense(features=self.intermediate_size, dtype=self.dtype, 
                                         use_bias=False, reduce_scatter=False)
        self.down_proj = TensorParallelDense(features=self.hidden_size, dtype=self.dtype, 
                                           use_bias=False, reduce_scatter=False)  # Stage 3 Step 2: Disable reduce first
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
        hidden_states = residual + self.mlp(hidden_states)
        
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

# --- Weight loading (real mapping from run_inference.py/model.py) ---
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

def process_safetensors_file(file_path, dtype=jnp.bfloat16, mesh=None):
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
            param_before_transpose = param
            param = transpose_if_needed(key, param)
            
            # Validate transpose worked as expected
            if "weight" in key and ("proj" in key or "lm_head" in key):
                if jnp.array_equal(param, param_before_transpose):
                    logger.warning(f"Expected transpose for {key} but array unchanged")
                else:
                    # Quick checksum to catch double-transpose
                    before_mean = jnp.mean(param_before_transpose[:min(2, param_before_transpose.shape[0]), :min(2, param_before_transpose.shape[1])])
                    after_mean = jnp.mean(param[:min(2, param.shape[0]), :min(2, param.shape[1])])
                    logger.debug(f"Transpose {key}: before_mean={float(before_mean):.6f}, after_mean={float(after_mean):.6f}")
            
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

def load_params(model, model_path, dtype, mesh=None):
    """Load model parameters from safetensors files."""
    logger.info("Loading weights...")
    
    # 1. Initialize full param tree with dummy input
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 2. Load weights from safetensors files
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
                if k in param_dict:  # <- use loaded value if present
                    out[k] = map_params(v, param_dict[k])
                else:
                    out[k] = v
            return out
        else:  # leaf – replace if we have it
            return param_dict if isinstance(param_dict, (jnp.ndarray, np.ndarray)) else params
    
    # 4. Update initialized params with loaded weights
    params = map_params(init_params, param_dict)
    
    # 4.5. Fix LM-head weight tying (after transpose is applied)
    # DISABLED: This fix improves numerical metrics but degrades generation quality
    # embed_tokens = params['params']['embed_tokens']['embedding']
    # lm_head = params['params']['lm_head']['kernel']
    # 
    # # For Qwen2.5, lm_head should be tied to embedding weights (transposed)
    # if embed_tokens.shape == (lm_head.shape[1], lm_head.shape[0]):
    #     # Shapes are compatible for transpose tying
    #     params['params']['lm_head']['kernel'] = embed_tokens.T
    #     logger.info("✓ LM-head weights tied to embedding weights (transposed)")
    # else:
    #     logger.warning(f"⚠️ Cannot tie weights: embed_shape={embed_tokens.shape}, lm_head_shape={lm_head.shape}")
    
    # 5. Validation checks
    logger.info("Validating loaded weights...")
    
    # Check that weights actually changed from initialization
    init_embed_std = jnp.std(init_params['params']['embed_tokens']['embedding'])
    loaded_embed_std = jnp.std(params['params']['embed_tokens']['embedding'])
    logger.info(f"Embedding std - init: {float(init_embed_std):.6f}, loaded: {float(loaded_embed_std):.6f}")
    
    if abs(float(init_embed_std) - float(loaded_embed_std)) < 1e-6:
        logger.warning("WARNING: Embedding weights appear unchanged from initialization!")
    
    # Count total parameters
    def count_params(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(np.prod(leaf.shape) for leaf in leaves)
    
    total_params = count_params(params)
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Check if embed_tokens and lm_head are tied (should be for Qwen 2.5)
    embed_tokens = params['params']['embed_tokens']['embedding']
    lm_head = params['params']['lm_head']['kernel']
    
    if embed_tokens.shape == lm_head.shape:
        max_diff = jnp.max(jnp.abs(embed_tokens - lm_head))
        logger.info(f"Embed↔LM-head tie check: max_diff = {float(max_diff):.2e}")
        if float(max_diff) < 1e-6:
            logger.info("✓ Weights are properly tied")
        else:
            logger.warning("✗ Weights are NOT tied (this may be expected)")
    else:
        logger.info(f"Embed shape: {embed_tokens.shape}, LM head shape: {lm_head.shape}")
    
    return params

# --- Validation removed for streamlined version ---

# --- Generation ---
def sample_next_token(logits, temperature=0.7):
    """Simple sampling from logits with temperature."""
    if temperature < 1e-5:
        return jnp.argmax(logits, axis=-1)
    else:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        return jax.random.categorical(rng_key, logits / temperature, axis=-1)

def generate_text(model, params, tokenizer, prompt, max_tokens, temperature=0.7):
    """Generate text using the model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create attention mask - use 4D format for Qwen model
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Position IDs - make sure to match the input_ids length
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

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B Inference (single device, real model)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    args = parser.parse_args()
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    # Load config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Stage 1 vs Stage 2 branching
    if args.tp == 1:
        logger.info("Stage 1: Single device mode")
        params = load_params(model, args.model_path, dtype, mesh=None)
        gc.collect(); jax.clear_caches()
        
        # Generate (Stage 1)
        generate_text(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
        
        # Clean up Stage 1
        del params; del model; gc.collect(); jax.clear_caches()
    else:
        logger.info(f"Stage 3: Multi-device mode with TP={args.tp}")
        mesh = create_mesh(model_parallel=args.tp, data_parallel=1)
        logger.info(f"Created mesh: {mesh.devices.shape} with axes {mesh.axis_names}")
        
        # Stage 3 test: mesh.axis_names == ('data','model')
        assert mesh.axis_names == ('data', 'model'), f"Wrong axis names: {mesh.axis_names}"
        logger.info("✓ Stage 3 mesh creation test passed")
        
        # Stage 3: Load parameters with mesh and test split-K matmul
        with mesh:
            with jax.named_scope("model"):
                # Load params under mesh context 
                params = load_params(model, args.model_path, dtype, mesh=mesh)
                
                # Stage 3 Test: Generate one token to test split maths
                logger.info("Stage 3: Testing split-K matmul with mesh...")
                generate_text(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
        
        # Clean up Stage 3
        del params; del model; gc.collect(); jax.clear_caches()

if __name__ == "__main__":
    main() 