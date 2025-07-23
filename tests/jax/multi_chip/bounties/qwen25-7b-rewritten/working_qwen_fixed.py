#!/usr/bin/env python3
"""
Fixed Working Qwen2.5-7B using exact patterns from successful original
"""
import os
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional
import gc

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open
from transformers import AutoTokenizer

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("working_qwen_fixed")

# === ORIGINAL HELPER FUNCTIONS (EXACT FROM WORKING) ===
def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    pos = jnp.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))
    freqs = jnp.einsum('bi,j->bij', pos, inv_freq)
    
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
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

# === MODEL COMPONENTS (EXACT PATTERN FROM WORKING) ===
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
        self.rope_theta = c.get("rope_theta", 10000.0)
        
        # Standard Dense layers (proven to work)
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=self.dtype, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=self.dtype, use_bias=False, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="o_proj")

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        
        # CRITICAL: Project and reshape (exact pattern from working version)
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            if cos is None or sin is None:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Handle past key-value cache (exact logic from working version)
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
        
        # Store cache for return (before repeating for GQA)
        cache_k = k
        cache_v = v
        
        # GQA: repeat k/v to match query heads
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Transpose for attention: [batch, heads, seq, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention computation (exact from working)
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq, self.hidden_size)
        
        # Return both output and updated cache (in same format as input: [batch, seq, num_kv_heads, head_dim])
        return self.o_proj(attn_out), (cache_k, cache_v)

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c.get("intermediate_size", 4 * self.hidden_size)
        
        self.gate_proj = nn.Dense(self.intermediate_size, dtype=self.dtype, use_bias=False, name="gate_proj")
        self.up_proj = nn.Dense(self.intermediate_size, dtype=self.dtype, use_bias=False, name="up_proj")
        self.down_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="down_proj")
    
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
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), 
                                        dtype=self.dtype, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), 
                                                 dtype=self.dtype, name="post_attention_layernorm")
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
        
        self.embed_tokens = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_size, 
                                   dtype=self.dtype, name="embed_tokens")
        # CRITICAL: Use layers_{i} naming like working version
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") 
                      for i in range(self.num_layers)]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5)), 
                             dtype=self.dtype, name="norm")
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype, use_bias=False, name="lm_head")

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
        
        # Create proper causal mask
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
        
        # Process each layer
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
            return {"logits": logits, "past_key_values": new_key_values}
        return logits

# === PARAMETER LOADING (EXACT FROM WORKING VERSION) ===
def get_param_path(name):
    """Map HF parameter names to Flax structure - EXACT from working version"""
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
        layer_name = f"layers_{layer_idx}"  # CRITICAL: layers_{i} naming
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
        
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        param_type = attn_match.group(3)
        layer_name = f"layers_{layer_idx}"  # CRITICAL: layers_{i} naming
        proj_name = f"{proj_type}_proj"
        param_name = "kernel" if param_type == "weight" else "bias"
        return (layer_name, "self_attn", proj_name, param_name)
        
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        layer_name = f"layers_{layer_idx}"  # CRITICAL: layers_{i} naming
        proj_name = f"{proj_type}_proj"
        return (layer_name, "mlp", proj_name, "kernel")
        
    rotary_match = re.match(rotary_pattern, name)
    if rotary_match:
        return None
        
    return None

def transpose_if_needed(key, param):
    """Transpose weights from HF format to JAX format"""
    if "embed_tokens.weight" in key or "norm.weight" in key or "layernorm.weight" in key:
        return param
    if "weight" in key and ("proj" in key or "lm_head" in key):
        return jnp.transpose(param)
    return param

def process_safetensors_file(file_path, dtype):
    """Load parameters from safetensors file"""
    flax_params = {"params": {}}
    
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            param_path = get_param_path(key)
            if param_path is None:
                continue
                
            param = f.get_tensor(key)
            if param.dtype == np.float16 and dtype == jnp.bfloat16:
                param = param.astype(np.float32)
            
            param = jnp.array(param, dtype=dtype)
            param = transpose_if_needed(key, param)
            
            current_dict = flax_params["params"]
            for path_part in param_path[:-1]:
                if path_part not in current_dict:
                    current_dict[path_part] = {}
                current_dict = current_dict[path_part]
            current_dict[param_path[-1]] = param
            
            del param
            gc.collect()
    
    return flax_params

def merge_param_dicts(base_dict, new_dict):
    """Merge parameter dictionaries"""
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict) and isinstance(base_dict[key], dict):
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files - EXACT from working version"""
    logger.info("Loading weights...")
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # Load weights from safetensors files
    param_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            logger.info(f"Loading {file}")
            file_params = process_safetensors_file(file_path, dtype)
            param_dict = merge_param_dicts(param_dict, file_params)
    
    # Map weights to model structure
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
    
    params = map_params(init_params, param_dict)
    
    # Validation checks
    logger.info("Validating loaded weights...")
    
    # Check that weights actually changed from initialization
    init_embed_std = jnp.std(init_params['params']['embed_tokens']['embedding'])
    loaded_embed_std = jnp.std(params['params']['embed_tokens']['embedding'])
    logger.info(f"Embedding std - init: {float(init_embed_std):.6f}, loaded: {float(loaded_embed_std):.6f}")
    
    if abs(float(init_embed_std) - float(loaded_embed_std)) < 1e-6:
        logger.warning("WARNING: Embedding weights appear unchanged from initialization!")
    
    total_params = sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    return params

# === GENERATION (GREEDY LIKE WORKING VERSION) ===
def simple_greedy_generate(model, params, tokenizer, prompt, max_tokens=50):
    """Generate using GREEDY decoding (argmax) - exact pattern from working gsm8k"""
    
    # Tokenize input (same as working version)
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # Create proper 4D attention mask (CRITICAL from working version)
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
    
    # Position IDs
    position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
    
    # Initialize generation state (exact from working version)
    state = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": None,
    }
    
    generated_text = ""
    
    for i in range(max_tokens):
        # Forward pass
        outputs = model.apply(
            params,
            input_ids=state["input_ids"],
            attention_mask=state["attention_mask"],
            position_ids=state["position_ids"],
            past_key_values=state["past_key_values"],
            return_dict=True
        )
        
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        # CRITICAL: GREEDY selection (argmax) - exact from working version
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        
        # Update state for next iteration (exact from working version)
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode token
        token = tokenizer.decode(next_token[0])
        generated_text += token
        print(token, end="", flush=True)
        
        # Stop conditions
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    print()  # New line
    return generated_text

# === MAIN ===
def main():
    parser = argparse.ArgumentParser(description="Fixed Working Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    dtype = jnp.bfloat16
    
    # Load config and model
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load parameters
    params = load_params(model, args.model_path, dtype)
    
    # Generate text using GREEDY
    logger.info(f"Generating from prompt: '{args.prompt}'")
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: ", end="")
    generated_text = simple_greedy_generate(model, params, tokenizer, args.prompt, args.max_tokens)
    print(f"\nComplete: {generated_text}")

if __name__ == "__main__":
    main() 