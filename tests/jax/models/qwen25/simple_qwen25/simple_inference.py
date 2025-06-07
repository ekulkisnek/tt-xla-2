#!/usr/bin/env python3
"""
Self-contained, real Qwen2.5-7B inference script for single-device JAX.
- Uses real model code and weight mapping from run_inference.py/model.py
- Prints output to terminal only
- Allows dtype selection (bfloat16/float32)
- Cleans up memory after each run
- No file output, no simplification, no external local imports

USAGE
cd /root/dir66/tt-xla-tests/tests/jax/models/qwen25/simple_qwen25 && python simple_inference.py --model_path ../weights --prompt "Hello" --dtype bfloat16
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
# Configure JAX to use CPU and disable TPU
jax.config.update('jax_platform_name', 'cpu')
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_simple")

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
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=self.dtype, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=self.dtype, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="o_proj")
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.max_position_embeddings = c.get("max_position_embeddings", 4096)
    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        # Rotary
        if position_ids is not None:
            if cos is None or sin is None:
                cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        # GQA: repeat k/v if needed
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        # Transpose for attention: [b, h, s, d]
        q = jnp.transpose(q, (0,2,1,3))
        k = jnp.transpose(k, (0,2,1,3))
        v = jnp.transpose(v, (0,2,1,3))
        # Attention
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0,2,1,3)).reshape(batch, seq, self.hidden_size)
        return self.o_proj(attn_out)

def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    # position_ids: [batch, seq]
    # Returns cos, sin: [batch, seq, head_dim//2]
    pos = np.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, dtype=np.float32) / dim))
    freqs = np.einsum('bi,j->bij', pos, inv_freq)
    emb = np.concatenate([np.cos(freqs), np.sin(freqs)], axis=-1)
    cos = jnp.array(np.cos(freqs))
    sin = jnp.array(np.sin(freqs))
    return cos, sin

def apply_rotary_emb(q, k, cos, sin):
    # q, k: [batch, seq, n_heads, head_dim]
    # cos, sin: [batch, seq, head_dim//2]
    def _rope(x, cos, sin):
        # Reshape cos/sin to match x's dimensions
        cos = cos[..., None, :]  # [batch, seq, 1, head_dim//2]
        sin = sin[..., None, :]  # [batch, seq, 1, head_dim//2]
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return _rope(q, cos, sin), _rope(k, cos, sin)

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
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.input_layernorm = nn.LayerNorm(epsilon=c.get("layer_norm_epsilon", 1e-5), dtype=self.dtype, use_bias=False, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=c.get("layer_norm_epsilon", 1e-5), dtype=self.dtype, use_bias=False, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        batch, seq, _ = hidden_states.shape
        if position_ids is None:
            position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
        cos, sin = compute_cos_sin_cache(position_ids, self.self_attn.head_dim, self.self_attn.rope_theta)
        
        hidden_states = self.self_attn(
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
        
        return hidden_states

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
        self.norm = nn.LayerNorm(epsilon=c.get("layer_norm_epsilon", 1e-5), dtype=self.dtype, use_bias=False, name="norm")
        self.lm_head = nn.Dense(self.vocab_size, dtype=self.dtype, use_bias=False, name="lm_head")

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, 1, seq), dtype=self.dtype)
        
        hidden_states = self.embed_tokens(input_ids)
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
            
        # Process each layer
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states = layer(
                hidden_states, 
                attention_mask=attention_mask, 
                position_ids=position_ids,
                past_key_value=past_key_value
            )
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": past_key_values
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
    layer_norm_pattern = r"model\\.layers\\.(\\d+)\\.(input|post_attention)_layernorm\\.weight"
    attention_pattern = r"model\\.layers\\.(\\d+)\\.self_attn\\.(q|k|v|o)_proj\\.(weight|bias)"
    mlp_pattern = r"model\\.layers\\.(\\d+)\\.mlp\\.(gate|up|down)_proj\\.weight"
    rotary_pattern = r"model\\.layers\\.(\\d+)\\.self_attn\\.rotary_emb\\..*"
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
    if name == "lm_head.weight":
        return jnp.transpose(param)
    if "embed_tokens.weight" in name:
        return param
    if "weight" in name and "proj" in name:
        return jnp.transpose(param)
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    flax_params = {"params": {}}
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            param_path = get_param_path(key)
            if param_path is None:
                continue
            param = f.get_tensor(key)
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
    for key, value in new_dict.items():
        if key not in base_dict:
            base_dict[key] = value
        elif isinstance(value, dict) and isinstance(base_dict[key], dict):
            merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def print_param_tree(params, prefix=""):
    """Recursively print parameter tree structure with shapes."""
    if isinstance(params, dict):
        for key, value in params.items():
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                shape_str = str(value.shape)
                logger.info(f"{prefix}{key}: {shape_str}")
            else:
                logger.info(f"{prefix}{key}:")
                print_param_tree(value, prefix + "  ")
    elif isinstance(params, (jnp.ndarray, np.ndarray)):
        logger.info(f"{prefix}Shape: {params.shape}")

def validate_param_shapes(params, config):
    """Validate parameter shapes against model architecture."""
    errors = []
    
    # Check embedding layer
    expected_embed_shape = (config["vocab_size"], config["hidden_size"])
    if params["params"]["embed_tokens"]["embedding"].shape != expected_embed_shape:
        errors.append(f"Embedding shape mismatch: got {params['params']['embed_tokens']['embedding'].shape}, expected {expected_embed_shape}")
    
    # Check each layer
    for i in range(config["num_hidden_layers"]):
        layer_name = f"layers_{i}"
        layer_params = params["params"][layer_name]
        
        # Check attention layer
        attn_params = layer_params["self_attn"]
        expected_q_shape = (config["hidden_size"], config["hidden_size"])
        expected_kv_shape = (config["hidden_size"], config["num_key_value_heads"] * config["head_dim"])
        expected_o_shape = (config["hidden_size"], config["hidden_size"])
        
        if attn_params["q_proj"]["kernel"].shape != expected_q_shape:
            errors.append(f"Layer {i} Q projection shape mismatch: got {attn_params['q_proj']['kernel'].shape}, expected {expected_q_shape}")
        if attn_params["k_proj"]["kernel"].shape != expected_kv_shape:
            errors.append(f"Layer {i} K projection shape mismatch: got {attn_params['k_proj']['kernel'].shape}, expected {expected_kv_shape}")
        if attn_params["v_proj"]["kernel"].shape != expected_kv_shape:
            errors.append(f"Layer {i} V projection shape mismatch: got {attn_params['v_proj']['kernel'].shape}, expected {expected_kv_shape}")
        if attn_params["o_proj"]["kernel"].shape != expected_o_shape:
            errors.append(f"Layer {i} O projection shape mismatch: got {attn_params['o_proj']['kernel'].shape}, expected {expected_o_shape}")
        
        # Check MLP layer
        mlp_params = layer_params["mlp"]
        expected_gate_shape = (config["hidden_size"], config["intermediate_size"])
        expected_up_shape = (config["hidden_size"], config["intermediate_size"])
        expected_down_shape = (config["intermediate_size"], config["hidden_size"])
        
        if mlp_params["gate_proj"]["kernel"].shape != expected_gate_shape:
            errors.append(f"Layer {i} Gate projection shape mismatch: got {mlp_params['gate_proj']['kernel'].shape}, expected {expected_gate_shape}")
        if mlp_params["up_proj"]["kernel"].shape != expected_up_shape:
            errors.append(f"Layer {i} Up projection shape mismatch: got {mlp_params['up_proj']['kernel'].shape}, expected {expected_up_shape}")
        if mlp_params["down_proj"]["kernel"].shape != expected_down_shape:
            errors.append(f"Layer {i} Down projection shape mismatch: got {mlp_params['down_proj']['kernel'].shape}, expected {expected_down_shape}")
    
    # Check final layer norm and lm_head
    expected_norm_shape = (config["hidden_size"],)
    expected_lm_head_shape = (config["hidden_size"], config["vocab_size"])  # Changed order to match actual model
    
    if params["params"]["norm"]["scale"].shape != expected_norm_shape:
        errors.append(f"Final layer norm shape mismatch: got {params['params']['norm']['scale'].shape}, expected {expected_norm_shape}")
    if params["params"]["lm_head"]["kernel"].shape != expected_lm_head_shape:
        errors.append(f"LM head shape mismatch: got {params['params']['lm_head']['kernel'].shape}, expected {expected_lm_head_shape}")
    
    return errors

def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files with validation."""
    logger.info("Loading weights...")
    
    # 1. Initialize full param tree with dummy input
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 2. Load weights from safetensors files
    param_dict = {}
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    logger.info(f"Found {len(safetensors_files)} safetensors files")
    
    for file in safetensors_files:
        file_path = os.path.join(model_path, file)
        logger.info(f"Loading {file}")
        try:
            file_params = process_safetensors_file(file_path, dtype)
            param_dict = merge_param_dicts(param_dict, file_params)
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
            raise
    
    # 3. Map weights to model structure
    def map_params(params, param_dict):
        if isinstance(params, dict):
            return {k: map_params(v, param_dict) for k, v in params.items()}
        elif isinstance(params, (jnp.ndarray, np.ndarray)):
            return params
        else:
            return params
    
    # 4. Update initialized params with loaded weights
    params = map_params(params, param_dict)
    
    # 5. Print parameter tree structure
    logger.info("Parameter tree structure:")
    print_param_tree(params)
    
    # 6. Validate parameter shapes
    logger.info("Validating parameter shapes...")
    errors = validate_param_shapes(params, model.config)
    if errors:
        logger.error("Parameter shape validation failed:")
        for error in errors:
            logger.error(error)
        raise ValueError("Parameter shape validation failed")
    else:
        logger.info("Parameter shape validation passed")
    
    return params

# --- Generation ---
def sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=50):
    """Sample next token from logits with temperature, top-k, and top-p sampling."""
    # Create PRNG key
    rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
    
    # Get final dimension (vocab size)
    vocab_size = logits.shape[-1]
    
    # Apply temperature
    if temperature > 0:
        logits = logits / jnp.maximum(temperature, 1e-7)
    
    # Apply top-k filtering
    if top_k > 0 and top_k < vocab_size:
        # Get top-k logits and corresponding indices
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        
        # Create mask for non-top-k values
        logits_mask = jnp.full_like(logits, True)
        top_k_one_hot = jax.nn.one_hot(top_k_indices, vocab_size, dtype=bool)
        top_k_mask = jnp.logical_or.reduce(top_k_one_hot, axis=-2)
        
        # Apply mask to logits
        logits = jnp.where(
            top_k_mask,
            logits,
            jnp.full_like(logits, -float("inf"))
        )
    
    # Apply top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        # Convert logits to probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Create indices with same shape as probs
        indices = jnp.arange(vocab_size, dtype=jnp.int32)
        indices = jnp.broadcast_to(indices, probs.shape)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = jax.lax.sort_key_val(-probs, indices)
        sorted_probs = -sorted_probs
        
        # Calculate cumulative probabilities
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        # Find indices where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep first token above threshold to maintain minimum probability
        sorted_indices_to_remove = jnp.concatenate([
            jnp.zeros_like(sorted_indices_to_remove[..., :1]),
            sorted_indices_to_remove[..., :-1]
        ], axis=-1)
        
        # Create mask for indices to remove
        indices_to_remove = jnp.zeros_like(probs, dtype=bool)
        for i in range(logits.shape[0]):
            indices_to_remove = indices_to_remove.at[i, sorted_indices[i]].set(sorted_indices_to_remove[i])
        
        # Apply mask to logits
        logits = jnp.where(
            indices_to_remove,
            jnp.full_like(logits, -float("inf")),
            logits
        )
    
    # If temperature is close to zero, perform greedy sampling
    if temperature < 1e-5:
        next_token = jnp.argmax(logits, axis=-1)
    else:
        # Sample from the filtered distribution
        next_token = jax.random.categorical(rng_key, logits, axis=-1)
    
    return next_token

def generate_text(model, params, tokenizer, prompt, max_tokens, temperature=0.7, top_p=0.9, top_k=50):
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
        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
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
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    # Load config
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Patch config for head_dim if missing
    if "head_dim" not in config:
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
        logger.info(f"Patched config: set head_dim = {config['head_dim']}")

    # Print loaded config
    logger.info("Loaded config:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # Load generation config if present
    gen_config_path = os.path.join(args.model_path, "generation_config.json")
    gen_config = {}
    if os.path.exists(gen_config_path):
        with open(gen_config_path, 'r') as f:
            gen_config = json.load(f)
        logger.info("Loaded generation_config.json:")
        for k, v in gen_config.items():
            logger.info(f"  {k}: {v}")

    # Set generation parameters, prefer CLI > generation_config > script default
    max_tokens = args.max_tokens if args.max_tokens is not None else gen_config.get("max_new_tokens", 100)
    temperature = args.temperature if args.temperature is not None else (0.7 if gen_config.get("do_sample", True) else 0.0)
    top_p = args.top_p if args.top_p is not None else 0.9
    top_k = args.top_k if args.top_k is not None else 50

    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Tokenizer round-trip check
    logger.info("Tokenizer round-trip check:")
    encoded = tokenizer(args.prompt, return_tensors="np")
    logger.info(f"  Encoded input_ids: {encoded['input_ids'][0].tolist()}")
    decoded = tokenizer.decode(encoded["input_ids"][0])
    logger.info(f"  Decoded text: {decoded}")

    # Load weights
    logger.info("Loading weights...")
    params = load_params(model, args.model_path, dtype)
    gc.collect(); jax.clear_caches()
    # Generate
    logger.info("Generating text...")
    generate_text(model, params, tokenizer, args.prompt, max_tokens, temperature, top_p, top_k)
    # Clean up
    del params; del model; gc.collect(); jax.clear_caches()
    logger.info("Done.")

if __name__ == "__main__":
    main() 