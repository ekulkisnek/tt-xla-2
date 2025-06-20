#!/usr/bin/env python3
"""
Working streamlined Qwen2.5-7B inference script for JAX.
Reduced complexity while maintaining all functionality.

Usage: python simple_inference_working.py --model_path ../weights --prompt "Hello" --max_tokens 20
"""
import os
import json
import gc
import argparse
import time
import re
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open

# --- Model code ---
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
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.kv_dim, dtype=self.dtype, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.kv_dim, dtype=self.dtype, use_bias=False, name="v_proj")
        self.o_proj = nn.Dense(self.hidden_size, dtype=self.dtype, use_bias=False, name="o_proj")
        self.rope_theta = c.get("rope_theta", 10000.0)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, cos=None, sin=None):
        batch, seq, _ = hidden_states.shape
        
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
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        
        cache_k, cache_v = k, v
        
        # GQA: repeat k/v to match query heads
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Attention
        q = jnp.transpose(q, (0,2,1,3))
        k = jnp.transpose(k, (0,2,1,3))
        v = jnp.transpose(v, (0,2,1,3))
        
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0,2,1,3)).reshape(batch, seq, self.hidden_size)
        
        return self.o_proj(attn_out), (cache_k, cache_v)

def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    pos = np.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, dim, dtype=np.float32) / dim))
    freqs = np.einsum('bi,j->bij', pos, inv_freq)
    cos = jnp.repeat(jnp.array(np.cos(freqs)), 2, axis=-1)
    sin = jnp.repeat(jnp.array(np.sin(freqs)), 2, axis=-1)
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
        eps = c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5))
        self.input_layernorm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="input_layernorm")
        self.self_attn = QwenAttention(config=c, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="post_attention_layernorm")
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
        self.num_layers = c["num_hidden_layers"]
        eps = c.get("rms_norm_eps", c.get("layer_norm_epsilon", 1e-5))
        self.embed_tokens = nn.Embed(self.vocab_size, c["hidden_size"], dtype=self.dtype, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=self.dtype, name=f"layers_{i}") for i in range(self.num_layers)]
        self.norm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="norm")
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
        
        # Create proper causal mask for variable lengths
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        
        # Convert attention_mask to bias: 0 -> -1e9, 1 -> 0
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
        
        return {"logits": logits, "past_key_values": new_key_values} if return_dict else logits

# --- Weight loading ---
def load_params(model, model_path, dtype):
    # Initialize params
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # Weight mapping
    def get_param_path(name):
        direct_mapping = {
            "model.embed_tokens.weight": ("embed_tokens", "embedding"),
            "model.norm.weight": ("norm", "scale"),
            "lm_head.weight": ("lm_head", "kernel"),
        }
        if name in direct_mapping:
            return direct_mapping[name]
        
        patterns = [
            (r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight", 
             lambda m: (f"layers_{m.group(1)}", f"{m.group(2)}_layernorm", "scale")),
            (r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", 
             lambda m: (f"layers_{m.group(1)}", "self_attn", f"{m.group(2)}_proj", "kernel")),
            (r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", 
             lambda m: (f"layers_{m.group(1)}", "mlp", f"{m.group(2)}_proj", "kernel")),
        ]
        
        for pattern, mapper in patterns:
            match = re.match(pattern, name)
            if match:
                return mapper(match)
        return None
    
    # Load weights
    param_dict = {"params": {}}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path is None:
                        continue
                    
                    param = f.get_tensor(key)
                    if param.dtype == np.float16 and dtype == jnp.bfloat16:
                        param = param.astype(np.float32)
                    param = jnp.array(param, dtype=dtype)
                    
                    # Transpose weights (not biases/norms)
                    if "weight" in key and ("proj" in key or "lm_head" in key):
                        param = jnp.transpose(param)
                    
                    # Set param
                    current = param_dict["params"]
                    for p in path[:-1]:
                        current = current.setdefault(p, {})
                    current[path[-1]] = param
    
    # Merge params
    def merge_params(base, new):
        if isinstance(new, dict):
            return {k: merge_params(base.get(k, {}), v) for k, v in new.items()}
        return new if isinstance(new, jnp.ndarray) else base
    
    return merge_params(params, param_dict)

# --- Generation ---
def generate_text(model, params, tokenizer, prompt, max_tokens, temperature=0.7, top_p=0.9, top_k=50):
    """Generate text using the model."""
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
        
        # Simple sampling (temperature only)
        if temperature < 1e-5:
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        else:
            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            next_token = jax.random.categorical(rng_key, logits[:, -1, :] / temperature, axis=-1)
        
        # Update state
        state["input_ids"] = next_token[:, None]
        state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
        state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
        state["past_key_values"] = past_key_values
        
        # Decode and print token
        token = tokenizer.decode(next_token[0])
        print(token, end="", flush=True)
        
        # Check for end of sequence
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    print()  # New line at end

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Load config and model
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    
    # Load tokenizer and weights
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # Generate
    generate_text(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()

if __name__ == "__main__":
    main() 