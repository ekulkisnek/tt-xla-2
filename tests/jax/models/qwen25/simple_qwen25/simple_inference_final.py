#!/usr/bin/env python3
"""
Final optimized Qwen2.5-7B inference script for JAX.
Reduced from 814 to ~250 lines while maintaining functionality.

Usage: python simple_inference_final.py --model_path ../weights --prompt "Hello" --max_tokens 20
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

# --- RoPE helpers ---
def compute_rope(position_ids, head_dim, theta=10000.0):
    pos = np.array(position_ids)
    if pos.ndim == 1:
        pos = pos[None, :]
    freqs = 1.0 / (theta ** (np.arange(0, head_dim // 2) / (head_dim // 2)))
    t = np.einsum('bi,j->bij', pos, freqs)
    cos = jnp.repeat(jnp.array(np.cos(t)), 2, axis=-1)
    sin = jnp.repeat(jnp.array(np.sin(t)), 2, axis=-1)
    return cos, sin

def apply_rope(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return jnp.concatenate([-x2, x1], axis=-1)
    
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)

def causal_mask(q_len, k_len):
    i, j = jnp.arange(q_len)[:, None], jnp.arange(k_len)[None, :]
    return jnp.where(i < j - (k_len - q_len), -jnp.inf, 0.0)

# --- Model ---
class QwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        c = self.config
        self.n_heads = c["num_attention_heads"]
        self.n_kv_heads = c.get("num_key_value_heads", self.n_heads)
        self.head_dim = c["hidden_size"] // self.n_heads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        
        for name, dim in [("q_proj", self.n_heads * self.head_dim), 
                         ("k_proj", self.n_kv_heads * self.head_dim),
                         ("v_proj", self.n_kv_heads * self.head_dim),
                         ("o_proj", c["hidden_size"])]:
            setattr(self, name, nn.Dense(dim, use_bias=False, dtype=self.dtype, name=name))

    def __call__(self, x, mask=None, position_ids=None, cache=None):
        B, L, _ = x.shape
        
        # QKV projections and reshape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        
        # RoPE
        if position_ids is not None:
            cos, sin = compute_rope(position_ids, self.head_dim)
            q, k = apply_rope(q, k, cos, sin)
        
        # KV cache
        if cache is not None:
            k = jnp.concatenate([cache[0], k], axis=1)
            v = jnp.concatenate([cache[1], v], axis=1)
        
        cache_kv = (k, v)
        
        # GQA: repeat k,v for attention
        if self.n_heads != self.n_kv_heads:
            k = jnp.repeat(k, self.n_heads // self.n_kv_heads, axis=2)
            v = jnp.repeat(v, self.n_heads // self.n_kv_heads, axis=2)
        
        # Attention computation
        q, k, v = [t.transpose(0, 2, 1, 3) for t in (q, k, v)]  # -> (B, H, L, D)
        scores = jnp.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            scores += mask
        
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bhij,bhjd->bhid', attn, v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        return self.o_proj(out), cache_kv

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        h, i = self.config["hidden_size"], self.config.get("intermediate_size", 4 * self.config["hidden_size"])
        self.gate_proj = nn.Dense(i, use_bias=False, dtype=self.dtype, name="gate_proj")
        self.up_proj = nn.Dense(i, use_bias=False, dtype=self.dtype, name="up_proj")
        self.down_proj = nn.Dense(h, use_bias=False, dtype=self.dtype, name="down_proj")
    
    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        eps = self.config.get("rms_norm_eps", 1e-5)
        self.input_layernorm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="input_layernorm")
        self.self_attn = QwenAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="post_attention_layernorm")
        self.mlp = QwenMLP(self.config, dtype=self.dtype)

    def __call__(self, x, mask=None, position_ids=None, cache=None):
        # Attention block
        h, cache = self.self_attn(self.input_layernorm(x), mask, position_ids, cache)
        x = x + h
        
        # MLP block
        x = x + self.mlp(self.post_attention_layernorm(x))
        
        return x, cache

class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        c = self.config
        eps = c.get("rms_norm_eps", 1e-5)
        self.embed_tokens = nn.Embed(c["vocab_size"], c["hidden_size"], dtype=self.dtype, name="embed_tokens")
        self.layers = [QwenLayer(c, dtype=self.dtype, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=eps, dtype=self.dtype, name="norm")
        self.lm_head = nn.Dense(c["vocab_size"], use_bias=False, dtype=self.dtype, name="lm_head")

    def __call__(self, input_ids, past_key_values=None):
        B, L = input_ids.shape
        
        # Compute attention mask
        past_len = past_key_values[0][0].shape[1] if past_key_values and past_key_values[0] else 0
        key_len = past_len + L
        mask = causal_mask(L, key_len)[None, None, :, :]
        
        # Extend mask for past context if needed
        if past_len > 0:
            pad_mask = jnp.zeros((B, 1, 1, past_len), dtype=self.dtype)
            mask = jnp.concatenate([pad_mask, mask], axis=-1)
        
        # Forward pass
        x = self.embed_tokens(input_ids)
        past_key_values = past_key_values or [None] * len(self.layers)
        
        new_kvs = []
        for layer, past_kv in zip(self.layers, past_key_values):
            x, new_kv = layer(x, mask, None, past_kv)
            new_kvs.append(new_kv)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return {"logits": logits, "past_key_values": new_kvs}

# --- Weight loading ---
def get_param_mapping():
    base = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    patterns = [
        (r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight", 
         lambda m: (f"layers_{m.group(1)}", f"{m.group(2)}_layernorm", "scale")),
        (r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", 
         lambda m: (f"layers_{m.group(1)}", "self_attn", f"{m.group(2)}_proj", "kernel")),
        (r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", 
         lambda m: (f"layers_{m.group(1)}", "mlp", f"{m.group(2)}_proj", "kernel")),
    ]
    
    return base, patterns

def load_params(model, model_path, dtype):
    # Initialize
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1), dtype=jnp.int32))
    base_mappings, pattern_mappings = get_param_mapping()
    
    # Load and map weights
    for file in os.listdir(model_path):
        if not file.endswith(".safetensors"):
            continue
            
        with safe_open(os.path.join(model_path, file), framework="numpy") as f:
            for key in f.keys():
                # Get mapping
                if key in base_mappings:
                    path = base_mappings[key]
                else:
                    path = None
                    for pattern, mapper in pattern_mappings:
                        match = re.match(pattern, key)
                        if match:
                            path = mapper(match)
                            break
                
                if not path:
                    continue
                
                # Load and process weight
                weight = jnp.array(f.get_tensor(key), dtype=dtype)
                if "proj" in key or "lm_head" in key:
                    weight = weight.T
                
                # Set parameter
                current = params["params"]
                for p in path[:-1]:
                    current = current[p]
                current[path[-1]] = weight
    
    return params

# --- Generation ---
def generate(model, params, tokenizer, prompt, max_tokens=50, temperature=0.7):
    tokens = jnp.array(tokenizer(prompt, return_tensors="np")["input_ids"])
    past_key_values = None
    
    print(prompt, end="", flush=True)
    
    for _ in range(max_tokens):
        outputs = model.apply(params, input_ids=tokens, past_key_values=past_key_values)
        logits, past_key_values = outputs["logits"], outputs["past_key_values"]
        
        # Sample
        if temperature < 1e-5:
            next_token = jnp.argmax(logits[0, -1])
        else:
            rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            next_token = jax.random.categorical(rng, logits[0, -1] / temperature)
        
        # Output
        print(tokenizer.decode([int(next_token)]), end="", flush=True)
        tokens = jnp.array([[int(next_token)]])
        
        if int(next_token) == tokenizer.eos_token_id:
            break
    
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Load everything
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    # Run
    generate(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()

if __name__ == "__main__":
    main() 