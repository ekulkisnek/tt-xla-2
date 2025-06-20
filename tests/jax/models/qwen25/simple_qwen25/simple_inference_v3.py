#!/usr/bin/env python3
"""
Ultra-minimal Qwen2.5-7B inference script for JAX.
Usage: python simple_inference_v3.py --model_path ../weights --prompt "Hello" --max_tokens 20
"""
import os
import json
import gc
import argparse
import time
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open

# --- Helpers ---
def rope_freqs(seq_len, head_dim, theta=10000.0):
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim // 2) / (head_dim // 2)))
    t = jnp.outer(pos, freqs)
    cos = jnp.cos(t)
    sin = jnp.sin(t)
    return jnp.repeat(cos, 2, axis=-1), jnp.repeat(sin, 2, axis=-1)

def apply_rope(x, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return jnp.concatenate([-x2, x1], axis=-1)
    
    cos = cos[:x.shape[1]][None, :, None, :]
    sin = sin[:x.shape[1]][None, :, None, :]
    return x * cos + rotate_half(x) * sin

def causal_mask(seq_len, key_len):
    mask = jnp.triu(jnp.ones((seq_len, key_len)), k=key_len - seq_len + 1)
    return jnp.where(mask, -jnp.inf, 0.0)

# --- Model ---
class Attention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        c = self.config
        self.n_heads = c["num_attention_heads"]
        self.n_kv_heads = c.get("num_key_value_heads", self.n_heads)
        self.head_dim = c["hidden_size"] // self.n_heads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        
        self.wq = nn.Dense(self.n_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.wk = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.wv = nn.Dense(self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.dtype)
        self.wo = nn.Dense(c["hidden_size"], use_bias=False, dtype=self.dtype)
        
        # Pre-compute RoPE frequencies
        self.cos, self.sin = rope_freqs(c.get("max_position_embeddings", 4096), self.head_dim)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        
        # QKV projections
        q = self.wq(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        
        # Handle cache
        if cache is not None:
            k_cache, v_cache = cache
            k = jnp.concatenate([k_cache, k], axis=1)
            v = jnp.concatenate([v_cache, v], axis=1)
        
        # GQA: repeat k,v to match q
        if self.n_heads != self.n_kv_heads:
            k = jnp.repeat(k, self.n_heads // self.n_kv_heads, axis=2)
            v = jnp.repeat(v, self.n_heads // self.n_kv_heads, axis=2)
        
        # Attention
        q, k, v = map(lambda t: t.transpose(0, 2, 1, 3), (q, k, v))  # (B, H, L, D)
        scores = jnp.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            scores += mask
        
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        return self.wo(out), (k[:, :, :self.n_kv_heads].transpose(0, 2, 1, 3), 
                             v[:, :, :self.n_kv_heads].transpose(0, 2, 1, 3))

class MLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        hidden_size = self.config["hidden_size"]
        intermediate_size = self.config.get("intermediate_size", 4 * hidden_size)
        self.w1 = nn.Dense(intermediate_size, use_bias=False, dtype=self.dtype)  # gate
        self.w2 = nn.Dense(hidden_size, use_bias=False, dtype=self.dtype)       # down
        self.w3 = nn.Dense(intermediate_size, use_bias=False, dtype=self.dtype)  # up
    
    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        eps = self.config.get("rms_norm_eps", 1e-5)
        self.attn_norm = nn.RMSNorm(epsilon=eps, dtype=self.dtype)
        self.attn = Attention(self.config, dtype=self.dtype)
        self.mlp_norm = nn.RMSNorm(epsilon=eps, dtype=self.dtype)
        self.mlp = MLP(self.config, dtype=self.dtype)
    
    def __call__(self, x, mask=None, cache=None):
        # Attention
        h, cache = self.attn(self.attn_norm(x), mask, cache)
        x = x + h
        
        # MLP
        x = x + self.mlp(self.mlp_norm(x))
        
        return x, cache

class Qwen(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        c = self.config
        self.embedding = nn.Embed(c["vocab_size"], c["hidden_size"], dtype=self.dtype)
        self.blocks = [Block(c, dtype=self.dtype) for _ in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-5), dtype=self.dtype)
        self.lm_head = nn.Dense(c["vocab_size"], use_bias=False, dtype=self.dtype)
    
    def __call__(self, tokens, cache=None):
        x = self.embedding(tokens)
        B, L = tokens.shape
        
        # Create causal mask
        if cache and cache[0] is not None:
            past_len = cache[0][0].shape[2]  # k_cache shape: (B, H, past_L, D)
            mask = causal_mask(L, past_len + L)[None, None, :, :]
        else:
            mask = causal_mask(L, L)[None, None, :, :]
            cache = [None] * len(self.blocks)
        
        # Forward through blocks
        new_cache = []
        for block, block_cache in zip(self.blocks, cache):
            x, new_block_cache = block(x, mask, block_cache)
            new_cache.append(new_block_cache)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_cache

# --- Weight loading ---
def load_weights(model, model_path, dtype):
    dummy_tokens = jnp.ones((1, 1), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_tokens)
    
    # Load safetensors
    weights = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for k in f.keys():
                    weights[k] = jnp.array(f.get_tensor(k), dtype=dtype)
    
    # Map to model structure
    def update_param(path, param):
        parts = path.split('.')
        current = params
        for part in parts[:-1]:
            if part.isdigit():
                part = int(part)
            current = current[part] if isinstance(current, dict) else current[part]
        
        key = parts[-1]
        if key.isdigit():
            key = int(key)
        
        if isinstance(current, dict):
            current[key] = param
        else:
            current = current.at[key].set(param)
    
    # Simple mapping
    mappings = {
        "model.embed_tokens.weight": ("params", "embedding", "embedding"),
        "model.norm.weight": ("params", "norm", "scale"),
        "lm_head.weight": ("params", "lm_head", "kernel"),
    }
    
    for orig_key, weight in weights.items():
        if orig_key in mappings:
            path = mappings[orig_key]
            current = params
            for p in path[:-1]:
                current = current[p]
            
            # Handle transpose for lm_head
            if "lm_head" in orig_key:
                weight = weight.T
            current[path[-1]] = weight
        
        # Layer weights
        elif "layers." in orig_key:
            parts = orig_key.split('.')
            layer_idx = int(parts[2])
            
            if "input_layernorm.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["attn_norm"]["scale"] = weight
            elif "post_attention_layernorm.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["mlp_norm"]["scale"] = weight
            elif "self_attn.q_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["attn"]["wq"]["kernel"] = weight.T
            elif "self_attn.k_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["attn"]["wk"]["kernel"] = weight.T
            elif "self_attn.v_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["attn"]["wv"]["kernel"] = weight.T
            elif "self_attn.o_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["attn"]["wo"]["kernel"] = weight.T
            elif "mlp.gate_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["mlp"]["w1"]["kernel"] = weight.T
            elif "mlp.up_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["mlp"]["w3"]["kernel"] = weight.T
            elif "mlp.down_proj.weight" in orig_key:
                params["params"]["blocks"][layer_idx]["mlp"]["w2"]["kernel"] = weight.T
    
    return params

# --- Generation ---
def generate(model, params, tokenizer, prompt, max_tokens=50, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="np")
    tokens = jnp.array(inputs["input_ids"])
    
    cache = None
    print(prompt, end="", flush=True)
    
    for _ in range(max_tokens):
        logits, cache = model.apply(params, tokens, cache)
        
        # Sample next token
        if temperature < 1e-5:
            next_token = jnp.argmax(logits[0, -1])
        else:
            rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            next_token = jax.random.categorical(rng, logits[0, -1] / temperature)
        
        # Decode and print
        token_text = tokenizer.decode([int(next_token)])
        print(token_text, end="", flush=True)
        
        # Update tokens for next iteration
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
    
    # Load
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    
    model = Qwen(config=config, dtype=dtype)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_weights(model, args.model_path, dtype)
    
    # Generate
    generate(model, params, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    # Cleanup
    del params, model
    gc.collect()
    jax.clear_caches()

if __name__ == "__main__":
    main() 