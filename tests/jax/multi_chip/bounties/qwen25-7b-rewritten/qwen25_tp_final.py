#!/usr/bin/env python3
"""
Complete Tensor Parallel Qwen2.5-7B implementation with proper NNX structure
"""
import os
import sys
import time
import json
import gc
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from safetensors import safe_open
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, SingleDeviceSharding
import flax.core.spmd as spmd
import flax.struct
from jax import ShapeDtypeStruct
import re

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tp")

# === DETERMINISTIC MESH FACTORY ===
def create_mesh(model_parallel: int, data_parallel: int = 1):
    """Create a deterministic device mesh for tensor parallelism."""
    devices = jax.devices()
    assert len(devices) >= model_parallel * data_parallel
    mesh = np.array(devices).reshape(data_parallel, model_parallel)
    return Mesh(mesh, ("data", "model"))

# === AXIS DEFINITIONS ===
class Axis(str, Enum):
    BATCH = "batch"
    SEQ = "seq"
    EMBED = "embed"
    HEAD = "head"
    QHEAD = "qhead"
    KVHEAD = "kvhead"
    MLP = "mlp"
    VOCAB = "vocab"

@flax.struct.dataclass
class KVCacheLayer:
    cache_k: jnp.ndarray
    cache_v: jnp.ndarray
    index: jnp.ndarray

    @property
    def max_seqlen(self) -> int:
        return self.cache_k.shape[1]

    @classmethod
    def create(cls, shape, dtype):
        return cls(
            cache_k=jnp.zeros(shape, dtype=dtype),
            cache_v=jnp.zeros(shape, dtype=dtype),
            index=jnp.array(0, dtype=jnp.int32)
        )

    def update(self, k, v):
        batch_size, seq_len, num_heads, head_dim = k.shape
        indices = (0, self.index, 0, 0)
        new_cache_k = jax.lax.dynamic_update_slice(self.cache_k, k, indices)
        new_cache_v = jax.lax.dynamic_update_slice(self.cache_v, v, indices)
        new_index = self.index + seq_len
        return self.replace(cache_k=new_cache_k, cache_v=new_cache_v, index=new_index)

@flax.struct.dataclass
class KVCache:
    layers: list[KVCacheLayer]

    @classmethod
    def create(cls, num_layers, batch_size, max_seqlen, num_kv_heads, head_dim, dtype):
        shape = (batch_size, max_seqlen, num_kv_heads, head_dim)
        return cls([KVCacheLayer.create(shape, dtype) for _ in range(num_layers)])

class RotaryEmbedding(nnx.Module):
    def __init__(self, features: int, length: int, theta: float):
        sin, cos = self._generate_fixed_pos_embedding(features, length, theta)
        self.sin = nnx.Variable(sin)
        self.cos = nnx.Variable(cos)

    def _generate_fixed_pos_embedding(self, features, length, theta):
        fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
        timescale = theta ** fraction
        rotational_frequency = 1.0 / timescale
        sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(length, dtype=jnp.float32), rotational_frequency, precision=jax.lax.Precision.HIGHEST)
        sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
        return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

    def __call__(self, position_ids):
        cos = self.cos.value[position_ids]
        sin = self.sin.value[position_ids]
        return cos, sin

# === ORIGINAL HELPER FUNCTIONS ===
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

# === MODEL COMPONENTS ===
class QwenAttention(nnx.Module):
    def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        
        init = lambda sh: nnx.with_partitioning(nnx.initializers.lecun_normal(), sh)
        
        self.q_proj = nnx.LinearGeneral(
            self.hidden_size, (self.num_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.HEAD, None)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.k_proj = nnx.LinearGeneral(
            self.hidden_size, (self.num_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, None)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.v_proj = nnx.LinearGeneral(
            self.hidden_size, (self.num_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, None)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.o_proj = nnx.LinearGeneral(
            (self.num_heads, self.head_dim), self.hidden_size,
            axis=(-2, -1),
            kernel_init=init((Axis.HEAD, None, Axis.EMBED)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        
        self.rope = RotaryEmbedding(self.head_dim, config["max_position_embeddings"], config.get("rope_theta", 10000.0))

    def __call__(self, x, cache=None, position_ids=None, attention_mask=None):
        if cache is not None:
            return self.decode(x, cache, position_ids, attention_mask)
        else:
            return self.forward(x, position_ids, attention_mask)

    def forward(self, x, position_ids, attention_mask):
        batch, seq, _ = x.shape
        
        # Project and reshape properly
        q = self.q_proj(x)  # (batch, seq, num_heads, head_dim)
        k = self.k_proj(x)  # (batch, seq, num_kv_heads, head_dim) 
        v = self.v_proj(x)  # (batch, seq, num_kv_heads, head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = self.rope(position_ids)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Store cache format
        cache_k = k
        cache_v = v
        
        # GQA: repeat k/v to match query heads if needed
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Attention computation
        scale = 1.0 / jnp.sqrt(self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # [batch, heads, seq, head_dim]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_out)
        return output, KVCacheLayer.create((batch, 0, self.num_kv_heads, self.head_dim), self.dtype).replace(cache_k=cache_k, cache_v=cache_v)

    def decode(self, x, cache, position_ids, attention_mask):
        batch, seq, _ = x.shape
        assert seq == 1, "Decode should only process one token at a time"
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Update cache
        cache = cache.update(k, v)
        
        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = self.rope(position_ids)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Use cached k,v
        k = cache.cache_k[:, :cache.index]
        v = cache.cache_v[:, :cache.index]
        
        # GQA: repeat k/v to match query heads if needed
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Attention computation
        scale = 1.0 / jnp.sqrt(self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_out)
        return output, cache

class QwenMLP(nnx.Module):
    def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config.get("intermediate_size", 4 * self.hidden_size)
        
        init = lambda sh: nnx.with_partitioning(nnx.initializers.lecun_normal(), sh)
        
        self.gate_proj = nnx.Linear(
            self.hidden_size, self.intermediate_size,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            self.hidden_size, self.intermediate_size,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            self.intermediate_size, self.hidden_size,
            kernel_init=init((Axis.MLP, Axis.EMBED)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
    
    def __call__(self, x):
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class QwenDecoderLayer(nnx.Module):
    def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.hidden_size = config['hidden_size']
        
        init = lambda sh: nnx.with_partitioning(nnx.initializers.ones_init(), sh)
        
        self.input_layernorm = nnx.RMSNorm(
            self.hidden_size, 
            epsilon=config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-5)), 
            scale_init=init((Axis.EMBED,)), 
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.self_attn = QwenAttention(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.post_attention_layernorm = nnx.RMSNorm(
            self.hidden_size, 
            epsilon=config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-5)), 
            scale_init=init((Axis.EMBED,)), 
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.mlp = QwenMLP(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, cache = self.self_attn(
            hidden_states, cache=cache, position_ids=position_ids, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        
        return hidden_states, cache

class Qwen25ForCausalLM(nnx.Module):
    def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0)):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        
        init_embed = lambda sh: nnx.with_partitioning(nnx.initializers.lecun_normal(), sh)
        init_norm = lambda sh: nnx.with_partitioning(nnx.initializers.ones_init(), sh)
        
        self.embed_tokens = nnx.Embed(
            num_embeddings=self.vocab_size, features=self.hidden_size,
            embedding_init=init_embed((Axis.VOCAB, Axis.EMBED)),
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.layers = [QwenDecoderLayer(config=config, dtype=dtype, param_dtype=param_dtype, rngs=rngs) for i in range(self.num_layers)]
        self.norm = nnx.RMSNorm(
            self.hidden_size, 
            epsilon=config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-5)), 
            scale_init=init_norm((Axis.EMBED,)), 
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.lm_head = nnx.Linear(
            self.hidden_size, self.vocab_size,
            kernel_init=init_embed((Axis.EMBED, Axis.VOCAB)),
            use_bias=False, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
        batch, seq = input_ids.shape
        
        if past_key_values is not None and len(past_key_values) > 0:
            # Use a fixed cache length to avoid dynamic shapes
            key_len = past_key_values[0].cache_k.shape[1]  # Use max cache length
        else:
            key_len = seq
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, 1, seq), dtype=self.dtype)
        
        causal_mask = make_causal_mask(seq, key_len)
        causal_mask = causal_mask[None, None, :, :]
        attention_bias = (1.0 - attention_mask) * -1e9
        
        if key_len > seq:
            pad_len = key_len - seq
            pad_bias = jnp.zeros((batch, 1, 1, pad_len), dtype=self.dtype)
            attention_bias = jnp.concatenate([pad_bias, attention_bias], axis=-1)
        
        attention_bias = attention_bias + causal_mask
        hidden_states = self.embed_tokens(input_ids)
        
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
        
        new_key_values = []
        for layer, cache in zip(self.layers, past_key_values):
            hidden_states, new_cache = layer(
                hidden_states, attention_mask=attention_bias, 
                position_ids=position_ids, cache=cache
            )
            new_key_values.append(new_cache)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits, "past_key_values": new_key_values}

    def decode(self, input_ids, cache):
        """Decode a single token with KV cache"""
        outputs = self(input_ids, past_key_values=cache.layers)
        return outputs["logits"], cache.replace(layers=outputs["past_key_values"])

    def create_cache(self, batch_size, max_seqlen):
        """Create KV cache for generation"""
        num_kv_heads = self.config.get("num_key_value_heads", self.config["num_attention_heads"])
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        return KVCache.create(self.num_layers, batch_size, max_seqlen, num_kv_heads, head_dim, self.dtype)

    @classmethod
    def load_from_hf_pt_model(cls, model_path, dtype=jnp.float32, param_dtype=jnp.bfloat16, mesh=None, sharding_rules=None):
        config = json.load(open(os.path.join(model_path, 'config.json')))
        abs_model = nnx.eval_shape(lambda: cls(config, dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(0)))
        graphdef = nnx.graphdef(abs_model)
        abs_params = nnx.state(abs_model, nnx.OfType(nnx.Param))

        if sharding_rules and mesh:
            with spmd.logical_axis_rules(sharding_rules):
                pspecs = nnx.get_partition_spec(abs_params)

            def add_sharding(param, p):
                return ShapeDtypeStruct(shape=param.shape, dtype=param.dtype, sharding=NamedSharding(mesh, p))

            abs_params = jax.tree.map(add_sharding, abs_params, pspecs)
        else:
            single = SingleDeviceSharding(jax.devices('cpu')[0])
            abs_params = jax.tree.map(lambda x: ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=single), abs_params)

        @jax.jit
        def non_param():
            model = cls(config, dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(0))
            return nnx.state(model, nnx.Not(nnx.OfType(nnx.Param)))

        non_params = jax.block_until_ready(non_param())
        params = _load_hf_params(abs_params, model_path)
        return nnx.merge(graphdef, non_params, params)

# === PARAMETER LOADING ===
def _get_param_path(name):
    """Map HF parameter names to NNX structure"""
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    if name in direct_mapping:
        return direct_mapping[name]
    
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"
    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
    
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return ("layers", layer_idx, norm_name, "scale")
        
    attn_match = re.match(attention_pattern, name)
    if attn_match:
        layer_idx = int(attn_match.group(1))
        proj_type = attn_match.group(2)
        return ("layers", layer_idx, "self_attn", f"{proj_type}_proj", "kernel")
        
    mlp_match = re.match(mlp_pattern, name)
    if mlp_match:
        layer_idx = int(mlp_match.group(1))
        proj_type = mlp_match.group(2)
        return ("layers", layer_idx, "mlp", f"{proj_type}_proj", "kernel")
        
    return None

def _transpose_if_needed(name, param):
    """Transpose parameters from HF format to JAX format"""
    if "embed_tokens.weight" in name or "norm.weight" in name or "layernorm.weight" in name:
        return param
    if "weight" in name and ("proj" in name or "lm_head" in name):
        return jnp.transpose(param)
    return param

def _load_hf_params(abs_state, model_path):
    """Load parameters from HF safetensors"""
    loaded_params = {}
    
    for file in os.listdir(model_path):
        if file.endswith('.safetensors'):
            with safe_open(os.path.join(model_path, file), framework='numpy') as f:
                for key in f.keys():
                    param_path = _get_param_path(key)
                    if param_path is None:
                        continue
                        
                    param = f.get_tensor(key)
                    param = _transpose_if_needed(key, param)
                    param = jnp.array(param)
                    
                    # Navigate to the right location in the parameter tree
                    current = loaded_params
                    for path_part in param_path[:-1]:
                        if path_part not in current:
                            current[path_part] = {}
                        current = current[path_part]
                    current[param_path[-1]] = param
    
    return loaded_params

# === GENERATION ===
def sample_next_token(logits, temperature=0.7):
    if temperature < 1e-5:
        return jnp.argmax(logits, axis=-1)
    else:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        return jax.random.categorical(rng_key, logits / temperature, axis=-1)

def generate_text(model, tokenizer, prompt, max_tokens, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs["input_ids"])
    
    batch_size = input_ids.shape[0]
    generated = []
    
    for _ in range(max_tokens):
        outputs = model(input_ids)
        logits = outputs["logits"]
        next_token = sample_next_token(logits[:, -1, :], temperature)
        
        generated.append(int(next_token[0]))
        input_ids = jnp.concatenate([input_ids, next_token[:, None]], axis=1)
        
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(inputs["input_ids"][0].tolist() + generated, skip_special_tokens=True)

# === MAIN ===
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B Tensor Parallel")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=16, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    if args.tp == 1:
        logger.info("Single device mode")
        model = Qwen25ForCausalLM.load_from_hf_pt_model(args.model_path, dtype)
    else:
        logger.info(f"Tensor parallel mode with {args.tp} devices")
        mesh = create_mesh(model_parallel=args.tp, data_parallel=args.dp)
        sharding_rules = [
            (str(Axis.EMBED), 'model'), 
            (str(Axis.MLP), 'model'), 
            (str(Axis.HEAD), 'model'),
            (str(Axis.QHEAD), None), 
            (str(Axis.KVHEAD), None), 
            (str(Axis.VOCAB), None)
        ]
        with mesh:
            model = Qwen25ForCausalLM.load_from_hf_pt_model(args.model_path, dtype, mesh=mesh, sharding_rules=sharding_rules)
    
    # Generate
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 