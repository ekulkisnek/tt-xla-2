#!/usr/bin/env python3
"""
Core model implementation for Qwen2.5-7B with tensor parallelism in JAX/Flax.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
from safetensors import safe_open
import os
import gc
from typing import Dict, Any, Tuple

# Global mesh (set externally in generation scripts)
mesh = None

def setup_device_mesh():
    """Set up device mesh for tensor parallelism."""
    global mesh
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("mp",))

def make_causal_mask(seq_len: int, key_len: int = None) -> jnp.ndarray:
    """Create causal attention mask."""
    if key_len is None:
        key_len = seq_len
    return jnp.tril(jnp.ones((seq_len, key_len))) - 1

def sample_next_token(logits: jnp.ndarray, temperature: float = 0.0) -> jnp.ndarray:
    """Sample next token from logits (greedy if temperature=0)."""
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)[0]  # Extract scalar from batch
    else:
        scaled_logits = logits / temperature
        probs = jax.nn.softmax(scaled_logits, axis=-1)
        return jax.random.categorical(jax.random.PRNGKey(42), probs)[0]  # Extract scalar from batch

# --- Precomputed RoPE ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs).astype(jnp.float32)
    return jnp.exp(1j * freqs)  # Complex for efficient application

def apply_rotary_emb_complex(q, k, freqs_cis):
    half_dim = q.shape[-1] // 2
    
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    
    q_complex = jax.lax.complex(q1.astype(jnp.float32), q2.astype(jnp.float32))
    k_complex = jax.lax.complex(k1.astype(jnp.float32), k2.astype(jnp.float32))
    
    freqs_cis_expanded = freqs_cis[..., None, :]
    q_rot = q_complex * freqs_cis_expanded
    k_rot = k_complex * freqs_cis_expanded
    
    q_rot_real = jnp.concatenate([jnp.real(q_rot), jnp.imag(q_rot)], axis=-1)
    k_rot_real = jnp.concatenate([jnp.real(k_rot), jnp.imag(k_rot)], axis=-1)
    
    return q_rot_real, k_rot_real

# --- Model Code ---
class FlaxQwenPreTrainedModel(nn.Module):
    """Base class for Qwen pretrained models with standard initialization."""
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: Dict = None) -> Dict:
        init_rngs = {"params": rng}
        if params is None:
            params = self.init(init_rngs, jnp.ones(input_shape), return_dict=True)["params"]
        return params
    
    def init_cache(self, batch_size: int, max_length: int) -> Dict:
        cache = {}
        num_kv_heads = self.config.get("num_key_value_heads", self.config["num_attention_heads"])
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        for layer_idx in range(self.config["num_hidden_layers"]):
            cache[f"layers_{layer_idx}"] = {
                "self_attn": {
                    "cached_key": jnp.zeros((batch_size, max_length, num_kv_heads, head_dim), dtype=jnp.bfloat16),
                    "cached_value": jnp.zeros((batch_size, max_length, num_kv_heads, head_dim), dtype=jnp.bfloat16),
                    "cache_index": jnp.array(0, dtype=jnp.int32)
                }
            }
        return cache

class FullyParallelQwenAttention(nn.Module):
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
        self.max_seq_len = c.get("max_position_embeddings", 2048)
        
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
        
        self.freqs_cis = precompute_freqs_cis(self.head_dim, c["max_position_embeddings"], self.rope_theta)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, self.num_kv_heads, self.head_dim)

        if position_ids is not None:
            if position_ids.shape[1] != seq:
                if past_key_value is not None:
                    start_pos = past_key_value[0].shape[1]
                    position_ids = jnp.arange(start_pos, start_pos + seq, dtype=jnp.int32)[None, :]
                else:
                    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
            
            position_ids = jnp.clip(position_ids, 0, len(self.freqs_cis) - 1)
            freqs_cis = self.freqs_cis[position_ids]
            q, k = apply_rotary_emb_complex(q, k, freqs_cis)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            if self.num_heads != self.num_kv_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = jnp.repeat(k, repeat, axis=2)
                v = jnp.repeat(v, repeat, axis=2)
            k_full = jnp.concatenate([past_k, k], axis=1)
            v_full = jnp.concatenate([past_v, v], axis=1)
        else:
            if self.num_heads != self.num_kv_heads:
                repeat = self.num_heads // self.num_kv_heads
                k_full = jnp.repeat(k, repeat, axis=2)
                v_full = jnp.repeat(v, repeat, axis=2)
            else:
                k_full = k
                v_full = v

        q = q.transpose(0, 2, 1, 3)
        k = k_full.transpose(0, 2, 1, 3)
        v = v_full.transpose(0, 2, 1, 3)

        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        if attention_mask is not None:
            scores += attention_mask
        
        probs = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden_size)

        attn_out = attn_out.astype(self.dtype)

        attn_out = self.o_proj(attn_out)

        return attn_out, (k_full, v_full)

class StandardEmbed(nn.Module):
    num_embeddings: int
    features: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    name: str = None

    def setup(self):
        self.embedding = self.param(
            "embedding",
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs):
        embedding = jnp.asarray(self.embedding, self.dtype)
        return embedding[inputs.astype("i4")]

class ParallelDense(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    use_bias: bool = False
    name: str = None

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        in_dim = x.shape[-1]
        out_dim = self.features
        
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), (in_dim, out_dim), self.param_dtype
        )
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (out_dim,), self.param_dtype)
        else:
            bias = None

        def matmul_fn(x, k, b=None):
            local_out = jnp.einsum("bsd,df->bsf", x, k)
            
            if b is not None:
                local_out = local_out + b
            
            full_out = jax.lax.all_gather(local_out, axis_name="mp", axis=0)
            
            result = jnp.reshape(
                jnp.transpose(full_out, (1, 2, 0, 3)), (x.shape[0], x.shape[1], -1)
            )
            return result

        if bias is not None:
            output = shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp"), P("mp",)),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel, bias)
        else:
            output = shard_map(
                matmul_fn,
                mesh=mesh,
                in_specs=(None, P(None, "mp")),
                out_specs=P(None),
                check_rep=False,
            )(x, kernel)
            
        return output

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.intermediate_size = c.get("intermediate_size", 4 * c["hidden_size"])
        self.gate_proj = ParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="gate_proj"
        )
        self.up_proj = ParallelDense(
            self.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="up_proj"
        )
        self.down_proj = ParallelDense(
            c["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.dtype,
            name="down_proj"
        )

    def __call__(self, x):
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="input_layernorm")
        self.self_attn = FullyParallelQwenAttention(config=c, dtype=jnp.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="post_attention_layernorm")
        self.mlp = QwenMLP(config=c, dtype=jnp.bfloat16)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, past_key_value

class Qwen25ForCausalLM(FlaxQwenPreTrainedModel):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        c = self.config
        self.embed_tokens = StandardEmbed(c["vocab_size"], c["hidden_size"], dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="embed_tokens")
        self.layers = [QwenDecoderLayer(config=c, dtype=jnp.bfloat16, name=f"layers_{i}") for i in range(c["num_hidden_layers"])]
        self.norm = nn.RMSNorm(epsilon=c.get("rms_norm_eps", 1e-6), dtype=jnp.bfloat16, name="norm")
        self.lm_head = nn.Dense(
            c["vocab_size"],
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            name="lm_head"
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch, seq = input_ids.shape
        key_len = seq if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + seq

        if attention_mask is None:
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=self.dtype)
        causal_mask = make_causal_mask(seq, key_len)[None, None, :, :]
        attention_bias = jnp.where(attention_mask == 0, -1e9, 0) + causal_mask

        hidden_states = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_key_values = []
        
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, new_kv = layer(hidden_states, attention_bias, position_ids, past_kv)
            new_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {"logits": logits, "past_key_values": new_key_values}
        return logits
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        batch, seq = input_ids.shape
        
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            if past_key_values is None or past_key_values[0] is None:
                position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
            else:
                position_ids = jnp.array([[past_key_values[0][0].shape[1]]], dtype=jnp.int32)
        
        if attention_mask is None:
            key_len = seq if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + seq
            attention_mask = jnp.ones((batch, 1, seq, key_len), dtype=jnp.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
        }
    
    def update_inputs_for_generation(self, model_outputs, **kwargs):
        current_position_ids = kwargs.get("position_ids", None)
        if current_position_ids is not None:
            next_position_ids = current_position_ids[:, -1:] + 1
        else:
            next_position_ids = None
        
        return {
            "past_key_values": model_outputs["past_key_values"],
            "position_ids": next_position_ids,
        }

# --- Weight Loading ---
def get_param_path(name):
    mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    if name in mapping:
        return mapping[name]
    import re
    if m := re.match(r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight", name):
        return (f"layers_{m.group(1)}", f"{m.group(2)}_layernorm", "scale")
    if m := re.match(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)", name):
        return (f"layers_{m.group(1)}", "self_attn", f"{m.group(2)}_proj", "kernel" if m.group(3) == "weight" else "bias")
    if m := re.match(r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", name):
        return (f"layers_{m.group(1)}", "mlp", f"{m.group(2)}_proj", "kernel")
    return None

def transpose_if_needed(name, param):
    if "weight" in name and "layernorm" not in name and "embed_tokens" not in name:
        return param.T
    return param

def load_params(model, model_path, dtype):
    print(f"Loading JAX model weights from {model_path}...")
    params = {"params": {}}
    loaded_count = 0
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, file), framework="numpy") as f:
                for key in f.keys():
                    path = get_param_path(key)
                    if path:
                        param = f.get_tensor(key)
                        param = jnp.array(param, dtype=jnp.bfloat16)
                        param = transpose_if_needed(key, param)
                        d = params["params"]
                        for p in path[:-1]:
                            d = d.setdefault(p, {})
                        d[path[-1]] = param
                        loaded_count += 1
    gc.collect()
    print(f"Weight loading completed. Loaded {loaded_count} parameters.")
    return params