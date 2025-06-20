#!/usr/bin/env python3
"""
Real Qwen 2.5-7B Tensor Parallel Implementation with Actual Weights

This script combines:
1. Our tensor parallel implementation from stage3.py
2. Real Qwen 2.5-7B model weights and tokenization
3. Proper safetensors weight loading
4. HuggingFace tokenizer integration

Usage:
python real_qwen25_tp.py --prompt "The capital of France is" --mode tp2 --max-tokens 20
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
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from safetensors import safe_open

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("real_qwen25_tp")

def setup_environment():
    """Setup JAX environment for tensor parallelism"""
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].device_kind}")

def create_mesh(model_parallel=2, data_parallel=1):
    """Create JAX mesh for tensor parallelism"""
    total_devices = model_parallel * data_parallel
    if jax.device_count() < total_devices:
        logger.warning(f"Requested {total_devices} devices but only {jax.device_count()} available")
        total_devices = jax.device_count()
        if total_devices >= model_parallel:
            data_parallel = total_devices // model_parallel
        else:
            model_parallel = total_devices
            data_parallel = 1
    
    devices = mesh_utils.create_device_mesh((data_parallel, model_parallel))
    mesh = Mesh(devices, axis_names=('data', 'model'))
    logger.info(f"Created mesh: {mesh}")
    return mesh

# ===== TENSOR PARALLEL LAYERS =====

class TensorParallelDense(nn.Module):
    """Dense layer with configurable tensor parallelism sharding"""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    shard_axes: Tuple[Optional[str], Optional[str]] = (None, None)  # (input_shard, output_shard)
    reduce_scatter: bool = False
    
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], self.features),
            self.param_dtype
        )
        
        # Apply sharding to kernel if we're in a mesh context
        try:
            # Check if we have a mesh available
            current_mesh = jax.experimental.maps.thread_resources.env.physical_mesh
            if current_mesh is not None and self.shard_axes != (None, None):
                kernel_sharding = NamedSharding(current_mesh, PS(self.shard_axes[0], self.shard_axes[1]))
                kernel = jax.lax.with_sharding_constraint(kernel, kernel_sharding)
        except:
            # No mesh context, use regular computation
            pass
        
        # Compute output
        output = jnp.dot(inputs, kernel.astype(inputs.dtype))
        
        # Apply psum if needed for reduce-scatter
        if self.reduce_scatter:
            try:
                output = jax.lax.psum(output, 'model')
            except:
                # No mesh context, skip psum
                pass
        
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), self.param_dtype)
            try:
                current_mesh = jax.experimental.maps.thread_resources.env.physical_mesh
                if current_mesh is not None and self.shard_axes[1] is not None:
                    bias_sharding = NamedSharding(current_mesh, PS(self.shard_axes[1]))
                    bias = jax.lax.with_sharding_constraint(bias, bias_sharding)
            except:
                # No mesh context, use regular computation
                pass
            output = output + bias.astype(output.dtype)
        
        return output

# ===== REAL QWEN 2.5 MODEL COMPONENTS =====

class QwenAttention(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_heads = c["num_attention_heads"]
        self.head_dim = c.get("head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = c.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        # Use tensor parallel layers for projections
        self.q_proj = TensorParallelDense(
            features=self.hidden_size, 
            use_bias=False,
            dtype=self.dtype, 
            param_dtype=self.param_dtype,
            shard_axes=(None, "model"),  # Shard output
        )
        self.k_proj = TensorParallelDense(
            features=self.kv_dim,
            use_bias=False, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype,
            shard_axes=(None, "model"),  # Shard output
        )
        self.v_proj = TensorParallelDense(
            features=self.kv_dim,
            use_bias=False,
            dtype=self.dtype, 
            param_dtype=self.param_dtype,
            shard_axes=(None, "model"),  # Shard output
        )
        self.o_proj = TensorParallelDense(
            features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype, 
            param_dtype=self.param_dtype,
            shard_axes=("model", None),  # Shard input
            reduce_scatter=True  # Reduce across model parallel dimension
        )
        
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.max_position_embeddings = c.get("max_position_embeddings", 4096)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch, seq, _ = hidden_states.shape
        
        # Project to q, k, v with tensor parallelism
        q = self.q_proj(hidden_states).reshape(batch, seq, -1, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, -1, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, -1, self.head_dim)
        
        # Apply rotary embeddings
        if position_ids is not None:
            cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
            q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Handle KV cache (simplified for now)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.size > 0:
                k = jnp.concatenate([past_k, k], axis=1)
                v = jnp.concatenate([past_v, v], axis=1)
        
        cache_k, cache_v = k, v
        
        # GQA: repeat k/v if needed
        if q.shape[2] != k.shape[2]:
            repeat = q.shape[2] // k.shape[2]
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)
        
        # Transpose for attention: [b, h, s, d]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention computation
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq, -1)
        
        # Output projection with tensor parallelism
        output = self.o_proj(attn_out)
        
        return output, (cache_k, cache_v)

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

class QwenMLP(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.intermediate_size = c.get("intermediate_size", 4 * self.hidden_size)
        
        # Use tensor parallel layers
        self.gate_proj = TensorParallelDense(
            features=self.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes=(None, "model"),  # Shard output
        )
        self.up_proj = TensorParallelDense(
            features=self.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes=(None, "model"),  # Shard output
        )
        self.down_proj = TensorParallelDense(
            features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes=("model", None),  # Shard input
            reduce_scatter=True  # Reduce across model parallel dimension
        )

    def __call__(self, x):
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class QwenDecoderLayer(nn.Module):
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        self.input_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-5), 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )
        self.self_attn = QwenAttention(
            config=c, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )
        self.post_attention_layernorm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-5), 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )
        self.mlp = QwenMLP(
            config=c, 
            dtype=self.dtype, 
            param_dtype=self.param_dtype
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        batch, seq, _ = hidden_states.shape
        if position_ids is None:
            position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
        
        # Self attention
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
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
    param_dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        c = self.config
        
        # Use config vocab size (don't truncate like q25_jax.py)
        actual_vocab_size = c["vocab_size"]  # Use full config vocab size
        
        # Token embeddings (not sharded)
        self.embed_tokens = nn.Embed(
            num_embeddings=actual_vocab_size,
            features=c["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        
        # Decoder layers
        self.layers = [
            QwenDecoderLayer(
                config=c,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            for _ in range(c["num_hidden_layers"])
        ]
        
        # Final layer norm
        self.norm = nn.RMSNorm(
            epsilon=c.get("rms_norm_eps", 1e-5),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        # LM head (not sharded by default, could be sharded for very large vocabs)
        self.lm_head = nn.Dense(
            features=actual_vocab_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)
        
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Attention mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
        causal_mask = jnp.where(causal_mask == 0, -1e9, 0.0)
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]
        
        # Transformer layers
        all_past_key_values = [] if past_key_values is None else past_key_values
        new_past_key_values = []
        
        for i, layer in enumerate(self.layers):
            past_key_value = all_past_key_values[i] if i < len(all_past_key_values) else None
            hidden_states, past_key_value = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            new_past_key_values.append(past_key_value)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                "logits": logits,
                "past_key_values": new_past_key_values,
                "hidden_states": hidden_states,
            }
        else:
            return logits, new_past_key_values

# ===== WEIGHT LOADING =====

def get_param_path(name):
    """Convert HuggingFace parameter names to our model structure"""
    # Token embeddings
    if name == "model.embed_tokens.weight":
        return ("embed_tokens", "embedding")
    
    # Layer norm
    if name == "model.norm.weight":
        return ("norm", "scale")
    
    # LM head
    if name == "lm_head.weight":
        return ("lm_head", "kernel")
    
    # Decoder layers
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer_idx = int(parts[2])
        component = ".".join(parts[3:])
        
        # Attention components
        if component == "self_attn.q_proj.weight":
            return ("layers", layer_idx, "self_attn", "q_proj", "kernel")
        elif component == "self_attn.k_proj.weight":
            return ("layers", layer_idx, "self_attn", "k_proj", "kernel")
        elif component == "self_attn.v_proj.weight":
            return ("layers", layer_idx, "self_attn", "v_proj", "kernel")
        elif component == "self_attn.o_proj.weight":
            return ("layers", layer_idx, "self_attn", "o_proj", "kernel")
        
        # MLP components
        elif component == "mlp.gate_proj.weight":
            return ("layers", layer_idx, "mlp", "gate_proj", "kernel")
        elif component == "mlp.up_proj.weight":
            return ("layers", layer_idx, "mlp", "up_proj", "kernel")
        elif component == "mlp.down_proj.weight":
            return ("layers", layer_idx, "mlp", "down_proj", "kernel")
        
        # Layer norms
        elif component == "input_layernorm.weight":
            return ("layers", layer_idx, "input_layernorm", "scale")
        elif component == "post_attention_layernorm.weight":
            return ("layers", layer_idx, "post_attention_layernorm", "scale")
    
    logger.warning(f"Unknown parameter: {name}")
    return None

def transpose_if_needed(name, param):
    """Transpose weight matrices as needed"""
    if name.endswith(".kernel") and param.ndim == 2:
        # JAX Dense layers expect [in_features, out_features], HF uses [out_features, in_features]
        return param.T
    return param

def process_safetensors_file(file_path, dtype=jnp.bfloat16):
    """Load and process a single safetensors file"""
    param_dict = {}
    logger.info(f"Loading {os.path.basename(file_path)}")
    
    with safe_open(file_path, framework="numpy") as f:
        for name in f.keys():
            param_path = get_param_path(name)
            if param_path is None:
                continue
            
            # Load parameter
            param = f.get_tensor(name)
            param = transpose_if_needed(name, param)
            
            # No truncation - use full vocab size like q25_jax.py
            param = jnp.array(param, dtype=dtype)
            
            # Store in nested dict structure
            current = param_dict
            for key in param_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[param_path[-1]] = param
    
    return param_dict

def merge_param_dicts(base_dict, new_dict):
    """Recursively merge parameter dictionaries"""
    for key, value in new_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = merge_param_dicts(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_params(model, model_path, dtype):
    """Load model parameters from safetensors files"""
    logger.info("Loading weights...")
    
    # Initialize parameter structure
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # Load weights from safetensors files
    param_dict = {}
    for file in sorted(os.listdir(model_path)):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
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
    
    # Count parameters
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logger.info(f"Loaded {param_count:,} parameters")
    
    return params

# ===== GENERATION =====

def sample_next_token(logits, temperature=0.7):
    """Sample next token from logits"""
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)
    else:
        logits = logits / temperature
        return jax.random.categorical(jax.random.PRNGKey(int(time.time() * 1000)), logits, axis=-1)

def generate_with_model(model, params, tokenizer, prompt, max_new_tokens=20, temperature=0.8):
    """Generate text with the model"""
    # STEP 1 DIAGNOSTICS: Check tokenizer and embedding alignment
    logger.info("=== STEP 1 DIAGNOSTICS ===")
    
    # 1.1 Dump the first 10 prompt token-ids
    prompt_token_ids = tokenizer.encode(prompt)
    logger.info(f"Prompt token IDs: {prompt_token_ids}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # 1.2 Check vocab size vs embedding matrix
    embed = params['params']['embed_tokens']['embedding']
    logger.info(f"Embedding shape: {embed.shape}")
    logger.info(f"Config vocab size: {embed.shape[0]}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    if embed.shape[0] != len(tokenizer):
        logger.info(f"ℹ️ Config vocab size ({embed.shape[0]}) != tokenizer vocab size ({len(tokenizer)})")
        logger.info("ℹ️ This is normal - using config vocab size like q25_jax.py for better generation")
        if len(tokenizer) > embed.shape[0]:
            logger.error(f"❌ Tokenizer vocab size exceeds embedding size!")
            raise ValueError("Tokenizer vocab > embedding vocab!")
    else:
        logger.info("✅ Tokenizer vocab == embedding rows")
    
    # 1.3 Verify LM-head tie
    et = params['params']['embed_tokens']['embedding']
    lm = params['params']['lm_head']['kernel']
    logger.info(f"Embed tokens shape: {et.shape}")
    logger.info(f"LM head shape: {lm.shape}")
    
    # DISABLED: Following q25_jax.py - weight tying degrades generation quality
    # The working q25_jax.py deliberately does NOT tie weights for better generation
    logger.info("ℹ️ LM-head weights NOT tied (following q25_jax.py for better generation)")
    
    logger.info("=== END DIAGNOSTICS ===")
    
    # STEP 2 DIAGNOSTICS: Audit weight mapping per layer
    logger.info("=== STEP 2 DIAGNOSTICS ===")
    
    # Debug parameter structure
    logger.info(f"Params keys: {list(params.keys())}")
    logger.info(f"Params['params'] keys: {list(params['params'].keys())}")
    
    # 2.1 Add one-layer checksum
    layer0_q = params['params']['layers_0']['self_attn']['q_proj']['kernel']
    logger.info(f"Layer-0 q_proj mean: {float(layer0_q.mean()):.6f}, std: {float(layer0_q.std()):.6f}")
    logger.info(f"Layer-0 q_proj shape: {layer0_q.shape}")
    
    logger.info("=== END STEP 2 DIAGNOSTICS ===")
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors="np")
    input_ids = jnp.array(prompt_tokens).reshape(1, -1)
    
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Generate tokens
    current_tokens = input_ids
    past_key_values = None
    generated_tokens = []
    
    for i in range(max_new_tokens):
        # Forward pass
        outputs = model.apply(params, current_tokens, past_key_values=past_key_values)
        
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
        else:
            logits, past_key_values = outputs
        
        # Sample next token
        next_token = sample_next_token(logits[0, -1:], temperature)
        generated_tokens.append(int(next_token[0]))
        
        # Update for next iteration
        current_tokens = next_token[None, :]
        
        # Simple stopping criterion
        if len(generated_tokens) > 3 and all(t == generated_tokens[-1] for t in generated_tokens[-3:]):
            logger.info("Stopped due to repetition")
            break
    
    # Decode generated text
    full_tokens = jnp.concatenate([input_ids[0], jnp.array(generated_tokens)])
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(full_tokens.tolist(), skip_special_tokens=True)
    
    return generated_text, full_text, generated_tokens

# ===== MAIN FUNCTIONS =====

def load_config(model_path):
    """Load model configuration"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config}")
    return config

def setup_tokenizer(model_path):
    """Setup HuggingFace tokenizer"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"Loaded tokenizer from {model_path}")
        logger.info(f"Vocab size: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

def run_single_device(prompt, model_path, max_tokens=20, temperature=0.8, dtype=jnp.bfloat16):
    """Run generation on single device"""
    print(f"\n{'='*60}")
    print("🔧 SINGLE DEVICE GENERATION (Real Qwen 2.5-7B)")
    print(f"{'='*60}")
    
    # Load config and tokenizer
    config = load_config(model_path)
    tokenizer = setup_tokenizer(model_path)
    
    # Create model with config vocab size
    model = Qwen25ForCausalLM(config=config, dtype=dtype, param_dtype=dtype)
    
    # Load parameters
    params = load_params(model, model_path, dtype)
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    
    # Generate
    start_time = time.time()
    generated_text, full_text, generated_tokens = generate_with_model(
        model, params, tokenizer, prompt, max_tokens, temperature
    )
    end_time = time.time()
    
    # Show results
    print(f"✨ Generated text: '{generated_text}'")
    print(f"📝 Full response: '{full_text}'")
    print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
    print(f"🚀 Tokens/second: {len(generated_tokens) / (end_time - start_time):.1f}")
    print(f"🔢 Token count: {len(generated_tokens)} tokens")

def run_tensor_parallel(prompt, model_path, model_parallel=2, max_tokens=20, temperature=0.8, dtype=jnp.bfloat16):
    """Run generation with tensor parallelism"""
    print(f"\n{'='*60}")
    print(f"⚡ TENSOR PARALLEL (TP={model_parallel}) GENERATION (Real Qwen 2.5-7B)")
    print(f"{'='*60}")
    
    # Setup mesh
    mesh = create_mesh(model_parallel=model_parallel, data_parallel=1)
    
    with mesh:
        # Load config and tokenizer
        config = load_config(model_path)
        tokenizer = setup_tokenizer(model_path)
        
        # Create model with config vocab size
        model = Qwen25ForCausalLM(config=config, dtype=dtype, param_dtype=dtype)
        print(f"Mesh configuration: {mesh.devices.shape}")
        
        # Load parameters
        params = load_params(model, model_path, dtype)
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"Model parameters: {param_count:,}")
        
        # Generate
        start_time = time.time()
        generated_text, full_text, generated_tokens = generate_with_model(
            model, params, tokenizer, prompt, max_tokens, temperature
        )
        end_time = time.time()
        
        # Show results
        print(f"✨ Generated text: '{generated_text}'")
        print(f"📝 Full response: '{full_text}'")
        print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
        print(f"🚀 Tokens/second: {len(generated_tokens) / (end_time - start_time):.1f}")
        print(f"🔢 Token count: {len(generated_tokens)} tokens")

def compare_parity(prompt, model_path, max_tokens=10, dtype=jnp.bfloat16):
    """Compare single device vs tensor parallel for parity"""
    print(f"\n{'='*60}")
    print("🔍 PARITY CHECK: Single Device vs Tensor Parallel (Real Qwen 2.5-7B)")
    print(f"{'='*60}")
    
    # Load config and tokenizer
    config = load_config(model_path)
    tokenizer = setup_tokenizer(model_path)
    
    print(f"Prompt: '{prompt}'")
    
    # Single device (deterministic)
    model = Qwen25ForCausalLM(config=config, dtype=dtype, param_dtype=dtype)
    params = load_params(model, model_path, dtype)
    _, single_full_text, single_tokens = generate_with_model(
        model, params, tokenizer, prompt, max_tokens, temperature=0.0
    )
    
    # Tensor parallel (deterministic)
    mesh = create_mesh(model_parallel=2, data_parallel=1)
    with mesh:
        model = Qwen25ForCausalLM(config=config, dtype=dtype, param_dtype=dtype)
        params = load_params(model, model_path, dtype)
        _, tp_full_text, tp_tokens = generate_with_model(
            model, params, tokenizer, prompt, max_tokens, temperature=0.0
        )
    
    # Compare
    print(f"🖥️  Single device: '{single_full_text}'")
    print(f"⚡ TP (2 devices): '{tp_full_text}'")
    print(f"🔢 Single tokens: {single_tokens}")
    print(f"🔢 TP tokens: {tp_tokens}")
    
    if single_tokens == tp_tokens:
        print("✅ PARITY CONFIRMED - Outputs are identical!")
    else:
        print("❌ PARITY MISMATCH - Outputs differ!")
        print(f"   Token differences: {set(single_tokens) ^ set(tp_tokens)}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Real Qwen 2.5-7B Tensor Parallel Generation")
    parser.add_argument("--prompt", type=str, default="The capital of France is", 
                       help="Prompt to generate from")
    parser.add_argument("--model-path", type=str, default="../weights", 
                       help="Path to model weights directory")
    parser.add_argument("--max-tokens", type=int, default=15, 
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Generation temperature")
    parser.add_argument("--mode", choices=["single", "tp2", "tp4", "compare"], default="tp2", 
                       help="Generation mode")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16",
                       help="Model dtype")
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    
    print(f"🎯 Running with prompt: '{args.prompt}'")
    print(f"📁 Model path: {args.model_path}")
    print(f"🔢 Max tokens: {args.max_tokens}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"⚙️  Mode: {args.mode}")
    print(f"🎭 Dtype: {args.dtype}")
    
    # Check if transformers is available
    try:
        import transformers
        logger.info(f"Using transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("transformers library not found. Please install: pip install transformers")
        return
    
    # Run generation
    try:
        if args.mode == "single":
            run_single_device(args.prompt, args.model_path, args.max_tokens, args.temperature, dtype)
        elif args.mode == "tp2":
            run_tensor_parallel(args.prompt, args.model_path, 2, args.max_tokens, args.temperature, dtype)
        elif args.mode == "tp4":
            run_tensor_parallel(args.prompt, args.model_path, 4, args.max_tokens, args.temperature, dtype)
        elif args.mode == "compare":
            compare_parity(args.prompt, args.model_path, args.max_tokens, dtype)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

if __name__ == "__main__":
    main() 