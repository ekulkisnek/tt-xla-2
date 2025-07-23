# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5 inference implemented in Flax NNX.

Can load weights from huggingface model.

Conventions used for axis labeling:

- B: batch
- S: seqlen
- V: vocab
- E: embed
- D: head_dim
- H: num_heads
- HQ: num_q_heads
- K: num_kv_heads

Note: The rotary embedding implementation used here is from flaxformers, which
is same as hugginface transformers'. This is not compatible with
mistral-inference's implementation. The order of the features must be swapped if
using mistral weights. More details in class `RotaryEmbedding`.

Not supported:
- Sliding window

"""

import functools
import os
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import flax.core.spmd
import flax.struct
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from flax.typing import Dtype, Initializer, LogicalRules
from jax import Array, ShapeDtypeStruct
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding
# from jaxtyping import Float, Integer  # Not available, using jax.Array instead
from safetensors import safe_open
from transformers import Qwen2Config
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from .embedding import apply_rotary_embedding, generate_fixed_pos_embedding
from .util import keystr_simple, update_sharding

PARAM_INDEX_FILE = "model.safetensors.index.json"


class Axis(str, Enum):
    EMBED = "embed"
    MLP = "mlp"
    HEAD = "head"
    QHEAD = "qhead"
    KVHEAD = "kvhead"
    VOCAB = "vocab"

    def __str__(self) -> str:
        return self.value


@flax.struct.dataclass
class KVCacheLayer:
    cache_k: Array  # Shape: (B, S, H, D)
    cache_v: Array  # Shape: (B, S, H, D) 
    index: Array    # Shape: ()

    @property
    def max_seqlen(self) -> int:
        return self.cache_k.shape[1]

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: Dtype,
        mesh: Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ) -> "KVCacheLayer":
        assert len(shape) == 4, f"shape should be (B,S,H,D), got: {shape}"

        # KV cache takes the output of K and V projections, which is sharded by
        # KVHEAD, HEAD_DIM.  The KV cache itself should be sharded the same way.
        if sharding_rules is None and mesh is None:
            sharding = SingleDeviceSharding(jax.devices("cpu")[0])
        elif sharding_rules is not None and mesh is not None:
            rules_dict = {k: v for k, v in sharding_rules}
            sharding = NamedSharding(
                mesh,
                PartitionSpec(
                    None, None, rules_dict[Axis.KVHEAD], rules_dict[Axis.HEAD]
                ),
            )
        else:
            raise ValueError("mesh and sharding_rules must be both None or both set")

        return cls(
            cache_k=jnp.zeros(shape, dtype=dtype, device=sharding),
            cache_v=jnp.zeros(shape, dtype=dtype, device=sharding),
            index=jnp.array(0, dtype="int32"),
        )

    def update(
        self, k: Array, v: Array
    ) -> "KVCacheLayer":
        """Update the cache at the given index.

        Can be used for prefill by passing array with seqlen > 1.

        Args:
            k: key array of shape (seqlen, num_kv_heads, head_dim)
            v: value array of same shape

        Returns:
            updated cache layer with seqlen incremented by the amount from input.
        """
        KB, KS, _KH, _KD = k.shape
        VB, VS, _VH, _VD = v.shape
        assert (KB, KS) == (
            VB,
            VS,
        ), f"k and v should have same batch,seqlen: {(KB, KS)} != {(VB,VS)}"
        Z = jnp.array(0, dtype=self.index.dtype)
        return KVCacheLayer(
            cache_k=jax.lax.dynamic_update_slice(
                self.cache_k, k, (Z, self.index, Z, Z)
            ),
            cache_v=jax.lax.dynamic_update_slice(
                self.cache_v, v, (Z, self.index, Z, Z)
            ),
            index=self.index + KS,
        )


@flax.struct.dataclass
class KVCache:
    layers: list[KVCacheLayer]

    @classmethod
    def create(
        cls,
        num_layers: int,
        batch_size: int,
        max_seqlen: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: Dtype,
        mesh: Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ) -> "KVCache":
        shape = (batch_size, max_seqlen, num_kv_heads, head_dim)
        return cls(
            layers=[
                KVCacheLayer.create(shape, dtype, mesh, sharding_rules)
                for _ in range(num_layers)
            ]
        )

    @classmethod
    def create_from_list(cls, past_list, dtype):
        layers = []
        for cache_k, cache_v in past_list:
            assert cache_k.shape == cache_v.shape
            index = jnp.array(cache_k.shape[1])
            layers.append(KVCacheLayer(cache_k, cache_v, index))
        return cls(layers=layers)


def compute_cos_sin_cache(position_ids, head_dim, rope_theta=10000.0):
    pos = jnp.array(position_ids, dtype=jnp.float32)
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


class Attention(nnx.Module):
    """Qwen attention supports different number of Q heads vs KV heads."""

    dim: int
    n_q_heads: int
    head_dim: int
    n_kv_heads: int
    dtype: Any

    wq: nnx.LinearGeneral
    wk: nnx.LinearGeneral
    wv: nnx.LinearGeneral
    wo: nnx.LinearGeneral

    def __init__(
        self,
        dim: int,
        n_q_heads: int,
        head_dim: int,
        n_kv_heads: int,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
        rope_theta: float = 10000.0,
    ):
        self.dim = dim
        self.n_q_heads = n_q_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.dtype = dtype
        self.rope_theta = rope_theta

        init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.wq = nnx.LinearGeneral(
            self.dim,
            (self.n_q_heads, self.head_dim),
            use_bias=False,
            kernel_init=init((Axis.EMBED, Axis.QHEAD, Axis.HEAD)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wk = nnx.LinearGeneral(
            self.dim,
            (self.n_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, Axis.HEAD)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wv = nnx.LinearGeneral(
            self.dim,
            (self.n_kv_heads, self.head_dim),
            kernel_init=init((Axis.EMBED, Axis.KVHEAD, Axis.HEAD)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.wo = nnx.LinearGeneral(
            (self.n_q_heads, self.head_dim),
            self.dim,
            axis=(-2, -1),
            kernel_init=init((Axis.QHEAD, Axis.HEAD, Axis.EMBED)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @property
    def _queries_per_head(self) -> int:
        return self.n_q_heads // self.n_kv_heads

    def __call__(self, x, attention_bias=None, position_ids=None, past_key_value=None):
        B, T, E = x.shape
        xq = self.wq(x).reshape(B, T, self.n_q_heads, self.head_dim)
        xk = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        if position_ids is None:
            position_ids = jnp.arange(T, dtype=jnp.int32)[None, :].repeat(B, axis=0)

        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_embedding(xq, xk, cos, sin)

        if past_key_value is not None:
            cache_k, cache_v = past_key_value
            k = jnp.concatenate([cache_k, xk], axis=1)
            v = jnp.concatenate([cache_v, xv], axis=1)
        else:
            k = xk
            v = xv

        new_past = (k, v)

        # GQA: repeat k/v to match query heads for attention computation
        if self.n_q_heads != self.n_kv_heads:
            repeat = self.n_q_heads // self.n_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Use JAX's dot_product_attention like mistral reference
        out = jax.nn.dot_product_attention(xq, k, v, is_causal=True)
        out = self.wo(out)
        return out, new_past

    def decode(self, x: Array, cache: KVCacheLayer) -> tuple[Array, KVCacheLayer]:
        B, T, _E = x.shape
        assert T == 1, "decode takes one token at a time"

        xq = self.wq(x).reshape(B, T, self.n_q_heads, self.head_dim)
        xk = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE with correct position
        position_ids = jnp.array([[cache.index]])
        cos, sin = compute_cos_sin_cache(position_ids, self.head_dim, self.rope_theta)
        xq, xk = apply_rotary_embedding(xq, xk, cos, sin)
        
        cache = cache.update(xk, xv)
        k = cache.cache_k
        v = cache.cache_v
        
        # Store new cache (before GQA repeat)
        new_past = (k, v)

        # GQA: repeat k/v to match query heads for attention computation
        if self.n_q_heads != self.n_kv_heads:
            repeat = self.n_q_heads // self.n_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        out = jax.nn.dot_product_attention(
            xq,
            k,
            v,
            query_seq_lengths=jnp.array([T], dtype="int32"),
            key_value_seq_lengths=cache.index.reshape(B),
        )
        out = self.wo(out)
        return out, cache


class FeedForward(nnx.Module):
    def __init__(
        self, dim: int, hidden_dim: int, dtype: Any, param_dtype: Dtype, rngs: nnx.Rngs
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.param_dtype = param_dtype

        init = _init_with_sharding(nnx.initializers.lecun_normal())

        self.w1 = nnx.LinearGeneral(
            self.dim,
            self.hidden_dim,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.w2 = nnx.LinearGeneral(
            self.hidden_dim,
            self.dim,
            kernel_init=init((Axis.MLP, Axis.EMBED)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.w3 = nnx.LinearGeneral(
            self.dim,
            self.hidden_dim,
            kernel_init=init((Axis.EMBED, Axis.MLP)),
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        dtype: Any,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
        rope_theta: float = 10000.0,
    ):
        init = _init_with_sharding(nnx.initializers.ones_init())

        self.n_q_heads = n_q_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention_norm = nnx.RMSNorm(
            dim,
            epsilon=norm_eps,
            scale_init=init((Axis.EMBED,)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.ffn_norm = nnx.RMSNorm(
            dim,
            epsilon=norm_eps,
            scale_init=init((Axis.EMBED,)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x, attention_bias=None, position_ids=None, past_key_value=None):
        residual = x
        x = self.attention_norm(x)
        attn_out, new_past = self.attention(x, attention_bias, position_ids, past_key_value)
        x = residual + attn_out
        residual = x
        x = self.ffn_norm(x)
        ff_out = self.feed_forward(x)
        x = residual + ff_out
        return x, new_past

    def decode(self, x, cache):
        residual = x
        x = self.attention_norm(x)
        attn_out, cache = self.attention.decode(x, cache)
        x = residual + attn_out
        residual = x
        x = self.ffn_norm(x)
        ff_out = self.feed_forward(x)
        x = residual + ff_out
        return x, cache


class QwenModel(nnx.Module):
    layers: list[TransformerBlock]
    config: Qwen2Config
    sharding_rules: LogicalRules | None

    def __init__(
        self,
        config: Qwen2Config,
        *,
        dtype: Dtype,
        param_dtype: Dtype,
        rngs: nnx.Rngs,
        sharding_rules: LogicalRules | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.sharding_rules = sharding_rules  # keep track of sharding rules for creating compatible kv cache

        embed_init = _init_with_sharding(nnx.initializers.lecun_normal())
        norm_init = _init_with_sharding(nnx.initializers.zeros_init())
        linear_init = _init_with_sharding(nnx.initializers.lecun_normal())

        head_dim = config.hidden_size // config.num_attention_heads

        self.embed = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=embed_init((Axis.VOCAB, Axis.EMBED)),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.norm = nnx.RMSNorm(
            config.hidden_size,
            scale_init=norm_init((Axis.EMBED,)),
            epsilon=config.rms_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=linear_init((Axis.EMBED, Axis.VOCAB)),
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.layers = [
            TransformerBlock(
                dim=config.hidden_size,
                hidden_dim=config.intermediate_size,
                n_q_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )
            for _ in range(0, config.num_hidden_layers)
        ]

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
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
        def make_causal_mask(q_len, k_len):
            i = jnp.arange(q_len)[:, None]
            j = jnp.arange(k_len)[None, :]
            return (i < j - (k_len - q_len)) * -1e9
        
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
        
        h = self.embed(input_ids)
        new_past = []
        for i, layer in enumerate(self.layers):
            h, new_p = layer(h, attention_bias, position_ids, past_key_values[i] if past_key_values else None)
            new_past.append(new_p)
        h = self.norm(h)
        return h, new_past

    def decode(self, input_ids, cache):
        h = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            h, cache.layers[i] = layer.decode(h, cache.layers[i])
        h = self.norm(h)
        return h, cache

    def create_cache(self, batch_size: int, max_seqlen: int, mesh=None, sharding_rules=None) -> KVCache:
        """Create a KV cache for generation."""
        return KVCache.init(
            self.config.num_hidden_layers,
            batch_size, 
            max_seqlen,
            self.config.num_key_value_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )

    def create_cache(
        self, batch_size: int, max_seqlen: int, mesh: Mesh | None = None
    ) -> KVCache:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        return KVCache.create(
            len(self.layers),
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            mesh=mesh,
            sharding_rules=self.sharding_rules,
        )

    def save_orbax(self, ckpt_dir: Path):
        """Save model parameters with orbax."""
        state = nnx.state(self, nnx.OfType(nnx.Param))
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(ckpt_dir.absolute(), state)
            checkpointer.wait_until_finished()

    @classmethod
    def _load_with(
        cls,
        loader: Callable[[nnx.State], nnx.State],
        config: Qwen2Config,
        dtype: Dtype,
        param_dtype: Dtype,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: LogicalRules | None = None,
    ):
        """Create a model instance using the specific weight loading function.

        Args:
            loader: weight loading function that takes abstract Param state and
                returns loaded Params.
            mesh: if specified with sharding rules, load to assigned devices.
                Otherwise, load to `SingleDeviceSharding(jax.devices("cpu")[0])`.
        """
        abs_model = nnx.eval_shape(
            lambda: cls(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=nnx.Rngs(0),
                sharding_rules=sharding_rules,
            )
        )
        graphdef = nnx.graphdef(abs_model)
        abs_params = nnx.state(abs_model, nnx.OfType(nnx.Param))

        # annotate abstract params with sharding
        if sharding_rules is not None and mesh is not None:
            # this is a bit awkward, can probably be done within eval_shape.
            with flax.core.spmd.logical_axis_rules(sharding_rules):
                pspecs = nnx.get_partition_spec(abs_params)

            def add_sharding(param: ShapeDtypeStruct, p):
                return update_sharding(param, jax.sharding.NamedSharding(mesh, p))

            abs_params = jax.tree.map(add_sharding, abs_params, pspecs)
        elif sharding_rules is None and mesh is None:
            single = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
            abs_params = jax.tree.map(lambda x: update_sharding(x, single), abs_params)
        else:
            raise ValueError(
                "sharding_rules and mesh should both be specified or both None"
            )

        # Initialize non-param states from an actual new instance.
        # The non-returned arrays should be eliminated by jit.
        # This gets things like precomputed rope-embedding constants.
        @jax.jit
        def non_param():
            model = cls(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=nnx.Rngs(0),
                sharding_rules=sharding_rules,
            )
            return nnx.state(model, nnx.Not(nnx.OfType(nnx.Param)))

        non_params = jax.block_until_ready(non_param())
        params = loader(abs_params)

        return nnx.merge(graphdef, non_params, params)

    @classmethod
    def load(
        cls,
        model_dir: Path,
        dtype="float32",
        param_dtype="bfloat16",
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: Sequence[tuple[str, str]] | None = None,
    ) -> "QwenModel":
        """Load converted hf model.

        Args:
            model_dir: path to the pre-converted model.
            dtype: computation dtype
            param_dtype: dtype of the parameters
            mesh: mesh used for sharding, should be set with sharding_rules.
            sharding_rules:
                If set, load the weights with the correct sharding, otherwise,
                weights are loaded as unsharded single device (jax default).
        """
        config = Qwen2Config.from_pretrained(model_dir)
        assert isinstance(config, Qwen2Config)

        return cls._load_with(
            functools.partial(_load_orbax, Path(model_dir) / "orbax"),
            config,
            dtype,
            param_dtype,
            mesh,
            sharding_rules,
        )

    @classmethod
    def load_from_hf_pt_model(
        cls,
        model_name: str,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
        sharding_rules: Sequence[tuple[str, str]] | None = None,
    ) -> "QwenModel":
        """Load model from HF pytorch model.

        Renames, transposes, and reshapes tensors as necessary.

        Args:
            model_name: HF model name
            dtype: computation dtype
            param_dtype: dtype of the parameters
            sharding_rules:
                If set, load the weights with the correct sharding, otherwise,
                weights are loaded as unsharded single device (jax default).

        Returns:
            QwenModel with loaded weights.
        """
        config = Qwen2Config.from_pretrained(model_name)
        assert isinstance(config, Qwen2Config)

        return cls._load_with(
            functools.partial(_load_hf_pt_params, model_name),
            config,
            dtype,
            param_dtype,
            mesh,
            sharding_rules,
        )


class Qwen25ForCausalLM(nnx.Module):
    config: Qwen2Config
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.bfloat16

    def __init__(self, config, dtype, param_dtype, rngs):
        self.model = QwenModel(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (Axis.EMBED, Axis.VOCAB)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, return_dict=True):
        outputs, past_key_values = self.model(input_ids, attention_mask, position_ids, past_key_values)
        logits = self.lm_head(outputs)
        if return_dict:
            return {'logits': logits, 'past_key_values': past_key_values}
        return logits, past_key_values

    def decode(self, input_ids, cache):
        outputs, cache = self.model.decode(input_ids, cache)
        logits = self.lm_head(outputs)
        return logits, cache

    def create_cache(self, batch_size, max_seqlen, mesh=None, sharding_rules=None):
        return self.model.create_cache(batch_size, max_seqlen, mesh, sharding_rules)

    @classmethod
    def load(cls, model_dir: str, dtype: str = "bfloat16", mesh=None, sharding_rules=None):
        """Load model from a directory containing config.json and safetensors files."""
        import json
        from pathlib import Path
        
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        
        # Convert dtype string to JAX dtype
        if dtype == "bfloat16":
            jax_dtype = jnp.bfloat16
        elif dtype == "float32":
            jax_dtype = jnp.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Create model instance
        config_obj = Qwen2Config(**config)
        instance = cls(config_obj, dtype=jax_dtype, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        
        # Load weights using the working safetensors approach
        from safetensors import safe_open
        import re
        import os
        
        def get_param_path(name):
            direct_mapping = {
                "model.embed_tokens.weight": ("embed", "embedding"),
                "model.norm.weight": ("norm", "scale"),
                "lm_head.weight": ("lm_head", "kernel"),
            }
            if name in direct_mapping:
                return direct_mapping[name]
            
            layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
            attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
            mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
            rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
            
            layer_norm_match = re.match(layer_norm_pattern, name)
            if layer_norm_match:
                layer_idx = int(layer_norm_match.group(1))
                norm_type = layer_norm_match.group(2)
                layer_name = f"layers.{layer_idx}"
                norm_name = "input_norm" if norm_type == "input" else "post_attention_norm"
                return (layer_name, norm_name, "scale")
                
            attn_match = re.match(attention_pattern, name)
            if attn_match:
                layer_idx = int(attn_match.group(1))
                proj_type = attn_match.group(2)
                param_type = attn_match.group(3)
                layer_name = f"layers.{layer_idx}"
                proj_name = f"w{proj_type}"
                param_name = "kernel" if param_type == "weight" else "bias"
                return (layer_name, "attention", proj_name, param_name)
                
            mlp_match = re.match(mlp_pattern, name)
            if mlp_match:
                layer_idx = int(mlp_match.group(1))
                proj_type = mlp_match.group(2)
                layer_name = f"layers.{layer_idx}"
                proj_name = f"w{proj_type}"
                return (layer_name, "feed_forward", proj_name, "kernel")
                
            if rotary_pattern.match(name):
                return None
                
            return None

        def transpose_if_needed(name, param):
            if "embed_tokens.weight" in name:
                return param
            if "layernorm.weight" in name or "norm.weight" in name:
                return param  # Don't transpose 1D layer norm weights
            if "weight" in name and ("proj" in name or "lm_head" in name):
                return param.T
            return param
            
        # Initialize with random weights first to get structure
        dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
        
        # Load weights from safetensors
        params_dict = {}
        for file in os.listdir(model_dir):
            if file.endswith(".safetensors"):
                file_path = os.path.join(model_dir, file)
                
                with safe_open(file_path, framework="numpy") as f:
                    for key in f.keys():
                        param_path = get_param_path(key)
                        if param_path is None:
                            continue
                            
                        param = f.get_tensor(key)
                        param = transpose_if_needed(key, param)
                        param = jnp.array(param, dtype=jax_dtype)
                        
                        # Navigate to the right place in the tree
                        current_dict = params_dict
                        for path_part in param_path[:-1]:
                            if path_part not in current_dict:
                                current_dict[path_part] = {}
                            current_dict = current_dict[path_part]
                        current_dict[param_path[-1]] = param
        
        # Update model state with loaded parameters
        def update_state(state_dict, params_dict):
            for key, value in params_dict.items():
                if key in state_dict:
                    if isinstance(value, dict):
                        update_state(state_dict[key], value)
                    else:
                        state_dict[key] = value
        
        # Convert to proper state format
        state = nnx.state(instance)
        update_state(state, params_dict)
        nnx.update(instance, state)
        
        return instance


def _init_with_sharding(
    init_fn: Initializer,
) -> Callable[[tuple[Axis, ...]], Initializer]:
    def init(sharding):
        return nnx.with_partitioning(init_fn, sharding=sharding)
    return init


def _load_orbax(ckpt_dir: Path, abs_state: nnx.State) -> nnx.State:
    """Load model from orbax checkpoint."""
    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(ckpt_dir.absolute(), abs_state)


def _load_hf_pt_params(hf_model: str, abs_state: nnx.State) -> nnx.State:
    """Load model from HF pytorch model.

    Renames, transposes, and reshapes tensors as necessary.

    Args:
        hf_model: HF model name or path.
        abs_state: abstract state of nnx.Params of the model.

    Returns:
        nnx.State with actual param arrays that can be `nnx.merge`d into the
        model.
    """
    PARAM_INDEX_FILE = "model.safetensors.index.json"
    index = cached_file(hf_model, PARAM_INDEX_FILE)
    shard_paths, meta = get_checkpoint_shard_files(hf_model, index)
    assert isinstance(shard_paths, list)

    with ExitStack() as stack:
        shards = {
            os.path.basename(s): stack.enter_context(safe_open(s, "flax"))
            for s in shard_paths
        }

        def transpose_only(param: jax.Array, _abs_param):
            return param.transpose()

        def transpose_reshape(param: jax.Array, abs_param):
            return param.transpose().reshape(abs_param.shape)

        def identity(param: jax.Array, _abs_param):
            return param

        def load_one(path, abs_param):
            # Map to hf pt model (name, need_transpose)
            name_map = {
                "embed/embedding": ("model.embed_tokens.weight", identity),
                "output/kernel": ("lm_head.weight", transpose_only),
                "norm/scale": ("model.norm.weight", identity),
            }
            layer_name_map = {
                "attention/wq/kernel": (
                    "self_attn.q_proj.weight",
                    transpose_reshape,
                ),
                "attention/wk/kernel": (
                    "self_attn.k_proj.weight",
                    transpose_reshape,
                ),
                "attention/wv/kernel": (
                    "self_attn.v_proj.weight",
                    transpose_reshape,
                ),
                "attention/wo/kernel": (
                    "self_attn.o_proj.weight",
                    transpose_reshape,
                ),
                "attention_norm/scale": ("input_layernorm.weight", identity),
                "feed_forward/w1/kernel": ("mlp.gate_proj.weight", transpose_only),
                "feed_forward/w2/kernel": ("mlp.down_proj.weight", transpose_only),
                "feed_forward/w3/kernel": ("mlp.up_proj.weight", transpose_only),
                "ffn_norm/scale": (
                    "post_attention_layernorm.weight",
                    identity,
                ),
            }
            if path[0].key == "layers":
                idx = path[1].key
                layer_name, postprocess = layer_name_map[
                    keystr_simple(path[2:], separator="/")
                ]
                name = f"model.layers.{idx}.{layer_name}"
            else:
                name, postprocess = name_map[keystr_simple(path, separator="/")]

            param = shards[meta["weight_map"][name]].get_tensor(name)
            param = postprocess(param, abs_param)
            assert (
                param.shape == abs_param.shape
            ), f"Wrong shape for {keystr_simple(path, separator='/')}. Expected: {abs_param.shape}, actual: {param.shape}"
            sharding = abs_param.sharding
            assert isinstance(sharding, jax.sharding.Sharding)
            return jax.device_put(param.astype(abs_param.dtype), device=sharding)

        return jax.tree_util.tree_map_with_path(load_one, abs_state)


def convert_hf_model(hf_weights, *, mesh: Mesh | None = None, sharding_rules: LogicalRules | None = None):
    """Convert HuggingFace weights to Qwen NNX format."""
    from safetensors import safe_open
    import re
    import os
    import numpy as np
    
    def get_param_path(name):
        direct_mapping = {
            "model.embed_tokens.weight": ("embed", "embedding"),
            "model.norm.weight": ("norm", "scale"),
            "lm_head.weight": ("lm_head", "kernel"),
        }
        if name in direct_mapping:
            return direct_mapping[name]
        
        layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
        attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
        mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight"
        rotary_pattern = r"model\.layers\.(\d+)\.self_attn\.rotary_emb\..*"
        
        layer_norm_match = re.match(layer_norm_pattern, name)
        if layer_norm_match:
            layer_idx = int(layer_norm_match.group(1))
            norm_type = layer_norm_match.group(2)
            layer_name = f"layers.{layer_idx}"
            norm_name = "input_norm" if norm_type == "input" else "post_attention_norm"
            return (layer_name, norm_name, "scale")
            
        attn_match = re.match(attention_pattern, name)
        if attn_match:
            layer_idx = int(attn_match.group(1))
            proj_type = attn_match.group(2)
            param_type = attn_match.group(3)
            layer_name = f"layers.{layer_idx}"
            proj_name = f"w{proj_type}"
            param_name = "kernel" if param_type == "weight" else "bias"
            return (layer_name, "attention", proj_name, param_name)
            
        mlp_match = re.match(mlp_pattern, name)
        if mlp_match:
            layer_idx = int(mlp_match.group(1))
            proj_type = mlp_match.group(2)
            layer_name = f"layers.{layer_idx}"
            proj_name = f"w{proj_type}"
            return (layer_name, "feed_forward", proj_name, "kernel")
            
        if rotary_pattern.match(name):
            return None
            
        return None

    def transpose_if_needed(name, param):
        if "embed_tokens.weight" in name:
            return param
        if "layernorm.weight" in name or "norm.weight" in name:
            return param  # Don't transpose 1D layer norm weights
        if "weight" in name and ("proj" in name or "lm_head" in name):
            return param.T
        return param

    converted = {}
    
    # If hf_weights is a path to a directory
    if isinstance(hf_weights, (str, os.PathLike)):
        model_dir = hf_weights
        
        for file in os.listdir(model_dir):
            if file.endswith(".safetensors"):
                file_path = os.path.join(model_dir, file)
                
                with safe_open(file_path, framework="numpy") as f:
                    for key in f.keys():
                        param_path = get_param_path(key)
                        if param_path is None:
                            continue
                            
                        param = f.get_tensor(key)
                        param = transpose_if_needed(key, param)
                        param = jnp.array(param)
                        
                        # Navigate to the right place in the tree
                        current_dict = converted
                        for path_part in param_path[:-1]:
                            if path_part not in current_dict:
                                current_dict[path_part] = {}
                            current_dict = current_dict[path_part]
                        current_dict[param_path[-1]] = param
    else:
        # Assume it's already a dict of weights
        for key, value in hf_weights.items():
            param_path = get_param_path(key)
            if param_path is None:
                continue
                
            param = transpose_if_needed(key, value)
            param = jnp.array(param)
            
            # Navigate to the right place in the tree
            current_dict = converted
            for path_part in param_path[:-1]:
                if path_part not in current_dict:
                    current_dict[path_part] = {}
                current_dict = current_dict[path_part]
            current_dict[param_path[-1]] = param
    
    return converted 