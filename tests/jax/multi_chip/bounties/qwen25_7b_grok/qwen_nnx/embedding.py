# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import jax
import jax.numpy as jnp
# from jaxtyping import Float, jaxtyping  # Not available

from flax import nnx
from flax.typing import Dtype


def _apply_rotary_embedding(
    x: jax.Array,  # Shape: (B, S, H, D)
    sin: jax.Array,  # Shape: (S, D)
    cos: jax.Array,  # Shape: (S, D)
    rotary_ndims: int = 1,
) -> jax.Array:
    assert rotary_ndims == 1, "only supports 1 rotary dim"
    assert x.shape[-1] % 2 == 0, "features must be even"
    
    # Reshape to match x dimensions: add None for heads
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    
    return (x * cos) + (rotate_half(x) * sin)


def _apply_rotary_embedding_decode(
    x: jax.Array,  # Shape: (B, S, H, D)
    sin: jax.Array,  # Shape: (S, D)
    cos: jax.Array,  # Shape: (S, D)
    rotary_ndims: int = 1,
    rotary_index: jax.Array | None = None,
) -> jax.Array:
    assert rotary_ndims == 1, "only supports 1 rotary dim"
    assert x.shape[-1] % 2 == 0, "features must be even"
    assert x.shape[1] == 1, f"only supports 1 token at a time, got shape {x.shape}"
    assert rotary_index is not None, "rotary_index must be specified for decode"
    
    # For decode, use the specific position indicated by rotary_index
    pos = rotary_index[0]
    cos = cos[pos:pos+1, :]  # Take single position
    sin = sin[pos:pos+1, :]
    
    # Reshape to match x dimensions: add None for heads
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_embedding(
    q: jax.Array,  # Shape: (B, S, H, D)
    k: jax.Array,  # Shape: (B, S, H, D)
    cos: jax.Array,  # Shape: (S, D/2)
    sin: jax.Array,  # Shape: (S, D/2)
    decode: bool = False,
    rotary_ndims: int = 1,
    rotary_index: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    if decode:
        return (
            _apply_rotary_embedding_decode(q, sin, cos, rotary_ndims, rotary_index),
            _apply_rotary_embedding_decode(k, sin, cos, rotary_ndims, rotary_index),
        )
    return (
        _apply_rotary_embedding(q, sin, cos, rotary_ndims),
        _apply_rotary_embedding(k, sin, cos, rotary_ndims),
    )


def generate_fixed_pos_embedding(
    features: int, length: int, dtype: Dtype = jnp.float32, max_timescale: float = 10000
) -> tuple[jax.Array, jax.Array]:  # Shape: (S, D/2), (S, D/2)
    """Generate fixed positional embedding."""
    half_feat = features // 2
    exponents = jnp.linspace(0, 1, half_feat)
    timescale = max_timescale**exponents
    positions = jnp.arange(0, length)[:, None]
    angle_rads = positions / timescale[None, :]
    sin_vals = jnp.sin(angle_rads)
    cos_vals = jnp.cos(angle_rads)
    sin_vals = jnp.repeat(sin_vals, 2, axis=-1)
    cos_vals = jnp.repeat(cos_vals, 2, axis=-1)
    return sin_vals.astype(dtype), cos_vals.astype(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(q, k, cos, sin):
    cos = cos[..., None, :]  # Match dimensions
    sin = sin[..., None, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin) 