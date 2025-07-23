# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import functools
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

def rotate_half(x: Float[Array, "... D"]) -> Float[Array, "... D"]:
    """rotate_half from flaxformer rotary embedding."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


@functools.partial(jax.jit, static_argnums=(4,))
def apply_rotary_embedding(q, k, cos, sin, decode=False, rotary_index=None):
    """Helper function to apply Rotary Embeddings."""

    if len(k.shape) == 3:
        # for multi query attention
        k = jnp.expand_dims(k, 2)
        multiquery = True
    else:
        multiquery = False

    batch, qlen, qheads, d = q.shape
    kbatch, klen, kheads, kd = k.shape
    assert batch == kbatch, f"{batch} != {kbatch}"
    assert d == kd, f"{d} != {kd}"

    # cos: [len, d]
    # sin: [len, d]
    # rotary_index: [batch]

    if decode and qlen == 1 and rotary_index is not None:
        # we check qlen == 1 so that we don't do this when initializing cache.
        qcos = cos[rotary_index, :]
        qsin = sin[rotary_index, :]
        # qcos, qsin: [batch, d]
        qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (0, 3))
        qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (0, 3))
        # qcos, qsin: [batch, qlen, qheads, d]
    else:
        qcos, qsin = cos[:qlen, :], sin[:qlen, :]
        # qcos, qsin: [qlen, d]
        qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (1, 3))
        qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (1, 3))
        # qcos, qsin: [batch, qlen, qheads, d]

    kcos, ksin = cos[:klen, :], sin[:klen, :]
    # kcos, ksin: [klen, d]
    kcos = jax.lax.broadcast_in_dim(kcos, (batch, klen, kheads, d), (1, 3))
    ksin = jax.lax.broadcast_in_dim(ksin, (batch, klen, kheads, d), (1, 3))
    # kcos, ksin: [batch, klen, kheads, d]

    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)
    if multiquery:
        out_k = jnp.squeeze(out_k, 2)
    return out_q, out_k


def generate_fixed_pos_embedding(features: int, length: int, theta: float = 10000.0) -> tuple[Array, Array]:
    """Generate fixed position embeddings (sin, cos) for rotary."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, features, 2, dtype="float32") / features))
    t = jnp.arange(length, dtype="float32")
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.sin(emb), jnp.cos(emb) 