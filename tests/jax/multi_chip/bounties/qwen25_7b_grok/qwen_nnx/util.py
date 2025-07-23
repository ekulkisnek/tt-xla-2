# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.typing import LogicalRules
from jax.sharding import NamedSharding, PartitionSpec, Mesh
from flax.core import spmd
from jax.experimental import shard_map
from contextlib import contextmanager


def keystr_simple(path: Sequence, separator: str = '.') -> str:
    parts = []
    for k in path:
        if hasattr(k, 'key'):
            parts.append(str(k.key))
        else:
            parts.append(str(k))
    # Filter out '.value' parts
    parts = [p for p in parts if p != '.value']
    return separator.join(parts)


def update_sharding(
    param,  # jax.Array or jax.ShapeDtypeStruct
    sharding: jax.sharding.Sharding,
):
    if isinstance(param, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(
            param.shape, param.dtype, sharding=sharding, weak_type=param.weak_type
        )
    return jax.device_put(param, device=sharding)


@contextmanager
def timer(name: str):
    import time
    start = time.time()
    yield
    print(f"{name} took {time.time() - start:.2f}s")


def sample_top_p(probs, p: float, rngs: nnx.Rngs):
    """Sample from the given probabilities with top-p sampling."""
    # Ensure probs is 1D for sampling
    if probs.ndim > 1:
        probs = probs.squeeze()
    
    vocab_size = probs.shape[0]
    sorted_probs, sorted_indices = jax.lax.sort_key_val(probs, jnp.arange(vocab_size), is_stable=False)
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs < (1 - p)
    top_prob = sorted_probs[-1] * (cumulative_probs[-1] < (1 - p))
    probs = jnp.where(mask, sorted_probs, 0.0) + top_prob
    probs /= probs.sum()
    next_token = jax.random.choice(rngs.default(), sorted_indices, shape=(), p=probs)
    return next_token


def generate(
    input_ids,
    model: nnx.Module,
    max_tokens: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.9,
    rngs = None,
    mesh = None,
    sharding_rules = None,
):
    if rngs is None:
        rngs = nnx.Rngs(0)

    batch_size, prompt_len = input_ids.shape
    max_seqlen = prompt_len + max_tokens

    cache = model.create_cache(batch_size, max_seqlen, mesh, sharding_rules)

    # prefill
    logits, cache = model.decode(input_ids, cache)
    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    tokens = jnp.concatenate([input_ids, next_token], axis=-1)

    # generate
    for _ in range(1, max_tokens):
        logits, cache = model.decode(next_token, cache)
        probs = jax.nn.softmax(logits / temperature, axis=-1)[:, 0, :]
        next_token = sample_top_p(probs, top_p, rngs)
        next_token = next_token[None, None]
        tokens = jnp.concatenate([tokens, next_token], axis=-1)

    return tokens 