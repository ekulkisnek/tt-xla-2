# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import functools
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Integer
from typing import Sequence

from .model import QwenModel, KVCache


def sample_top_p(logits: Float[jax.Array, "V"], temp: float, top_p: float, key: nnx.Rngs) -> Integer[jax.Array, ""]:
    if temp != 1.0:
        logits /= temp

    probs = jax.nn.softmax(logits)
    cum_probs = jnp.cumsum(jax.nn.softmax(logits), axis=-1)

    # mask out tokens with prob < top_p
    mask = cum_probs < (1 - top_p)
    logits = jnp.where(mask, -jnp.inf, logits)

    return jax.random.categorical(key(), logits, axis=-1)


class Generator:
    """Token generator using the Qwen model.

    - Holds on to a jitted function for running the model.
    - Instantiates and uses KV cache to do incremental decoding.
    """

    def __init__(self, model: QwenModel, max_seqlen: int):
        """
        Args:
            model: Qwen model.
            max_seqlen: Maximum sequence length (input + max_tokens). This
                controls the size of the allocated KV cache.

        Note: max_seqlen is fixed at startup to prevent excessive jitting with
        different kv cache sizes.
        """
        self.model = model
        self.max_seqlen = max_seqlen

        # Use jax.jit with pre-split model to avoid nnx.jit's cpu and memory
        # overhead.
        self.graphdef, self.state = nnx.split(self.model)
        self._jit_decode = jax.jit(
            self._jit_decode_impl,
            donate_argnames=("cache",),
        )

        # Use nnx jit for this one to handle rngs, which is simple enough.
        self._jit_sample_top_p = nnx.jit(sample_top_p)

    @staticmethod
    def _jit_decode_impl(graphdef, state, input, cache):
        model = nnx.merge(graphdef, state)
        logits, cache = model.decode(input, cache)
        return logits, cache

    def generate(
        self,
        tokens: Integer[jax.Array, "S"],
        rngs: nnx.Rngs,
        max_tokens: int,
        temp: float = 1.0,
        top_p: float = 1.0,
        mesh: jax.sharding.Mesh | None = None,
    ) -> Sequence[int]:
        """Generate tokens.

        Args:
            tokens: Pre-tokenized input tokens as 1D array.
            rngs: Random number generator.
            max_tokens: Maximum number of tokens to generate.
            temp: Sampling temperature.
            top_p: Top-p sampling parameter.
            mesh: Optional mesh for sharding.

        Returns:
            List of generated token ids.
        """
        batch_size = 1
        cache = self.model.create_cache(batch_size, self.max_seqlen, mesh)

        # Prefill
        logits, kvs = self.model(tokens[None], return_kv=True)
        for i in range(len(kvs)):
            k, v = kvs[i]
            cache.layers[i] = cache.layers[i].update(k, v)
        next_token = self._jit_sample_top_p(logits[0, -1], temp, top_p, rngs)

        result = [next_token.item()]

        for _ in range(1, max_tokens):
            logits, cache = self._jit_decode(self.graphdef, self.state, next_token[None, None], cache)
            next_token = self._jit_sample_top_p(logits[0, 0], temp, top_p, rngs)
            result.append(next_token.item())

        return result 