#!/usr/bin/env python3
"""
Step 3: Basic token generation
"""

import jax
import jax.numpy as jnp
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_step3")

def sample_next_token(
    logits: jnp.ndarray, 
    temperature: float = 0.7, 
    top_k: int = 50, 
    top_p: float = 0.9,
    rng_key: jax.random.PRNGKey = None
) -> jnp.ndarray:
    """Basic token sampling implementation"""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
    
    # Apply temperature
    if temperature > 0:
        logits = logits / jnp.maximum(temperature, 1e-7)
    
    # Simple top-k sampling
    if top_k > 0:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        logits_mask = jnp.full_like(logits, True)
        top_k_one_hot = jax.nn.one_hot(top_k_indices, logits.shape[-1], dtype=bool)
        top_k_mask = jnp.logical_or.reduce(top_k_one_hot, axis=-2)
        logits = jnp.where(top_k_mask, logits, jnp.full_like(logits, -float("inf")))
    
    # Sample from the filtered distribution
    next_token = jax.random.categorical(rng_key, logits, axis=-1)
    return next_token

def test_token_generation():
    """Test basic token generation"""
    # Create dummy logits
    batch_size = 2
    vocab_size = 1000
    logits = jnp.random.normal(size=(batch_size, 1, vocab_size))
    
    # Test sampling
    next_token = sample_next_token(logits)
    logger.info(f"Generated token shape: {next_token.shape}")
    return next_token.shape == (batch_size,)

if __name__ == "__main__":
    if test_token_generation():
        logger.info("✅ Token generation test passed!")
    else:
        logger.error("❌ Token generation test failed!") 