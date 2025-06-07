#!/usr/bin/env python3
"""
Step 1: Basic model structure and initialization
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_step1")

class QwenAttention(nn.Module):
    """Basic attention module - start with minimal implementation"""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config["hidden_size"]
        self.num_heads = self.config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        
        # Basic projections
        self.q_proj = nn.Dense(self.hidden_size, dtype=self.dtype)
        self.k_proj = nn.Dense(self.hidden_size, dtype=self.dtype)
        self.v_proj = nn.Dense(self.hidden_size, dtype=self.dtype)
        self.o_proj = nn.Dense(self.hidden_size, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None):
        # Basic attention implementation
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Compute attention
        attention_scores = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / jnp.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        attention_output = jnp.matmul(attention_probs, value_states)
        
        # Reshape and project output
        attention_output = attention_output.reshape(batch_size, seq_length, self.hidden_size)
        attention_output = self.o_proj(attention_output)
        
        return attention_output

def test_model_structure():
    """Test the basic model structure"""
    # Create a minimal config
    config = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "vocab_size": 1000,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-6
    }
    
    # Create a small test input
    batch_size = 2
    seq_length = 10
    hidden_size = config["hidden_size"]
    test_input = jnp.ones((batch_size, seq_length, hidden_size))
    
    # Initialize and test attention module
    attention = QwenAttention(config=config)
    params = attention.init(jax.random.PRNGKey(0), test_input)
    output = attention.apply(params, test_input)
    
    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    return output.shape == test_input.shape

if __name__ == "__main__":
    logger.info("Testing basic model structure...")
    if test_model_structure():
        logger.info("✅ Basic model structure test passed!")
    else:
        logger.error("❌ Basic model structure test failed!") 