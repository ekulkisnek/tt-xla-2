#!/usr/bin/env python3
"""
Quick test to debug RoPE implementation.
"""

import numpy as np
import jax.numpy as jnp

def our_current_rope_cache(position_ids, head_dim, rope_theta=10000.0):
    """Our current (broken) implementation."""
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    
    # position_ids: [batch, seq] -> [batch, seq, 1]
    seq_freqs = position_ids[..., None] * freqs[None, None, :]
    
    cos = jnp.cos(seq_freqs)
    sin = jnp.sin(seq_freqs)
    
    # WRONG: This repeats instead of interleaving
    cos = jnp.repeat(cos, 2, axis=-1)
    sin = jnp.repeat(sin, 2, axis=-1)
    
    return cos, sin

def correct_rope_cache(position_ids, head_dim, rope_theta=10000.0):
    """Correct PyTorch-style implementation."""
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    
    # position_ids: [batch, seq] -> [batch, seq, 1]
    seq_freqs = position_ids[..., None] * freqs[None, None, :]
    
    cos_half = jnp.cos(seq_freqs)
    sin_half = jnp.sin(seq_freqs)
    
    # CORRECT: Interleave properly to match PyTorch
    cos = jnp.stack([cos_half, cos_half], axis=-1).reshape(*cos_half.shape[:-1], -1)
    sin = jnp.stack([sin_half, sin_half], axis=-1).reshape(*sin_half.shape[:-1], -1)
    
    return cos, sin

def test_rope_difference():
    """Test the difference between our broken and correct RoPE."""
    # Simple test case
    position_ids = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)  # [1, 4]
    head_dim = 128
    
    print("Testing RoPE implementations...")
    print(f"Position IDs: {position_ids.tolist()}")
    print(f"Head dim: {head_dim}")
    
    # Test our current implementation
    cos_broken, sin_broken = our_current_rope_cache(position_ids, head_dim)
    print(f"\nOur (broken) implementation:")
    print(f"  cos shape: {cos_broken.shape}")
    print(f"  sin shape: {sin_broken.shape}")
    print(f"  cos[0,1,:8]: {cos_broken[0,1,:8]}")  # Position 1, not 0
    print(f"  sin[0,1,:8]: {sin_broken[0,1,:8]}")  # Position 1, not 0
    
    # Test correct implementation
    cos_correct, sin_correct = correct_rope_cache(position_ids, head_dim)
    print(f"\nCorrect implementation:")
    print(f"  cos shape: {cos_correct.shape}")
    print(f"  sin shape: {sin_correct.shape}")
    print(f"  cos[0,1,:8]: {cos_correct[0,1,:8]}")  # Position 1, not 0
    print(f"  sin[0,1,:8]: {sin_correct[0,1,:8]}")  # Position 1, not 0
    
    # Compare differences
    cos_diff = jnp.max(jnp.abs(cos_broken - cos_correct))
    sin_diff = jnp.max(jnp.abs(sin_broken - sin_correct))
    
    print(f"\nDifferences:")
    print(f"  Max cos difference: {float(cos_diff):.6f}")
    print(f"  Max sin difference: {float(sin_diff):.6f}")
    
    # Show detailed comparison for non-zero positions
    print(f"\nDetailed comparison at position 1:")
    print(f"  cos_broken[0,1,0:4]: {cos_broken[0,1,0:4]}")
    print(f"  cos_correct[0,1,0:4]: {cos_correct[0,1,0:4]}")
    print(f"  sin_broken[0,1,0:4]: {sin_broken[0,1,0:4]}")
    print(f"  sin_correct[0,1,0:4]: {sin_correct[0,1,0:4]}")
    
    if cos_diff > 1e-6 or sin_diff > 1e-6:
        print("❌ SIGNIFICANT DIFFERENCES: RoPE implementations don't match!")
        print("   This explains the logits differences!")
    else:
        print("✅ Implementations match")

if __name__ == "__main__":
    test_rope_difference() 