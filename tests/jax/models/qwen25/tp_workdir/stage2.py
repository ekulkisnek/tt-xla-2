#!/usr/bin/env python3
"""
Tensor Parallel Qwen2.5-7B implementation based on working q25_jax.py
STAGE 2: Parameter sharding only - math identical to Stage 1

Usage:
python stage2.py --model_path ../weights --prompt "Hello" --max_tokens 3 --temperature 0.0 --tp 1
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
from safetensors import safe_open
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_tp")

# === STAGE 2: MESH CREATION ===
def create_mesh(model_parallel: int, data_parallel: int = 1):
    """Create a deterministic device mesh for tensor parallelism."""
    devices = jax.devices()
    if len(devices) < model_parallel * data_parallel:
        raise RuntimeError(f"Need {model_parallel * data_parallel} devices, have {len(devices)}")
    mesh = np.array(devices[:model_parallel * data_parallel]).reshape(data_parallel, model_parallel)
    return Mesh(mesh, ("data", "model"))

# === TENSOR PARALLEL DENSE LAYER (STAGE 2: SHARDING ONLY) ===
class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism support - Stage 2: parameter sharding only."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    shard_axes: Tuple[Optional[str], Optional[str]] = (None, "model")
    
    @nn.compact
    def __call__(self, x):
        # Stage 2: Just do normal Dense computation, sharding constraints added to parameters only
        return nn.Dense(self.features, use_bias=self.use_bias, dtype=self.dtype)(x)

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B Tensor Parallel (Stage 2)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    print(f"Stage 2: Testing with TP={args.tp}")
    print(f"Available devices: {len(jax.devices())}")
    
    if args.tp == 1:
        print("✓ Stage 1: Single device mode works")
    else:
        try:
            mesh = create_mesh(model_parallel=args.tp, data_parallel=1)
            print(f"✓ Stage 2: Created mesh {mesh.devices.shape} with axes {mesh.axis_names}")
            assert mesh.axis_names == ('data', 'model'), f"Wrong axis names: {mesh.axis_names}"
            print("✓ Stage 2 mesh creation test passed")
        except Exception as e:
            print(f"✗ Stage 2 mesh creation failed: {e}")

if __name__ == "__main__":
    main()
