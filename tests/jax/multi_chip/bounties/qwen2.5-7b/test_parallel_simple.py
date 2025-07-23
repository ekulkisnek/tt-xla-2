# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Simple test to compare single vs parallel device performance."""

import time
import sys
import os

import jax
import jax.numpy as jnp
import transformers
import qwen_nnx
from qwen_nnx.util import timer


MODEL = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"


def run_test(use_parallel: bool = False):
    """Run inference test with specified configuration."""
    n_devices = len(jax.devices())
    print(f"Running with {n_devices} devices (parallel={use_parallel})")
    
    if use_parallel:
        sharding_rules = list({
            qwen_nnx.Axis.EMBED: None,
            qwen_nnx.Axis.MLP: "x",       # Shard across devices
            qwen_nnx.Axis.HEAD: "x",      # Shard across devices
            qwen_nnx.Axis.QHEAD: None,
            qwen_nnx.Axis.KVHEAD: None,
            qwen_nnx.Axis.VOCAB: None,
        }.items())
    else:
        sharding_rules = list({
            qwen_nnx.Axis.EMBED: None,
            qwen_nnx.Axis.MLP: None,      # No sharding
            qwen_nnx.Axis.HEAD: None,     # No sharding
            qwen_nnx.Axis.QHEAD: None,
            qwen_nnx.Axis.KVHEAD: None,
            qwen_nnx.Axis.VOCAB: None,
        }.items())
    
    # Create mesh
    devices = jax.devices("cpu")
    mesh = jax.make_mesh((len(devices),), axis_names=("x",), devices=devices)
    
    print(f"Device configuration: {[str(d) for d in devices]}")
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model = qwen_nnx.QwenModel.load_from_hf_pt_model(
        MODEL,
        dtype=jnp.float32,
        mesh=mesh,
        sharding_rules=sharding_rules,
    )
    print(f"Model loading: {(time.time() - start_time) * 1000:.1f}ms")
    
    # Load tokenizer  
    start_time = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    print(f"Tokenizer loading: {(time.time() - start_time) * 1000:.1f}ms")
    
    # Test inputs of varying lengths
    test_inputs = [
        "Hello world",
        "The capital of France is",
        "Machine learning is a field of artificial intelligence that focuses on algorithms",
    ]
    
    results = []
    
    # Compile first
    print("Compiling model...")
    sample_tokens = tokenizer("Test", return_tensors="jax")["input_ids"]
    
    @jax.jit
    def jit_model(tokens):
        with mesh:
            return model(tokens)
    
    start_time = time.time()
    _ = jit_model(sample_tokens).block_until_ready()
    print(f"JIT compilation: {(time.time() - start_time) * 1000:.1f}ms")
    
    print("Running inference tests...")
    for i, text in enumerate(test_inputs):
        tokens = tokenizer(text, return_tensors="jax")["input_ids"]
        seq_len = tokens.shape[1]
        
        # Warmup
        for _ in range(2):
            _ = jit_model(tokens).block_until_ready()
        
        # Timed runs
        times = []
        for _ in range(3):
            start = time.perf_counter()
            with mesh:
                output = jit_model(tokens)
                output.block_until_ready()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        
        # Get prediction
        next_token = jnp.argmax(output[0, -1])
        next_text = tokenizer.decode([next_token])
        
        result = {
            'input': text,
            'seq_len': seq_len,
            'avg_time_ms': avg_time * 1000,
            'throughput': seq_len / avg_time,
            'next_token': int(next_token),
            'next_text': next_text
        }
        results.append(result)
        
        print(f"  Test {i+1} (len={seq_len}): {avg_time*1000:.1f}ms, {seq_len/avg_time:.1f} tok/s -> '{next_text}'")
    
    # Summary
    total_time = sum(r['avg_time_ms'] for r in results)
    avg_throughput = sum(r['throughput'] for r in results) / len(results)
    
    print(f"Summary: {total_time:.1f}ms total, {avg_throughput:.1f} avg tok/s")
    
    return {
        'n_devices': n_devices,
        'use_parallel': use_parallel,
        'results': results,
        'total_time_ms': total_time,
        'avg_throughput': avg_throughput
    }


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "parallel":
        result = run_test(use_parallel=True)
        print(f"RESULT: parallel,{result['n_devices']},{result['total_time_ms']:.1f},{result['avg_throughput']:.1f}")
        
        # Print outputs for correctness check
        for i, r in enumerate(result['results']):
            print(f"OUTPUT_{i}: {r['next_token']}")
    else:
        result = run_test(use_parallel=False)
        print(f"RESULT: single,{result['n_devices']},{result['total_time_ms']:.1f},{result['avg_throughput']:.1f}")
        
        # Print outputs for correctness check
        for i, r in enumerate(result['results']):
            print(f"OUTPUT_{i}: {r['next_token']}")


if __name__ == "__main__":
    main() 