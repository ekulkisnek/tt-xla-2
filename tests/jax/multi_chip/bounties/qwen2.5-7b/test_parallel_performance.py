# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test parallel performance vs single device performance.

Tests the JAX tensor parallel Qwen2.5-7B implementation with simulated devices
to verify that parallel execution works as well as single device execution.
"""

import os
import time
from contextlib import contextmanager
from typing import Dict, Any, List

import jax
import jax.numpy as jnp
import transformers
import qwen_nnx
from qwen_nnx.util import timer


MODEL = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"

SHARDING_RULES_SINGLE = list({
    qwen_nnx.Axis.EMBED: None,
    qwen_nnx.Axis.MLP: None,      # No sharding for single device
    qwen_nnx.Axis.HEAD: None,     # No sharding for single device
    qwen_nnx.Axis.QHEAD: None,
    qwen_nnx.Axis.KVHEAD: None,
    qwen_nnx.Axis.VOCAB: None,
}.items())

SHARDING_RULES_PARALLEL = list({
    qwen_nnx.Axis.EMBED: None,
    qwen_nnx.Axis.MLP: "x",       # Shard across devices
    qwen_nnx.Axis.HEAD: "x",      # Shard across devices
    qwen_nnx.Axis.QHEAD: None,
    qwen_nnx.Axis.KVHEAD: None,
    qwen_nnx.Axis.VOCAB: None,
}.items())


@contextmanager
def force_device_count(n_devices: int):
    """Force JAX to use a specific number of simulated CPU devices."""
    original_flag = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_devices}'
    
    # Clear JAX caches to pick up new device count
    jax.config.update('jax_platform_name', 'cpu')
    jax.clear_caches()
    
    try:
        yield
    finally:
        os.environ['XLA_FLAGS'] = original_flag
        jax.clear_caches()


def load_model_and_tokenizer(mesh: jax.sharding.Mesh, sharding_rules: List):
    """Load model and tokenizer with specified mesh and sharding rules."""
    print(f"Loading model with {len(mesh.devices)} devices...")
    
    with timer("Model loading"):
        model = qwen_nnx.QwenModel.load_from_hf_pt_model(
            MODEL,
            dtype=jnp.float32,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )
    
    with timer("Tokenizer loading"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    
    return model, tokenizer


def run_inference_benchmark(model, tokenizer, mesh: jax.sharding.Mesh, test_name: str) -> Dict[str, Any]:
    """Run inference benchmark with various input sizes."""
    print(f"\n=== {test_name} ===")
    
    test_inputs = [
        "Hello, how are you?",
        "The capital of France is Paris, which is known for",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that computer systems use to perform tasks without explicit instructions, relying instead on",
        "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together."
    ]
    
    results = {
        'test_name': test_name,
        'device_count': len(mesh.devices),
        'timings': [],
        'outputs': [],
        'shapes': []
    }
    
    # Compile the model first
    print("Compiling model...")
    sample_tokens = tokenizer("Hello world", return_tensors="jax")["input_ids"]
    
    @jax.jit
    def jit_model(input_tokens):
        with mesh:
            return model(input_tokens)
    
    with timer("JIT compilation"):
        # Trigger compilation
        _ = jit_model(sample_tokens).block_until_ready()
    
    print("Running benchmarks...")
    for i, text in enumerate(test_inputs):
        tokens = tokenizer(text, return_tensors="jax")["input_ids"]
        seq_len = tokens.shape[1]
        
        # Warmup runs
        for _ in range(2):
            _ = jit_model(tokens).block_until_ready()
        
        # Timed runs
        times = []
        for run in range(5):
            start_time = time.perf_counter()
            with mesh:
                output = jit_model(tokens)
                output.block_until_ready()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        # Get predicted next token
        next_token = jnp.argmax(output[0, -1])
        next_text = tokenizer.decode([next_token])
        
        results['timings'].append({
            'input_length': seq_len,
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'throughput_tokens_per_sec': seq_len / avg_time
        })
        results['outputs'].append({
            'input': text[:50] + "..." if len(text) > 50 else text,
            'predicted_next': next_text,
            'next_token_id': int(next_token)
        })
        results['shapes'].append(output.shape)
        
        print(f"  Input {i+1} (len={seq_len}): {avg_time*1000:.2f}ms avg, {min_time*1000:.2f}ms min, {seq_len/avg_time:.1f} tok/s")
        print(f"    -> '{next_text}'")
    
    return results


def test_correctness_across_devices(single_result: Dict, parallel_result: Dict) -> bool:
    """Test that single and parallel executions produce the same outputs."""
    print("\n=== Correctness Test ===")
    
    all_correct = True
    
    for i, (single_out, parallel_out) in enumerate(zip(single_result['outputs'], parallel_result['outputs'])):
        single_token = single_out['next_token_id']
        parallel_token = parallel_out['next_token_id']
        
        if single_token == parallel_token:
            print(f"  ✅ Input {i+1}: Both predict token {single_token} ('{single_out['predicted_next']}')")
        else:
            print(f"  ❌ Input {i+1}: Single={single_token} ('{single_out['predicted_next']}'), Parallel={parallel_token} ('{parallel_out['predicted_next']}')")
            all_correct = False
    
    if all_correct:
        print("  🎉 All outputs match! Parallel execution is correct.")
    else:
        print("  ⚠️  Some outputs differ between single and parallel execution.")
    
    return all_correct


def analyze_performance(single_result: Dict, parallel_result: Dict):
    """Analyze performance differences between single and parallel execution."""
    print("\n=== Performance Analysis ===")
    
    print(f"Single Device ({single_result['device_count']} device):")
    for i, timing in enumerate(single_result['timings']):
        print(f"  Input {i+1}: {timing['avg_time_ms']:.2f}ms, {timing['throughput_tokens_per_sec']:.1f} tok/s")
    
    print(f"\nParallel ({parallel_result['device_count']} devices):")
    for i, timing in enumerate(parallel_result['timings']):
        print(f"  Input {i+1}: {timing['avg_time_ms']:.2f}ms, {timing['throughput_tokens_per_sec']:.1f} tok/s")
    
    print("\nSpeedup Analysis:")
    total_single_time = sum(t['avg_time_ms'] for t in single_result['timings'])
    total_parallel_time = sum(t['avg_time_ms'] for t in parallel_result['timings'])
    overall_speedup = total_single_time / total_parallel_time
    
    print(f"  Overall speedup: {overall_speedup:.2f}x")
    
    for i, (single_timing, parallel_timing) in enumerate(zip(single_result['timings'], parallel_result['timings'])):
        speedup = single_timing['avg_time_ms'] / parallel_timing['avg_time_ms']
        print(f"  Input {i+1} speedup: {speedup:.2f}x")
    
    if overall_speedup > 1.1:
        print("  🚀 Parallel execution is faster!")
    elif overall_speedup < 0.9:
        print("  🐌 Parallel execution is slower (may be due to overhead)")
    else:
        print("  ⚖️  Similar performance (overhead vs parallelism)")


def main():
    """Main test function."""
    print("🧪 Testing JAX Tensor Parallel Qwen2.5-7B Performance")
    print("=" * 60)
    
    # Test with 1 device (single)
    print("\n🔧 Setting up single device test...")
    with force_device_count(1):
        devices = jax.devices("cpu")
        mesh_single = jax.make_mesh((len(devices),), axis_names=("x",), devices=devices)
        model_single, tokenizer = load_model_and_tokenizer(mesh_single, SHARDING_RULES_SINGLE)
        single_result = run_inference_benchmark(model_single, tokenizer, mesh_single, "Single Device")
    
    # Test with 4 devices (parallel)
    print("\n🔧 Setting up parallel device test...")
    with force_device_count(4):
        devices = jax.devices("cpu")
        mesh_parallel = jax.make_mesh((len(devices),), axis_names=("x",), devices=devices)
        model_parallel, tokenizer = load_model_and_tokenizer(mesh_parallel, SHARDING_RULES_PARALLEL)
        parallel_result = run_inference_benchmark(model_parallel, tokenizer, mesh_parallel, "Parallel (4 devices)")
    
    # Compare results
    correctness_ok = test_correctness_across_devices(single_result, parallel_result)
    analyze_performance(single_result, parallel_result)
    
    print("\n" + "=" * 60)
    if correctness_ok:
        print("✅ SUCCESS: Tensor parallel implementation works correctly!")
        print("   - Outputs match between single and parallel execution")
        print("   - Performance characteristics analyzed")
    else:
        print("❌ ISSUE: Output differences detected between single and parallel")
    print("=" * 60)


if __name__ == "__main__":
    main() 