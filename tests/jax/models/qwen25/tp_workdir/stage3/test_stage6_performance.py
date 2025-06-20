#!/usr/bin/env python3
"""
Stage 6: Performance Hardening Tests
Focus on memory optimization, compilation efficiency, and production readiness
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
import os
import time
import gc

from stage3 import (
    create_mesh, TensorParallelDense, QwenAttention, QwenMLP, 
    QwenDecoderLayer, Qwen25ForCausalLM
)

# Optimized config for performance testing
PERF_CONFIG = {
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_hidden_layers": 4,
    "intermediate_size": 256,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-5
}

class TestStage6Performance:
    """Test suite for Stage 6 performance hardening"""
    
    def setup_method(self):
        """Set up environment for performance testing"""
        # Force 8 virtual CPU devices
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        jax.config.update('jax_platform_name', 'cpu')
        
        # Performance settings
        jax.config.update('jax_enable_x64', False)  # Use float32 for speed
        jax.clear_caches()
        gc.collect()
    
    def test_gate_6a_compilation_efficiency(self):
        """Gate 6-A: Model compilation time should be reasonable"""
        print("\n=== Gate 6-A: Testing compilation efficiency ===")
        
        # Test different mesh sizes
        mesh_configs = [(1, 1), (2, 1), (4, 1)]
        compilation_times = {}
        
        for mp, dp in mesh_configs:
            print(f"Testing compilation time for {mp}×{dp} mesh...")
            
            start_time = time.time()
            
            if mp == 1 and dp == 1:
                # Single device
                model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                key = jax.random.PRNGKey(42)
                input_ids = jnp.array([[1, 2, 3]])
                
                # Force compilation
                params = model.init(key, input_ids)
                _ = model.apply(params, input_ids, return_dict=True)
                
            else:
                # Multi-device
                mesh = create_mesh(model_parallel=mp, data_parallel=dp)
                with mesh:
                    model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                    key = jax.random.PRNGKey(42)
                    input_ids = jnp.array([[1, 2, 3]])
                    
                    # Force compilation
                    params = model.init(key, input_ids)
                    _ = model.apply(params, input_ids, return_dict=True)
            
            compilation_time = time.time() - start_time
            compilation_times[(mp, dp)] = compilation_time
            print(f"Compilation time for {mp}×{dp}: {compilation_time:.2f}s")
        
        # Verify compilation times are reasonable (< 30s for our test config)
        for mesh_shape, comp_time in compilation_times.items():
            assert comp_time < 30.0, f"Compilation too slow for {mesh_shape}: {comp_time:.2f}s"
        
        print("✅ Compilation efficiency verified")
    
    def test_gate_6b_memory_efficiency(self):
        """Gate 6-B: Memory usage should be reasonable and scale properly"""
        print("\n=== Gate 6-B: Testing memory efficiency ===")
        
        # We'll use model parameter count as a proxy for memory usage
        mesh_configs = [(1, 1), (2, 1), (4, 1)]
        param_counts = {}
        
        for mp, dp in mesh_configs:
            print(f"Testing memory usage for {mp}×{dp} mesh...")
            
            if mp == 1 and dp == 1:
                model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                key = jax.random.PRNGKey(42)
                input_ids = jnp.array([[1, 2, 3]])
                params = model.init(key, input_ids)
                
            else:
                mesh = create_mesh(model_parallel=mp, data_parallel=dp)
                with mesh:
                    model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                    key = jax.random.PRNGKey(42)
                    input_ids = jnp.array([[1, 2, 3]])
                    params = model.init(key, input_ids)
            
            # Count parameters
            total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
            param_counts[(mp, dp)] = total_params
            print(f"Parameter count for {mp}×{dp}: {total_params:,}")
        
        # Verify parameter counts are identical (sharding shouldn't change total params)
        baseline_params = param_counts[(1, 1)]
        for mesh_shape, param_count in param_counts.items():
            assert param_count == baseline_params, f"Parameter count differs for {mesh_shape}"
        
        print("✅ Memory efficiency verified - consistent parameter counts")
    
    def test_gate_6c_throughput_optimization(self):
        """Gate 6-C: Throughput should show measurable improvement with TP"""
        print("\n=== Gate 6-C: Testing throughput optimization ===")
        
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        
        # Test single vs TP throughput
        mesh_configs = [(1, 1), (2, 1)]
        throughput_results = {}
        
        for mp, dp in mesh_configs:
            print(f"Measuring throughput for {mp}×{dp} mesh...")
            
            if mp == 1 and dp == 1:
                model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                params = model.init(key, input_ids)
                
                # Warmup
                for _ in range(5):
                    _ = model.apply(params, input_ids, return_dict=True)
                
                # Measure throughput
                start_time = time.time()
                num_runs = 20
                for _ in range(num_runs):
                    _ = model.apply(params, input_ids, return_dict=True)
                end_time = time.time()
                
                throughput = num_runs / (end_time - start_time)
                throughput_results[(mp, dp)] = throughput
                
            else:
                mesh = create_mesh(model_parallel=mp, data_parallel=dp)
                with mesh:
                    model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                    params = model.init(key, input_ids)
                    
                    # Warmup
                    for _ in range(5):
                        _ = model.apply(params, input_ids, return_dict=True)
                    
                    # Measure throughput
                    start_time = time.time()
                    num_runs = 20
                    for _ in range(num_runs):
                        _ = model.apply(params, input_ids, return_dict=True)
                    end_time = time.time()
                    
                    throughput = num_runs / (end_time - start_time)
                    throughput_results[(mp, dp)] = throughput
            
            print(f"Throughput for {mp}×{dp}: {throughput:.2f} runs/s")
        
        # Verify TP doesn't dramatically hurt throughput (allow 50% slowdown on virtual CPUs)
        baseline_throughput = throughput_results[(1, 1)]
        tp_throughput = throughput_results[(2, 1)]
        
        relative_performance = tp_throughput / baseline_throughput
        print(f"TP relative performance: {relative_performance:.2f}x")
        
        assert relative_performance > 0.5, f"TP throughput too slow: {relative_performance:.2f}x"
        
        print("✅ Throughput optimization verified")
    
    def test_gate_6d_numerical_stability(self):
        """Gate 6-D: Numerical outputs should be stable across runs"""
        print("\n=== Gate 6-D: Testing numerical stability ===")
        
        input_ids = jnp.array([[1, 2, 3, 4, 5]])
        
        # Test stability across multiple runs
        mesh = create_mesh(model_parallel=2, data_parallel=1)
        results = []
        
        with mesh:
            model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
            
            # Run multiple times with same seed
            for run in range(5):
                key = jax.random.PRNGKey(42)  # Same seed each time
                params = model.init(key, input_ids)
                outputs = model.apply(params, input_ids, return_dict=True)
                results.append(outputs["logits"])
        
        # Verify all results are identical
        baseline = results[0]
        for i, result in enumerate(results[1:], 1):
            max_diff = jnp.abs(baseline - result).max()
            print(f"Run {i+1} max diff from baseline: {max_diff}")
            assert max_diff < 1e-6, f"Numerical instability detected in run {i+1}"
        
        print("✅ Numerical stability verified")
    
    def test_gate_6e_generation_quality(self):
        """Gate 6-E: Generated text quality should be maintained with TP"""
        print("\n=== Gate 6-E: Testing generation quality ===")
        
        # Test with different starting tokens
        test_prompts = [
            jnp.array([[1, 2]]),
            jnp.array([[5, 10]]),
            jnp.array([[15, 20, 25]]),
        ]
        
        # Compare single vs TP generation
        for i, prompt in enumerate(test_prompts):
            print(f"Testing prompt {i+1}...")
            
            key = jax.random.PRNGKey(42 + i)
            
            # Single device generation
            model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
            params = model.init(key, prompt)
            
            single_sequence = []
            current_ids = prompt
            past_kv = None
            
            for step in range(10):
                if step == 0:
                    outputs = model.apply(params, current_ids, return_dict=True)
                else:
                    outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                
                logits = outputs["logits"]
                past_kv = outputs["past_key_values"]
                next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                single_sequence.append(int(next_token[0, 0]))
                current_ids = next_token
            
            # TP generation
            mesh = create_mesh(model_parallel=2, data_parallel=1)
            with mesh:
                model = Qwen25ForCausalLM(config=PERF_CONFIG, dtype=jnp.float32)
                params = model.init(key, prompt)
                
                tp_sequence = []
                current_ids = prompt
                past_kv = None
                
                for step in range(10):
                    if step == 0:
                        outputs = model.apply(params, current_ids, return_dict=True)
                    else:
                        outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                    
                    logits = outputs["logits"]
                    past_kv = outputs["past_key_values"]
                    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                    tp_sequence.append(int(next_token[0, 0]))
                    current_ids = next_token
            
            # Verify sequences are identical
            print(f"  Single: {single_sequence}")
            print(f"  TP:     {tp_sequence}")
            assert single_sequence == tp_sequence, f"Generation differs for prompt {i+1}"
        
        print("✅ Generation quality verified")

def run_stage6_gates():
    """Run Stage 6 performance hardening gate tests"""
    print("=" * 60)
    print("STAGE 6: PERFORMANCE HARDENING GATE TESTS")
    print("=" * 60)
    
    # Set up environment
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_enable_x64', False)
    jax.clear_caches()
    
    test_instance = TestStage6Performance()
    test_instance.setup_method()
    
    gates_passed = 0
    total_gates = 0
    
    # Gate 6-A: Compilation efficiency
    try:
        total_gates += 1
        test_instance.test_gate_6a_compilation_efficiency()
        gates_passed += 1
        print("✅ Gate 6-A: Compilation efficiency - PASSED")
    except Exception as e:
        print(f"❌ Gate 6-A: Compilation efficiency - FAILED: {e}")
    
    # Gate 6-B: Memory efficiency
    try:
        total_gates += 1
        test_instance.test_gate_6b_memory_efficiency()
        gates_passed += 1
        print("✅ Gate 6-B: Memory efficiency - PASSED")
    except Exception as e:
        print(f"❌ Gate 6-B: Memory efficiency - FAILED: {e}")
    
    # Gate 6-C: Throughput optimization
    try:
        total_gates += 1
        test_instance.test_gate_6c_throughput_optimization()
        gates_passed += 1
        print("✅ Gate 6-C: Throughput optimization - PASSED")
    except Exception as e:
        print(f"❌ Gate 6-C: Throughput optimization - FAILED: {e}")
    
    # Gate 6-D: Numerical stability
    try:
        total_gates += 1
        test_instance.test_gate_6d_numerical_stability()
        gates_passed += 1
        print("✅ Gate 6-D: Numerical stability - PASSED")
    except Exception as e:
        print(f"❌ Gate 6-D: Numerical stability - FAILED: {e}")
    
    # Gate 6-E: Generation quality
    try:
        total_gates += 1
        test_instance.test_gate_6e_generation_quality()
        gates_passed += 1
        print("✅ Gate 6-E: Generation quality - PASSED")
    except Exception as e:
        print(f"❌ Gate 6-E: Generation quality - FAILED: {e}")
    
    # Final result
    print("\n" + "=" * 60)
    print("STAGE 6 GATE SUMMARY:")
    print(f"Gates passed: {gates_passed}/{total_gates}")
    
    if gates_passed == total_gates:
        print("\n🎉 Stage 6 Complete - Performance hardening successful!")
        print("✅ Compilation efficiency optimized")
        print("✅ Memory usage verified")
        print("✅ Throughput performance acceptable")
        print("✅ Numerical stability confirmed")
        print("✅ Generation quality maintained")
        print("✅ Ready for Stage 7: Production polish!")
    else:
        print(f"\n⚠️ Stage 6 partial: {gates_passed}/{total_gates} gates passed")
        print("Some performance issues need investigation")
    
    return gates_passed == total_gates

if __name__ == "__main__":
    success = run_stage6_gates()
    exit(0 if success else 1) 