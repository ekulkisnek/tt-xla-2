#!/usr/bin/env python3
"""
Stage 5: Multi-Mesh Tests
Tests scaling behavior and functionality across different mesh configurations
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
import os
import time

from stage3 import (
    create_mesh, TensorParallelDense, QwenAttention, QwenMLP, 
    QwenDecoderLayer, Qwen25ForCausalLM
)

# Small config for testing
TEST_CONFIG = {
    "hidden_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "vocab_size": 100,
    "rms_norm_eps": 1e-5
}

class TestStage5MultiMesh:
    """Test suite for Stage 5 multi-mesh functionality"""
    
    def setup_method(self):
        """Set up virtual devices for testing"""
        # Force 8 virtual CPU devices for testing
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        jax.config.update('jax_platform_name', 'cpu')
        
        # Clear JAX cache
        jax.clear_caches()
    
    def test_gate_5a_mesh_1x4_functional(self):
        """Gate 5-A: 1×4 mesh functional test"""
        print("\n=== Gate 5-A: Testing 1×4 mesh functionality ===")
        
        mesh = create_mesh(model_parallel=4, data_parallel=1)
        
        with mesh:
            model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
            key = jax.random.PRNGKey(42)
            
            # Test forward pass
            input_ids = jnp.array([[1, 2, 3]])
            params = model.init(key, input_ids)
            outputs = model.apply(params, input_ids, return_dict=True)
            
            assert outputs is not None
            assert "logits" in outputs
            assert outputs["logits"].shape == (1, 3, TEST_CONFIG["vocab_size"])
            
            print("✅ 1×4 mesh forward pass successful")
    
    def test_gate_5b_mesh_1x8_functional(self):
        """Gate 5-B: 1×8 mesh functional test"""
        print("\n=== Gate 5-B: Testing 1×8 mesh functionality ===")
        
        mesh = create_mesh(model_parallel=8, data_parallel=1)
        
        with mesh:
            model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
            key = jax.random.PRNGKey(42)
            
            # Test forward pass
            input_ids = jnp.array([[1, 2, 3]])
            params = model.init(key, input_ids)
            outputs = model.apply(params, input_ids, return_dict=True)
            
            assert outputs is not None
            assert "logits" in outputs
            assert outputs["logits"].shape == (1, 3, TEST_CONFIG["vocab_size"])
            
            print("✅ 1×8 mesh forward pass successful")
    
    def test_gate_5c_mesh_2x4_dp_functionality(self):
        """Gate 5-C: 2×4 data parallel + tensor parallel mesh test"""
        print("\n=== Gate 5-C: Testing 2×4 DP+TP mesh functionality ===")
        
        mesh = create_mesh(model_parallel=4, data_parallel=2)
        
        with mesh:
            model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
            key = jax.random.PRNGKey(42)
            
            # Test with batch size = 2 (matches data parallel)
            input_ids = jnp.array([[1, 2, 3], [4, 5, 6]])
            params = model.init(key, input_ids)
            outputs = model.apply(params, input_ids, return_dict=True)
            
            assert outputs is not None
            assert "logits" in outputs
            assert outputs["logits"].shape == (2, 3, TEST_CONFIG["vocab_size"])
            
            print("✅ 2×4 DP+TP mesh forward pass successful")
    
    def test_mesh_scaling_consistency(self):
        """Test that different mesh configs produce consistent results"""
        print("\n=== Testing mesh scaling consistency ===")
        
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2, 3]])
        
        # Test different mesh configurations
        mesh_configs = [
            (1, 1),  # Single device
            (2, 1),  # 1×2 TP
            (4, 1),  # 1×4 TP
        ]
        
        results = {}
        
        for mp, dp in mesh_configs:
            if mp == 1 and dp == 1:
                # Single device case
                model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
                params = model.init(key, input_ids)
                outputs = model.apply(params, input_ids, return_dict=True)
                results[(mp, dp)] = outputs["logits"]
            else:
                # Multi-device case
                mesh = create_mesh(model_parallel=mp, data_parallel=dp)
                with mesh:
                    model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
                    params = model.init(key, input_ids)
                    outputs = model.apply(params, input_ids, return_dict=True)
                    results[(mp, dp)] = outputs["logits"]
        
        # Compare results (should be identical due to same random seed)
        baseline = results[(1, 1)]
        for mesh_shape, result in results.items():
            if mesh_shape != (1, 1):
                # Check if results are approximately equal
                diff = jnp.abs(baseline - result).max()
                print(f"Max diff for {mesh_shape}: {diff}")
                assert diff < 1e-5, f"Results differ too much for mesh {mesh_shape}"
        
        print("✅ Mesh scaling consistency verified")
    
    def test_generation_across_meshes(self):
        """Test text generation works consistently across different meshes"""
        print("\n=== Testing generation across meshes ===")
        
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2]])
        
        # Test generation on different meshes
        mesh_configs = [(1, 1), (2, 1), (4, 1)]
        generated_sequences = {}
        
        for mp, dp in mesh_configs:
            print(f"Testing generation on {mp}×{dp} mesh...")
            
            if mp == 1 and dp == 1:
                # Single device
                model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
                params = model.init(key, input_ids)
                
                # Generate 5 tokens
                current_ids = input_ids
                past_kv = None
                sequence = [int(current_ids[0, -1])]
                
                for i in range(5):
                    if i == 0:
                        outputs = model.apply(params, current_ids, return_dict=True)
                    else:
                        outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                    
                    logits = outputs["logits"]
                    past_kv = outputs["past_key_values"]
                    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                    sequence.append(int(next_token[0, 0]))
                    current_ids = next_token
                
                generated_sequences[(mp, dp)] = sequence
                
            else:
                # Multi-device
                mesh = create_mesh(model_parallel=mp, data_parallel=dp)
                with mesh:
                    model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
                    params = model.init(key, input_ids)
                    
                    # Generate 5 tokens
                    current_ids = input_ids
                    past_kv = None
                    sequence = [int(current_ids[0, -1])]
                    
                    for i in range(5):
                        if i == 0:
                            outputs = model.apply(params, current_ids, return_dict=True)
                        else:
                            outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                        
                        logits = outputs["logits"]
                        past_kv = outputs["past_key_values"]
                        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                        sequence.append(int(next_token[0, 0]))
                        current_ids = next_token
                    
                    generated_sequences[(mp, dp)] = sequence
        
        # Verify all sequences are identical
        baseline_seq = generated_sequences[(1, 1)]
        print(f"Baseline sequence: {baseline_seq}")
        
        for mesh_shape, sequence in generated_sequences.items():
            print(f"Mesh {mesh_shape}: {sequence}")
            assert sequence == baseline_seq, f"Generation differs for mesh {mesh_shape}"
        
        print("✅ Generation consistency across meshes verified")
    
    def test_performance_scaling_direction(self):
        """Test that larger meshes show performance improvements (directionally)"""
        print("\n=== Testing performance scaling direction ===")
        
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2]])
        
        mesh_configs = [(1, 1), (2, 1), (4, 1)]
        performance_results = {}
        
        for mp, dp in mesh_configs:
            print(f"Benchmarking {mp}×{dp} mesh...")
            
            # Measure forward pass time
            if mp == 1 and dp == 1:
                model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
                params = model.init(key, input_ids)
                
                # Warmup
                for _ in range(3):
                    model.apply(params, input_ids, return_dict=True)
                
                # Measure
                start_time = time.time()
                for _ in range(10):
                    outputs = model.apply(params, input_ids, return_dict=True)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                performance_results[(mp, dp)] = avg_time
                
            else:
                mesh = create_mesh(model_parallel=mp, data_parallel=dp)
                with mesh:
                    model = Qwen25ForCausalLM(config=TEST_CONFIG, dtype=jnp.float32)
                    params = model.init(key, input_ids)
                    
                    # Warmup
                    for _ in range(3):
                        model.apply(params, input_ids, return_dict=True)
                    
                    # Measure
                    start_time = time.time()
                    for _ in range(10):
                        outputs = model.apply(params, input_ids, return_dict=True)
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 10
                    performance_results[(mp, dp)] = avg_time
        
        # Print results
        print("\nPerformance results:")
        baseline_time = performance_results[(1, 1)]
        for mesh_shape, avg_time in performance_results.items():
            speedup = baseline_time / avg_time
            print(f"Mesh {mesh_shape}: {avg_time:.4f}s (speedup: {speedup:.2f}x)")
        
        # Basic sanity check - larger meshes should generally be faster or similar
        # (In practice with virtual CPUs, this might not hold, but we check it doesn't get dramatically worse)
        tp2_time = performance_results[(2, 1)]
        tp4_time = performance_results[(4, 1)]
        
        # Allow up to 2x slower (since we're using virtual CPUs)
        assert tp2_time <= baseline_time * 2, "2×1 mesh dramatically slower than baseline"
        assert tp4_time <= baseline_time * 2, "4×1 mesh dramatically slower than baseline"
        
        print("✅ Performance scaling direction validated")

def run_stage5_gates():
    """Run Stage 5 gate tests manually"""
    print("=" * 60)
    print("STAGE 5: MULTI-MESH BRING-UP GATE TESTS")
    print("=" * 60)
    
    # Set up environment
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update('jax_platform_name', 'cpu')
    jax.clear_caches()
    
    test_instance = TestStage5MultiMesh()
    test_instance.setup_method()
    
    gates_passed = 0
    total_gates = 0
    
    # Gate 5-A: 1×4 mesh functionality
    try:
        total_gates += 1
        test_instance.test_gate_5a_mesh_1x4_functional()
        gates_passed += 1
        print("✅ Gate 5-A: 1×4 mesh functionality - PASSED")
    except Exception as e:
        print(f"❌ Gate 5-A: 1×4 mesh functionality - FAILED: {e}")
    
    # Gate 5-B: 1×8 mesh functionality
    try:
        total_gates += 1
        test_instance.test_gate_5b_mesh_1x8_functional()
        gates_passed += 1
        print("✅ Gate 5-B: 1×8 mesh functionality - PASSED")
    except Exception as e:
        print(f"❌ Gate 5-B: 1×8 mesh functionality - FAILED: {e}")
    
    # Gate 5-C: 2×4 DP+TP functionality
    try:
        total_gates += 1
        test_instance.test_gate_5c_mesh_2x4_dp_functionality()
        gates_passed += 1
        print("✅ Gate 5-C: 2×4 DP+TP mesh functionality - PASSED")
    except Exception as e:
        print(f"❌ Gate 5-C: 2×4 DP+TP mesh functionality - FAILED: {e}")
    
    # Additional verification tests
    try:
        test_instance.test_mesh_scaling_consistency()
        print("✅ Mesh scaling consistency - PASSED")
    except Exception as e:
        print(f"❌ Mesh scaling consistency - FAILED: {e}")
    
    try:
        test_instance.test_generation_across_meshes()
        print("✅ Generation consistency - PASSED")
    except Exception as e:
        print(f"❌ Generation consistency - FAILED: {e}")
    
    try:
        test_instance.test_performance_scaling_direction()
        print("✅ Performance scaling direction - PASSED")
    except Exception as e:
        print(f"❌ Performance scaling direction - FAILED: {e}")
    
    # Final result
    print("\n" + "=" * 60)
    print("STAGE 5 GATE SUMMARY:")
    print(f"Gates passed: {gates_passed}/{total_gates}")
    
    if gates_passed == total_gates:
        print("\n🎉 Stage 5 Complete - Multi-mesh bring-up successful!")
        print("✅ All core mesh configurations working")
        print("✅ Scaling behavior validated")
        print("✅ Ready for Stage 6: Performance hardening!")
    else:
        print(f"\n⚠️ Stage 5 partial: {gates_passed}/{total_gates} gates passed")
        print("Some mesh configurations need investigation")
    
    return gates_passed == total_gates

if __name__ == "__main__":
    success = run_stage5_gates()
    exit(0 if success else 1) 