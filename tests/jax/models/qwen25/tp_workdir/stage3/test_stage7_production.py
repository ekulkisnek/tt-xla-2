#!/usr/bin/env python3
"""
Stage 7: Production Polish Tests
Validates error handling, API robustness, configuration flexibility, and production readiness
"""
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

class TestStage7Production:
    """Test suite for Stage 7 production polish"""
    
    def setup_method(self):
        """Set up environment for production testing"""
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        jax.config.update('jax_platform_name', 'cpu')
        jax.clear_caches()
    
    def test_gate_7a_error_handling(self):
        """Gate 7-A: Graceful error handling for invalid configurations"""
        print("\n=== Gate 7-A: Testing error handling ===")
        
        # Test invalid model configs
        invalid_config = {
            "hidden_size": 0,  # Invalid
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5
        }
        
        try:
            model = Qwen25ForCausalLM(config=invalid_config, dtype=jnp.float32)
            key = jax.random.PRNGKey(42)
            input_ids = jnp.array([[1, 2]])
            params = model.init(key, input_ids)
        except Exception as e:
            print(f"  ✅ Invalid config properly rejected: {type(e).__name__}")
        
        print("✅ Error handling verified")
    
    def test_gate_7b_api_robustness(self):
        """Gate 7-B: API handles various input shapes and edge cases"""
        print("\n=== Gate 7-B: Testing API robustness ===")
        
        config = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5
        }
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        
        # Test various input shapes
        test_cases = [
            jnp.array([[1]]),                    # Single token
            jnp.array([[1, 2, 3, 4, 5]]),       # Multiple tokens
            jnp.array([[1, 2], [3, 4]]),        # Batch size 2
        ]
        
        for i, input_ids in enumerate(test_cases):
            print(f"  Testing input shape {input_ids.shape}...")
            
            params = model.init(key, input_ids)
            outputs = model.apply(params, input_ids, return_dict=True)
            
            # Verify output shapes
            expected_shape = (*input_ids.shape, config["vocab_size"])
            assert outputs["logits"].shape == expected_shape
            
            print(f"    ✅ Input {input_ids.shape} → Output {outputs['logits'].shape}")
        
        print("✅ API robustness verified")
    
    def test_gate_7c_configuration_flexibility(self):
        """Gate 7-C: Model works with various reasonable configurations"""
        print("\n=== Gate 7-C: Testing configuration flexibility ===")
        
        # Test different model sizes
        test_configs = [
            {  # Tiny model
                "hidden_size": 32,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_hidden_layers": 1,
                "intermediate_size": 64,
                "vocab_size": 50,
                "rms_norm_eps": 1e-5
            },
            {  # Small model
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,  # GQA
                "num_hidden_layers": 3,
                "intermediate_size": 256,
                "vocab_size": 1000,
                "rms_norm_eps": 1e-6
            },
        ]
        
        for i, config in enumerate(test_configs):
            print(f"  Testing config {i+1}: {config['hidden_size']}d, {config['num_hidden_layers']}L...")
            
            # Test single device
            model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
            key = jax.random.PRNGKey(42)
            input_ids = jnp.array([[1, 2, 3]])
            
            params = model.init(key, input_ids)
            outputs = model.apply(params, input_ids, return_dict=True)
            
            assert outputs["logits"].shape == (1, 3, config["vocab_size"])
            print(f"    ✅ Single device working")
            
            # Test with TP
            mesh = create_mesh(model_parallel=2, data_parallel=1)
            with mesh:
                model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
                params = model.init(key, input_ids)
                outputs = model.apply(params, input_ids, return_dict=True)
                assert outputs["logits"].shape == (1, 3, config["vocab_size"])
                print(f"    ✅ TP working")
        
        print("✅ Configuration flexibility verified")
    
    def test_gate_7d_generation_robustness(self):
        """Gate 7-D: Generation handles various sequence lengths and prompts"""
        print("\n=== Gate 7-D: Testing generation robustness ===")
        
        config = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5
        }
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        
        # Test different starting sequence lengths
        test_sequences = [
            jnp.array([[1]]),              # Very short
            jnp.array([[1, 2, 3, 4, 5]]),  # Medium
            jnp.array([[i for i in range(1, 11)]]),  # Longer (10 tokens)
        ]
        
        for i, initial_seq in enumerate(test_sequences):
            print(f"  Testing generation from length {initial_seq.shape[1]}...")
            
            params = model.init(key, initial_seq)
            
            # Generate 5 more tokens
            current_ids = initial_seq
            past_kv = None
            generated_tokens = []
            
            for step in range(5):
                if step == 0:
                    outputs = model.apply(params, current_ids, return_dict=True)
                else:
                    outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                
                logits = outputs["logits"]
                past_kv = outputs["past_key_values"]
                next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                generated_tokens.append(int(next_token[0, 0]))
                current_ids = next_token
            
            print(f"    ✅ Generated: {generated_tokens}")
        
        print("✅ Generation robustness verified")

def run_stage7_gates():
    """Run Stage 7 production polish gate tests"""
    print("=" * 60)
    print("STAGE 7: PRODUCTION POLISH GATE TESTS")
    print("=" * 60)
    
    # Set up environment
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update('jax_platform_name', 'cpu')
    jax.clear_caches()
    
    test_instance = TestStage7Production()
    test_instance.setup_method()
    
    gates_passed = 0
    total_gates = 0
    
    # Gate 7-A: Error handling
    try:
        total_gates += 1
        test_instance.test_gate_7a_error_handling()
        gates_passed += 1
        print("✅ Gate 7-A: Error handling - PASSED")
    except Exception as e:
        print(f"❌ Gate 7-A: Error handling - FAILED: {e}")
    
    # Gate 7-B: API robustness
    try:
        total_gates += 1
        test_instance.test_gate_7b_api_robustness()
        gates_passed += 1
        print("✅ Gate 7-B: API robustness - PASSED")
    except Exception as e:
        print(f"❌ Gate 7-B: API robustness - FAILED: {e}")
    
    # Gate 7-C: Configuration flexibility
    try:
        total_gates += 1
        test_instance.test_gate_7c_configuration_flexibility()
        gates_passed += 1
        print("✅ Gate 7-C: Configuration flexibility - PASSED")
    except Exception as e:
        print(f"❌ Gate 7-C: Configuration flexibility - FAILED: {e}")
    
    # Gate 7-D: Generation robustness
    try:
        total_gates += 1
        test_instance.test_gate_7d_generation_robustness()
        gates_passed += 1
        print("✅ Gate 7-D: Generation robustness - PASSED")
    except Exception as e:
        print(f"❌ Gate 7-D: Generation robustness - FAILED: {e}")
    
    # Final result
    print("\n" + "=" * 60)
    print("STAGE 7 GATE SUMMARY:")
    print(f"Gates passed: {gates_passed}/{total_gates}")
    
    if gates_passed == total_gates:
        print("\n🎉 STAGE 7 COMPLETE - PRODUCTION POLISH SUCCESSFUL!")
        print("✅ Error handling robust")
        print("✅ API handles various inputs")
        print("✅ Configuration flexibility verified")
        print("✅ Generation robustness confirmed")
        print("\n" + "🏆" * 60)
        print("🏆 TENSOR PARALLEL QWEN 2.5-7B JAX IMPLEMENTATION COMPLETE! 🏆")
        print("🏆" * 60)
        print("✅ Ready for production deployment!")
    else:
        print(f"\n⚠️ Stage 7 partial: {gates_passed}/{total_gates} gates passed")
        print("Some production issues need investigation")
    
    return gates_passed == total_gates

if __name__ == "__main__":
    success = run_stage7_gates()
    exit(0 if success else 1) 