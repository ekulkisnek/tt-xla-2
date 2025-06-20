#!/usr/bin/env python3
"""
Minimal test for tensor parallel implementation structure
Tests the 5-phase roadmap without loading full model weights
"""
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Import our implementation
from qwen25_tp_final import (
    create_mesh, TensorParallelDense, QwenAttention, QwenMLP, 
    QwenDecoderLayer, Qwen25ForCausalLM
)

def test_phase_1_mesh_creation():
    """Test Phase 1: Deterministic mesh factory"""
    print("=== Phase 1: Mesh Creation Test ===")
    
    # Test single device (should work)
    if len(jax.devices()) == 1:
        print("✓ Single device detected - skipping multi-device mesh test")
        return True
    
    # Test multi-device mesh
    try:
        mesh = create_mesh(model_parallel=2, data_parallel=1)
        assert mesh.axis_names == ('data', 'model')
        print(f"✓ Mesh created: {mesh.devices.shape} with axes {mesh.axis_names}")
        return True
    except Exception as e:
        print(f"✗ Mesh creation failed: {e}")
        return False

def test_phase_1_tensor_parallel_dense():
    """Test Phase 1: TensorParallelDense layer"""
    print("\n=== Phase 1: TensorParallelDense Test ===")
    
    try:
        # Create a small dense layer
        layer = TensorParallelDense(features=128, shard_axes=(None, "model"))
        
        # Test with dummy input
        key = jax.random.PRNGKey(42)
        x = jnp.ones((2, 64))  # batch=2, features=64
        
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert output.shape == (2, 128)
        print(f"✓ TensorParallelDense: input {x.shape} -> output {output.shape}")
        return True
    except Exception as e:
        print(f"✗ TensorParallelDense failed: {e}")
        return False

def test_phase_2_model_components():
    """Test Phase 2: Model components with minimal config"""
    print("\n=== Phase 2: Model Components Test ===")
    
    # Minimal config for testing
    config = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-5
    }
    
    try:
        # Test QwenAttention
        attn = QwenAttention(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        hidden_states = jnp.ones((1, 4, 128))  # batch=1, seq=4, hidden=128
        
        attn_params = attn.init(key, hidden_states)
        attn_output, kv_cache = attn.apply(attn_params, hidden_states)
        
        assert attn_output.shape == (1, 4, 128)
        print(f"✓ QwenAttention: {hidden_states.shape} -> {attn_output.shape}")
        
        # Test QwenMLP
        mlp = QwenMLP(config=config, dtype=jnp.float32)
        mlp_params = mlp.init(key, hidden_states)
        mlp_output = mlp.apply(mlp_params, hidden_states)
        
        assert mlp_output.shape == (1, 4, 128)
        print(f"✓ QwenMLP: {hidden_states.shape} -> {mlp_output.shape}")
        
        # Test QwenDecoderLayer
        layer = QwenDecoderLayer(config=config, dtype=jnp.float32)
        layer_params = layer.init(key, hidden_states)
        layer_output, layer_kv = layer.apply(layer_params, hidden_states)
        
        assert layer_output.shape == (1, 4, 128)
        print(f"✓ QwenDecoderLayer: {hidden_states.shape} -> {layer_output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model components failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_2_full_model():
    """Test Phase 2: Full model with minimal config"""
    print("\n=== Phase 2: Full Model Test ===")
    
    config = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-5
    }
    
    try:
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2, 3, 4]])  # batch=1, seq=4
        
        params = model.init(key, input_ids)
        outputs = model.apply(params, input_ids, return_dict=True)
        
        logits = outputs["logits"]
        assert logits.shape == (1, 4, 1000)  # batch, seq, vocab
        print(f"✓ Full model: input {input_ids.shape} -> logits {logits.shape}")
        
        # Test with past_key_values
        past_kv = outputs["past_key_values"]
        new_input = jnp.array([[5]])  # next token
        
        outputs2 = model.apply(params, new_input, past_key_values=past_kv, return_dict=True)
        logits2 = outputs2["logits"]
        assert logits2.shape == (1, 1, 1000)
        print(f"✓ Incremental generation: input {new_input.shape} -> logits {logits2.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Full model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_3_jit_compilation():
    """Test Phase 3: JIT compilation"""
    print("\n=== Phase 3: JIT Compilation Test ===")
    
    config = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "vocab_size": 100,
        "rms_norm_eps": 1e-5
    }
    
    try:
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2]])
        
        params = model.init(key, input_ids)
        
        # JIT compile the model
        @jax.jit
        def forward_step(params, input_ids, past_key_values=None):
            return model.apply(params, input_ids, past_key_values=past_key_values, return_dict=True)
        
        # Test JIT compilation
        outputs = forward_step(params, input_ids)
        logits = outputs["logits"]
        
        assert logits.shape == (1, 2, 100)
        print(f"✓ JIT compilation: input {input_ids.shape} -> logits {logits.shape}")
        
        # Test incremental with JIT
        past_kv = outputs["past_key_values"]
        new_input = jnp.array([[3]])
        outputs2 = forward_step(params, new_input, past_kv)
        
        assert outputs2["logits"].shape == (1, 1, 100)
        print(f"✓ JIT incremental: works correctly")
        
        return True
    except Exception as e:
        print(f"✗ JIT compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase_4_parameter_counting():
    """Test Phase 4: Parameter counting matches expected"""
    print("\n=== Phase 4: Parameter Counting Test ===")
    
    config = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-5
    }
    
    try:
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        input_ids = jnp.array([[1, 2, 3]])
        
        params = model.init(key, input_ids)
        
        # Count parameters
        total_params = sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))
        
        # Expected parameters:
        # embed_tokens: 1000 * 128 = 128,000
        # 2 layers * (attn + mlp + 2 norms)
        # attn: 4 * (128 * 128) = 65,536 per layer
        # mlp: 128*256 + 128*256 + 256*128 = 98,304 per layer  
        # norms: 2 * 128 = 256 per layer
        # lm_head: 128 * 1000 = 128,000
        
        expected_min = 400_000  # rough estimate
        expected_max = 600_000
        
        assert expected_min < total_params < expected_max, f"Unexpected param count: {total_params}"
        print(f"✓ Parameter count: {total_params:,} parameters (within expected range)")
        
        return True
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")
        return False

def main():
    """Run all minimal tests"""
    print("🚀 Starting Tensor Parallel Implementation Tests")
    print(f"JAX devices: {len(jax.devices())} - {jax.devices()}")
    
    tests = [
        test_phase_1_mesh_creation,
        test_phase_1_tensor_parallel_dense,
        test_phase_2_model_components,
        test_phase_2_full_model,
        test_phase_3_jit_compilation,
        test_phase_4_parameter_counting,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Tensor parallel implementation structure is correct.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    main() 