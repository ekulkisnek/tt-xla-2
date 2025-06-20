#!/usr/bin/env python3
"""
Quick test of real tensor parallelism - just forward pass verification
"""

import os
# CRITICAL: Set XLA_FLAGS to create multiple JAX devices BEFORE importing JAX
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec, Mesh

def test_real_tensor_parallelism():
    """Test that we have real tensor parallelism working"""
    
    print("🔥 Testing Real Tensor Parallelism")
    print("=" * 50)
    
    # Setup devices
    devices = jax.devices()
    print(f'JAX Devices: {len(devices)}')
    for i, device in enumerate(devices):
        print(f'  Device {i}: {device}')
    
    # Create mesh
    if len(devices) >= 8:
        mesh_devices = np.array(devices[:8]).reshape(2, 4)
        mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
    elif len(devices) >= 4:
        mesh_devices = np.array(devices[:4]).reshape(1, 4)
        mesh = Mesh(mesh_devices, axis_names=('data', 'model'))
    else:
        print("❌ Not enough devices for tensor parallelism")
        return
    
    print(f'🚀 Mesh: {mesh}')
    print(f'📊 Model parallel size: {mesh.shape["model"]}')
    print(f'🔄 Data parallel size: {mesh.shape["data"]}')
    
    # Test tensor operations with sharding
    with mesh:
        # Create some test data
        x = jnp.ones((4, 1024))  # batch=4, features=1024
        
        # Create sharded weight matrix
        weight = jax.random.normal(jax.random.PRNGKey(42), (1024, 2048))
        
        # Apply sharding constraint - shard the weight matrix along the model axis
        weight_spec = PartitionSpec(None, 'model')  # shard output dimension
        weight_sharded = jax.lax.with_sharding_constraint(weight, weight_spec)
        
        # Apply data sharding to input
        data_spec = PartitionSpec('data', None)  # shard batch dimension
        x_sharded = jax.lax.with_sharding_constraint(x, data_spec)
        
        print("\n🎯 Testing sharded matrix multiplication...")
        
        # Define a simple computation that requires all-reduce
        @jax.jit
        def tensor_parallel_matmul(x, weight):
            # Forward pass: x @ weight (input: [batch, in_features], weight: [in_features, out_features])
            y = jnp.dot(x, weight)
            
            # All-reduce across model dimension to get final result
            y = jax.lax.psum(y, axis_name='model')
            
            return y
        
        # Run the computation
        try:
            result = tensor_parallel_matmul(x_sharded, weight_sharded)
            print(f"✅ Tensor parallel computation successful!")
            print(f"   Input shape: {x.shape}")
            print(f"   Weight shape: {weight.shape}")
            print(f"   Output shape: {result.shape}")
            print(f"   Output mean: {jnp.mean(result):.6f}")
            
            # Verify sharding
            print(f"\n📍 Sharding verification:")
            print(f"   Input sharding: {x_sharded.sharding}")
            print(f"   Weight sharding: {weight_sharded.sharding}")
            print(f"   Output sharding: {result.sharding}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in tensor parallel computation: {e}")
            return False

def test_model_forward_pass():
    """Test a simple forward pass with the real TP model"""
    print("\n🧪 Testing Model Forward Pass...")
    
    try:
        # Import our real TP model
        from real_tp_qwen25 import Qwen25ForCausalLM, setup_devices
        import json
        
        # Setup
        mesh = setup_devices()
        
        # Minimal config for testing
        config = {
            "vocab_size": 1000,  # Small vocab for testing
            "hidden_size": 512,  # Small hidden size
            "num_hidden_layers": 2,  # Just 2 layers
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 2048,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5
        }
        
        with mesh:
            # Create model
            model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
            
            # Initialize with random weights
            dummy_input = jnp.ones((1, 4), dtype=jnp.int32)  # batch=1, seq=4
            params = model.init(jax.random.PRNGKey(0), dummy_input)
            
            print("✅ Model initialization successful!")
            
            # Test forward pass
            @jax.jit
            def forward_pass(params, input_ids):
                return model.apply(params, input_ids=input_ids, return_dict=True)
            
            # Run forward pass
            outputs = forward_pass(params, dummy_input)
            logits = outputs["logits"]
            
            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Logits mean: {jnp.mean(logits):.6f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test basic tensor parallelism
    tp_success = test_real_tensor_parallelism()
    
    # Test model forward pass
    model_success = test_model_forward_pass()
    
    print("\n" + "=" * 50)
    if tp_success and model_success:
        print("🎉 ALL TESTS PASSED - Real Tensor Parallelism Working!")
    elif tp_success:
        print("⚠️  Basic TP working, but model test failed")
    else:
        print("❌ Tensor parallelism tests failed") 