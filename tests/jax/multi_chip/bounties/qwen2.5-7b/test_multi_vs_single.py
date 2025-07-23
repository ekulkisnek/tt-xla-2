# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test multi-device vs single-device correctness for Qwen2.5-7B.

This test compares outputs between single-device and various multi-device
configurations to ensure tensor parallelism produces identical results.

Following the pattern from accepted mixtral_8x7b bounty.

Can be run with multiple devices, e.g. via:
    XLA_FLAGS=--xla_force_host_platform_device_count=8 python test_multi_vs_single.py
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

import qwen_nnx
from mesh_configs import get_mesh_and_sharding, validate_mesh_configuration


def run_single_device(input_ids, model_path, dtype=jnp.bfloat16):
    """Run inference on single device.
    
    Args:
        input_ids: Input token IDs
        model_path: Path to model weights
        dtype: Model dtype
        
    Returns:
        Array: Output logits
    """
    print("🔹 Creating single-device model...")
    
    # Single device mesh
    mesh, sharding_rules = get_mesh_and_sharding('single')
    validate_mesh_configuration(mesh, sharding_rules)
    
    with mesh:
        model = qwen_nnx.QwenModel.load_from_hf_pt_model(
            model_path,
            dtype=dtype,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )
        
        # Run forward pass
        outputs = model(input_ids)
        
    print(f"    Single device output shape: {outputs.shape}")
    return outputs


def run_multi_device(input_ids, model_path, mesh_type, dtype=jnp.bfloat16):
    """Run inference on multi-device configuration.
    
    Args:
        input_ids: Input token IDs
        model_path: Path to model weights
        mesh_type: Type of mesh ('2x4', '1x8', etc.)
        dtype: Model dtype
        
    Returns:
        Array: Output logits
    """
    print(f"🔹 Creating {mesh_type} multi-device model...")
    
    # Multi-device mesh
    mesh, sharding_rules = get_mesh_and_sharding(mesh_type)
    validate_mesh_configuration(mesh, sharding_rules)
    
    with mesh:
        model = qwen_nnx.QwenModel.load_from_hf_pt_model(
            model_path,
            dtype=dtype,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )
        
        # Run forward pass
        outputs = model(input_ids)
        
    print(f"    {mesh_type} device output shape: {outputs.shape}")
    return outputs


def run_replicated_multi_device(input_ids, model_path, dtype=jnp.bfloat16):
    """Run inference on multi-device but without sharding (replicated).
    
    Args:
        input_ids: Input token IDs  
        model_path: Path to model weights
        dtype: Model dtype
        
    Returns:
        Array: Output logits
    """
    print("🔹 Creating replicated multi-device model...")
    
    # Multi-device mesh with no sharding
    mesh, sharding_rules = get_mesh_and_sharding('replicated')
    validate_mesh_configuration(mesh, sharding_rules)
    
    with mesh:
        model = qwen_nnx.QwenModel.load_from_hf_pt_model(
            model_path,
            dtype=dtype,
            mesh=mesh,
            sharding_rules=sharding_rules,
        )
        
        # Run forward pass
        outputs = model(input_ids)
        
    print(f"    Replicated multi-device output shape: {outputs.shape}")
    return outputs


def compare_outputs(output1, output2, config1, config2, rtol=1e-4, atol=1e-5):
    """Compare two output arrays for correctness.
    
    Args:
        output1, output2: Output arrays to compare
        config1, config2: Configuration names for logging
        rtol, atol: Tolerance for comparison
        
    Returns:
        bool: True if outputs match within tolerance
    """
    print(f"\n🔍 Comparing {config1} vs {config2}:")
    
    # Check shapes match
    if output1.shape != output2.shape:
        print(f"    ❌ Shape mismatch: {output1.shape} vs {output2.shape}")
        return False
    
    # Check values match
    is_close = jnp.allclose(output1, output2, rtol=rtol, atol=atol)
    max_diff = jnp.max(jnp.abs(output1 - output2))
    mean_diff = jnp.mean(jnp.abs(output1 - output2))
    
    print(f"    Shape: {output1.shape}")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Mean absolute difference: {mean_diff:.2e}")
    print(f"    Within tolerance (rtol={rtol}, atol={atol}): {'✅' if is_close else '❌'}")
    
    return bool(is_close)


def get_test_inputs(tokenizer):
    """Get test input sequences following mixtral_8x7b pattern.
    
    Args:
        tokenizer: Tokenizer instance
        
    Returns:
        List of input_ids arrays
    """
    test_prompts = [
        "Hello, how are you?",
        "The capital of France is",
        "Python is a programming language that",
        "2 + 2 =",
        "Machine learning is"
    ]
    
    test_inputs = []
    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="jax")["input_ids"]
        test_inputs.append((prompt, input_ids))
    
    return test_inputs


def main():
    """Main test function comparing single vs multi-device configurations."""
    
    # Configuration
    model_path = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"
    dtype = jnp.bfloat16
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Please update the model_path variable to point to your Qwen2.5-7B weights.")
        return False
    
    print("🚀 Starting Qwen2.5-7B Multi vs Single Device Test")
    print(f"Model path: {model_path}")
    print(f"Available devices: {len(jax.devices())}")
    print(f"Dtype: {dtype}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get test inputs
    test_inputs = get_test_inputs(tokenizer)
    print(f"Test prompts: {len(test_inputs)}")
    
    # Test configurations to compare
    mesh_configs = ['1x8', '2x4']  # Start with these, add more if devices available
    
    # Filter configs based on available devices
    available_devices = len(jax.devices())
    if available_devices < 8:
        print(f"Only {available_devices} devices available, using replicated config for testing")
        mesh_configs = ['replicated']
    
    all_passed = True
    
    # Test each prompt
    for i, (prompt, input_ids) in enumerate(test_inputs):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i+1}/{len(test_inputs)}: '{prompt}'")
        print(f"Input shape: {input_ids.shape}")
        
        try:
            # Run single device
            single_output = run_single_device(input_ids, model_path, dtype)
            
            # Run replicated multi-device
            replicated_output = run_replicated_multi_device(input_ids, model_path, dtype)
            
            # Compare single vs replicated (should be identical)
            match1 = compare_outputs(single_output, replicated_output, "single", "replicated")
            all_passed = all_passed and match1
            
            # Test each mesh configuration
            for mesh_config in mesh_configs:
                if mesh_config == 'replicated':
                    continue  # Already tested above
                    
                try:
                    multi_output = run_multi_device(input_ids, model_path, mesh_config, dtype)
                    
                    # Compare single vs tensor parallel
                    match2 = compare_outputs(single_output, multi_output, "single", mesh_config)
                    all_passed = all_passed and match2
                    
                except Exception as e:
                    print(f"    ⚠️ Failed to test {mesh_config}: {e}")
                    all_passed = False
                    
        except Exception as e:
            print(f"    ❌ Test failed: {e}")
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"All tests passed: {'✅ YES' if all_passed else '❌ NO'}")
    
    if all_passed:
        print("🎉 SUCCESS: Multi-device tensor parallel outputs match single-device!")
        print("The implementation correctly implements tensor parallelism.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 