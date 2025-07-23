# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test tensor parallel sharding correctness."""

import jax
import jax.numpy as jnp
import transformers
import qwen_nnx

MODEL = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"

def test_with_sharding_config(sharding_config, config_name):
    """Test model with specific sharding configuration."""
    print(f"\n=== Testing {config_name} ===")
    
    devices = jax.devices("cpu")
    mesh = jax.make_mesh((len(devices),), axis_names=("x",), devices=devices)
    
    print(f"Devices: {len(devices)} ({[str(d) for d in devices]})")
    print(f"Sharding config: {dict(sharding_config)}")
    
    try:
        # Load model
        print("Loading model...")
        model = qwen_nnx.QwenModel.load_from_hf_pt_model(
            MODEL,
            dtype=jnp.float32,
            param_dtype=jnp.bfloat16,
            mesh=mesh,
            sharding_rules=sharding_config,
        )
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        
        # Test inputs
        test_inputs = ["Hello", "The capital", "Python is"]
        
        print("Running inference...")
        results = []
        
        for i, text in enumerate(test_inputs):
            tokens = tokenizer(text, return_tensors="jax")["input_ids"]
            
            with mesh:
                output = model(tokens)
            
            next_token = jnp.argmax(output[0, -1])
            next_text = tokenizer.decode([next_token])
            
            results.append({
                'input': text,
                'token': int(next_token),
                'text': next_text,
                'shape': output.shape
            })
            
            print(f"  '{text}' -> {next_token} ('{next_text}')")
        
        print(f"✅ {config_name} completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ {config_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Testing JAX Tensor Parallel Sharding")
    print("=" * 50)
    
    n_devices = len(jax.devices())
    print(f"Available devices: {n_devices}")
    
    if n_devices < 2:
        print("⚠️  Need at least 2 devices for tensor parallel testing")
        print("   Run with XLA_FLAGS=--xla_force_host_platform_device_count=2")
        return 1
    
    # Configuration 1: No sharding (baseline)
    no_sharding = list({
        qwen_nnx.Axis.EMBED: None,
        qwen_nnx.Axis.MLP: None,
        qwen_nnx.Axis.HEAD: None,
        qwen_nnx.Axis.QHEAD: None,
        qwen_nnx.Axis.KVHEAD: None,
        qwen_nnx.Axis.VOCAB: None,
    }.items())
    
    results_no_shard = test_with_sharding_config(no_sharding, "No Sharding")
    
    if results_no_shard is None:
        print("❌ Baseline test failed")
        return 1
    
    # Configuration 2: Tensor parallel sharding
    tensor_parallel = list({
        qwen_nnx.Axis.EMBED: None,
        qwen_nnx.Axis.MLP: "x",      # Shard MLP across devices
        qwen_nnx.Axis.HEAD: "x",     # Shard attention heads
        qwen_nnx.Axis.QHEAD: None,
        qwen_nnx.Axis.KVHEAD: None,
        qwen_nnx.Axis.VOCAB: None,
    }.items())
    
    results_tp = test_with_sharding_config(tensor_parallel, "Tensor Parallel")
    
    if results_tp is None:
        print("❌ Tensor parallel test failed")
        return 1
    
    # Compare results
    print("\n=== Correctness Comparison ===")
    
    all_match = True
    for i, (r1, r2) in enumerate(zip(results_no_shard, results_tp)):
        if r1['token'] == r2['token']:
            print(f"✅ Test {i}: Both predict token {r1['token']} ('{r1['text']}')")
        else:
            print(f"❌ Test {i}: No-shard={r1['token']} ('{r1['text']}'), TP={r2['token']} ('{r2['text']}')")
            all_match = False
    
    print("\n" + "=" * 50)
    if all_match:
        print("🎉 SUCCESS: Tensor parallel sharding produces correct results!")
        print("   - No sharding and tensor parallel outputs match")
        print("   - The implementation correctly handles distributed computation")
        return 0
    else:
        print("⚠️  ISSUE: Outputs differ between sharding configurations")
        print("   This indicates a problem with the tensor parallel implementation")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 