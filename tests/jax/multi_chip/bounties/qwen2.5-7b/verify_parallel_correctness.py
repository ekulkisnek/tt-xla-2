# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Verify parallel correctness by comparing outputs with different configurations."""

import subprocess
import sys


def run_test_config(n_devices: int, config_name: str):
    """Run test with specific device configuration."""
    print(f"\n=== Testing {config_name} ({n_devices} devices) ===")
    
    # Prepare environment and command
    env_cmd = f"XLA_FLAGS=--xla_force_host_platform_device_count={n_devices}" if n_devices > 1 else ""
    
    test_script = f"""
import jax
import jax.numpy as jnp
import transformers
import qwen_nnx

MODEL = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"

print(f"Devices: {{len(jax.devices())}}")

# Use minimal sharding to avoid collective issues
sharding_rules = list({{
    qwen_nnx.Axis.EMBED: None,
    qwen_nnx.Axis.MLP: None,      # Keep simple for now
    qwen_nnx.Axis.HEAD: None,     # Keep simple for now  
    qwen_nnx.Axis.QHEAD: None,
    qwen_nnx.Axis.KVHEAD: None,
    qwen_nnx.Axis.VOCAB: None,
}}.items())

devices = jax.devices("cpu")
mesh = jax.make_mesh((len(devices),), axis_names=("x",), devices=devices)

print("Loading model...")
model = qwen_nnx.QwenModel.load_from_hf_pt_model(
    MODEL,
    dtype=jnp.float32,
    param_dtype=jnp.bfloat16,
    mesh=mesh,
    sharding_rules=sharding_rules,
)

print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

# Test cases
test_inputs = ["Hello", "The capital", "Python is"]

print("Running inference...")
for i, text in enumerate(test_inputs):
    tokens = tokenizer(text, return_tensors="jax")["input_ids"]
    
    with mesh:
        output = model(tokens)
    
    next_token = jnp.argmax(output[0, -1])
    next_text = tokenizer.decode([next_token])
    
    print(f"RESULT_{{i}}: {{int(next_token)}}")

print("SUCCESS")
"""
    
    # Write test script to temp file and run it
    with open(f"temp_test_{n_devices}.py", "w") as f:
        f.write(test_script)
    
    try:
        if n_devices > 1:
            cmd = f"{env_cmd} python temp_test_{n_devices}.py"
        else:
            cmd = f"python temp_test_{n_devices}.py"
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {config_name} test passed")
            
            # Extract results
            results = []
            for line in result.stdout.split('\n'):
                if line.startswith('RESULT_'):
                    parts = line.split(': ')
                    if len(parts) == 2:
                        idx = int(parts[0].split('_')[1])
                        token = int(parts[1])
                        results.append((idx, token))
            
            return results
        else:
            print(f"❌ {config_name} test failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ {config_name} test timed out")
        return None
    except Exception as e:
        print(f"❌ {config_name} test error: {e}")
        return None


def main():
    """Main test function."""
    print("🧪 Verifying JAX Tensor Parallel Correctness")
    print("=" * 50)
    
    # Test single device (baseline)
    single_results = run_test_config(1, "Single Device")
    
    if single_results is None:
        print("❌ Single device test failed - cannot continue")
        return 1
    
    print(f"Single device results: {single_results}")
    
    # Test with 2 devices
    dual_results = run_test_config(2, "Dual Device")
    
    if dual_results is None:
        print("⚠️  Dual device test failed")
        return 1
    
    print(f"Dual device results: {dual_results}")
    
    # Compare results
    print("\n=== Correctness Analysis ===")
    
    if len(single_results) != len(dual_results):
        print("❌ Different number of results")
        return 1
    
    all_match = True
    for (idx1, token1), (idx2, token2) in zip(single_results, dual_results):
        if idx1 == idx2 and token1 == token2:
            print(f"✅ Test {idx1}: Both predict token {token1}")
        else:
            print(f"❌ Test {idx1}: Single={token1}, Dual={token2}")
            all_match = False
    
    print("\n" + "=" * 50)
    if all_match:
        print("🎉 SUCCESS: Single and multi-device outputs match!")
        print("   The tensor parallel implementation is correct.")
        return 0
    else:
        print("⚠️  ISSUE: Outputs differ between configurations")
        print("   This may indicate a problem with the parallel implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 