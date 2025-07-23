# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test correctness between single and parallel execution."""

import sys
import jax
import jax.numpy as jnp
import transformers
import qwen_nnx


MODEL = "/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights"


def test_basic_functionality():
    """Test basic model functionality with current device configuration."""
    n_devices = len(jax.devices())
    print(f"Testing with {n_devices} devices")
    
    # Choose sharding based on device count
    if n_devices > 1:
        print("Using parallel sharding rules")
        sharding_rules = list({
            qwen_nnx.Axis.EMBED: None,
            qwen_nnx.Axis.MLP: "x",       # Shard across devices
            qwen_nnx.Axis.HEAD: "x",      # Shard across devices
            qwen_nnx.Axis.QHEAD: None,
            qwen_nnx.Axis.KVHEAD: None,
            qwen_nnx.Axis.VOCAB: None,
        }.items())
    else:
        print("Using single device sharding rules")
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
    
    print(f"Devices: {[str(d) for d in devices]}")
    
    # Load model
    print("Loading model...")
    model = qwen_nnx.QwenModel.load_from_hf_pt_model(
        MODEL,
        dtype=jnp.float32,
        param_dtype=jnp.bfloat16,  # Use bfloat16 to save memory
        mesh=mesh,
        sharding_rules=sharding_rules,
    )
    
    # Load tokenizer  
    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    
    # Simple test cases
    test_inputs = [
        "Hello",
        "The capital",
        "Python is",
    ]
    
    print("Testing inference...")
    results = []
    
    for i, text in enumerate(test_inputs):
        tokens = tokenizer(text, return_tensors="jax")["input_ids"]
        print(f"Input {i+1}: '{text}' (tokens: {tokens.shape})")
        
        # Run inference
        with mesh:
            output = model(tokens)
        
        # Get prediction
        next_token = jnp.argmax(output[0, -1])
        next_text = tokenizer.decode([next_token])
        
        result = {
            'input': text,
            'next_token': int(next_token),
            'next_text': next_text,
            'output_shape': output.shape
        }
        results.append(result)
        
        print(f"  -> Next token: {next_token} ('{next_text}')")
    
    # Output results in structured format
    print(f"\nRESULT_SUMMARY: {n_devices}_devices")
    for i, r in enumerate(results):
        print(f"OUTPUT_{i}: {r['next_token']},'{r['next_text']}'")
    
    print(f"SUCCESS: Model works with {n_devices} devices")
    return results


def main():
    """Main function."""
    try:
        results = test_basic_functionality()
        print("Test completed successfully!")
        return 0
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 