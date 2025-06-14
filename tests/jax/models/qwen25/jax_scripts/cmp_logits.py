#!/usr/bin/env python3
"""
Compare JAX Qwen2.5 logits against PyTorch reference.
Core script for Gates 1-3 of the parity ladder.
"""
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import json
import argparse
import gc

# Import our JAX implementation
import simple_inference as si

def main():
    parser = argparse.ArgumentParser(description="Compare JAX logits with PyTorch reference")
    parser.add_argument("--model_path", type=str, default="../weights", help="Path to model weights")
    parser.add_argument("--ids", type=str, required=True, help="Space-separated token IDs")
    parser.add_argument("--ref", type=str, required=True, help="Reference .npy file from PyTorch")
    parser.add_argument("--use_cache", action="store_true", help="Test cached generation (Gate 3)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    # Set JAX environment
    jax.config.update("jax_enable_x64", False)
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    
    # Parse token IDs
    ids = jnp.array([[int(i) for i in args.ids.split()]], dtype=jnp.int32)
    
    print(f"JAX Qwen2.5 Logits Comparison")
    print(f"Model path: {args.model_path}")
    print(f"Token IDs: {args.ids}")
    print(f"Reference file: {args.ref}")
    print(f"Use cache: {args.use_cache}")
    print(f"Dtype: {args.dtype}")
    print(f"Input shape: {ids.shape}")
    
    # Load config and create model
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading JAX model in {args.dtype}...")
    model = si.Qwen25ForCausalLM(config=config, dtype=dtype)
    params = si.load_params(model, args.model_path, dtype)
    
    if not args.use_cache:
        # Standard forward pass (Gates 1-2)
        print("Running standard forward pass...")
        outputs = model.apply(
            params,
            input_ids=ids,
            past_key_values=None,
            return_dict=True
        )
        logits = outputs["logits"]
    else:
        # Cached generation test (Gate 3)
        print("Running cached generation test...")
        if ids.shape[1] < 2:
            print("ERROR: Need at least 2 tokens for cache test")
            sys.exit(1)
            
        # First pass: generate cache with all but last token
        print(f"First pass: processing {ids.shape[1]-1} tokens...")
        outputs = model.apply(
            params,
            input_ids=ids[:, :-1],  # All but last token
            past_key_values=None,
            return_dict=True
        )
        past_key_values = outputs["past_key_values"]
        
        # Second pass: use cache with last token
        print(f"Second pass: processing 1 token with cache...")
        outputs = model.apply(
            params,
            input_ids=ids[:, -1:],  # Only last token
            past_key_values=past_key_values,
            return_dict=True
        )
        logits = outputs["logits"]
    
    print(f"JAX logits shape: {logits.shape}")
    print(f"JAX logits range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
    print(f"JAX logits std: {float(jnp.std(logits)):.6f}")
    print(f"JAX logits mean: {float(jnp.mean(logits)):.6f}")
    
    # Sample some logits for debugging
    if logits.size > 0:
        flat_logits = logits.flatten()
        sample_indices = jnp.linspace(0, flat_logits.size - 1, min(10, flat_logits.size)).astype(int)
        sample_logits = flat_logits[sample_indices]
        print(f"Sample JAX logits: {[float(x) for x in sample_logits]}")
    
    # Load PyTorch reference
    print(f"Loading PyTorch reference from {args.ref}...")
    ref_logits = np.load(args.ref)
    print(f"PyTorch reference shape: {ref_logits.shape}")
    print(f"PyTorch reference range: [{ref_logits.min():.6f}, {ref_logits.max():.6f}]")
    print(f"PyTorch reference std: {ref_logits.std():.6f}")
    print(f"PyTorch reference mean: {ref_logits.mean():.6f}")
    
    # Sample PyTorch logits for debugging
    if ref_logits.size > 0:
        flat_ref = ref_logits.flatten()
        sample_indices = np.linspace(0, flat_ref.size - 1, min(10, flat_ref.size)).astype(int)
        sample_ref = flat_ref[sample_indices]
        print(f"Sample PyTorch logits: {sample_ref.tolist()}")
    
    # Compare shapes
    if logits.shape != ref_logits.shape:
        print(f"ERROR: Shape mismatch! JAX: {logits.shape}, PyTorch: {ref_logits.shape}")
        sys.exit(1)
    
    # Compare values
    jax_logits_np = np.array(logits)
    diff = np.abs(jax_logits_np - ref_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Std absolute difference: {std_diff:.2e}")
    
    if np.abs(ref_logits).max() > 0:
        rel_error_max = max_diff / np.abs(ref_logits).max()
        rel_error_mean = mean_diff / (np.abs(ref_logits).mean() + 1e-10)
        print(f"Relative error (max): {rel_error_max:.2e}")
        print(f"Relative error (mean): {rel_error_mean:.2e}")
    
    # Gate-specific thresholds
    if args.use_cache:
        threshold = 1e-4  # Gate 3
        gate_name = "G-3 (Cached)"
    else:
        if ids.shape[1] == 1:
            threshold = 1e-5  # Gate 1
            gate_name = "G-1 (Single Token)"
        else:
            threshold = 1e-4  # Gate 2
            gate_name = "G-2 (Multi Token)"
    
    # Adjust threshold for bfloat16
    if args.dtype == "bfloat16":
        threshold = max(threshold, 1e-3)
    
    print(f"\nGate: {gate_name}")
    print(f"Threshold: {threshold:.2e}")
    
    if max_diff < threshold:
        print(f"✅ PASS: max_diff ({max_diff:.2e}) < threshold ({threshold:.2e})")
        result = True
    else:
        print(f"❌ FAIL: max_diff ({max_diff:.2e}) >= threshold ({threshold:.2e})")
        result = False
    
    # Additional analysis for failures
    if not result:
        print(f"\nFAILURE ANALYSIS:")
        print(f"Scale ratio (max): {np.max(jax_logits_np) / np.max(ref_logits):.6f}")
        print(f"Scale ratio (std): {np.std(jax_logits_np) / np.std(ref_logits):.6f}")
        
        # Find locations of largest differences
        diff_flat = diff.flatten()
        worst_indices = np.argsort(diff_flat)[-5:]  # 5 worst differences
        print(f"Worst differences at indices: {worst_indices}")
        for i, idx in enumerate(worst_indices):
            jax_val = float(jax_logits_np.flatten()[idx])
            pt_val = float(ref_logits.flatten()[idx])
            diff_val = float(diff_flat[idx])
            print(f"  {i+1}. idx={idx}: JAX={jax_val:.6f}, PT={pt_val:.6f}, diff={diff_val:.6f}")
    
    print(f"{'='*60}")
    
    # Cleanup
    del model, params, logits, outputs
    if 'past_key_values' in locals():
        del past_key_values
    gc.collect()
    jax.clear_caches()
    
    print("JAX cleanup completed.")
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main() 