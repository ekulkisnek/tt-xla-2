#!/usr/bin/env python3
"""
Lightweight weight validation script to check bit-perfect weight matching.
Only loads weights, not full models, to conserve memory.
"""

import os
import json
import hashlib
import gc
from typing import Dict, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from simple_inference import process_safetensors_file, merge_param_dicts

def sha256_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of numpy array."""
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()

def load_pytorch_weights(model_path: str) -> Dict:
    """Load PyTorch weights efficiently."""
    print("Loading PyTorch model for weight extraction...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Extract state dict
    state_dict = model.state_dict()
    
    # Convert to numpy and store
    pt_weights = {}
    for key, tensor in state_dict.items():
        pt_weights[key] = tensor.cpu().numpy()
    
    # Cleanup model
    del model
    gc.collect()
    
    print(f"✓ Extracted {len(pt_weights)} PyTorch weight tensors")
    return pt_weights

def load_jax_weights(model_path: str) -> Dict:
    """Load JAX weights from safetensors."""
    print("Loading JAX weights from safetensors...")
    
    # Load weights from safetensors files
    param_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            print(f"  Processing {file}")
            file_params = process_safetensors_file(file_path, np.float32)  # Use float32
            param_dict = merge_param_dicts(param_dict, file_params)
    
    print(f"✓ Loaded JAX weights")
    return param_dict

def compare_weights(pt_weights: Dict, jax_weights: Dict) -> Tuple[int, int, int]:
    """Compare PyTorch and JAX weights with detailed analysis."""
    
    print("\n" + "="*60)
    print("WEIGHT COMPARISON ANALYSIS")
    print("="*60)
    
    # Key mapping for comparison
    key_mappings = {
        # Direct mappings
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"), 
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    # Load config to get layer count
    with open(os.path.join("../weights", "config.json"), 'r') as f:
        config = json.load(f)
    
    # Add layer mappings
    for i in range(config["num_hidden_layers"]):
        layer_prefix = f"model.layers.{i}"
        jax_prefix = f"layers_{i}"
        
        # Layer norms
        key_mappings[f"{layer_prefix}.input_layernorm.weight"] = (jax_prefix, "input_layernorm", "scale")
        key_mappings[f"{layer_prefix}.post_attention_layernorm.weight"] = (jax_prefix, "post_attention_layernorm", "scale")
        
        # Attention projections
        for proj in ["q", "k", "v", "o"]:
            key_mappings[f"{layer_prefix}.self_attn.{proj}_proj.weight"] = (jax_prefix, "self_attn", f"{proj}_proj", "kernel")
            if f"{layer_prefix}.self_attn.{proj}_proj.bias" in pt_weights:
                key_mappings[f"{layer_prefix}.self_attn.{proj}_proj.bias"] = (jax_prefix, "self_attn", f"{proj}_proj", "bias")
        
        # MLP projections
        for proj in ["gate", "up", "down"]:
            key_mappings[f"{layer_prefix}.mlp.{proj}_proj.weight"] = (jax_prefix, "mlp", f"{proj}_proj", "kernel")
    
    matches = 0
    mismatches = 0
    skipped = 0
    
    print(f"Comparing {len(key_mappings)} weight mappings...")
    
    for pt_key, jax_path in key_mappings.items():
        if pt_key not in pt_weights:
            print(f"⚠️  PyTorch key missing: {pt_key}")
            skipped += 1
            continue
            
        # Get PyTorch tensor
        pt_tensor = pt_weights[pt_key]
        
        # Get JAX tensor
        try:
            jax_tensor = jax_weights["params"]
            for path_part in jax_path:
                jax_tensor = jax_tensor[path_part]
            jax_tensor = np.array(jax_tensor)
        except KeyError:
            print(f"⚠️  JAX tensor not found for {pt_key} -> {jax_path}")
            skipped += 1
            continue
        
        # Handle transpose for weight matrices
        pt_tensor_processed = pt_tensor
        if "weight" in pt_key and ("proj" in pt_key or "lm_head" in pt_key):
            if "layernorm" not in pt_key and "norm.weight" not in pt_key:
                pt_tensor_processed = pt_tensor.T  # Transpose PyTorch to match JAX convention
        
        # Compare shapes
        if pt_tensor_processed.shape != jax_tensor.shape:
            print(f"❌ Shape mismatch {pt_key}:")
            print(f"   PT: {pt_tensor.shape} -> {pt_tensor_processed.shape}")
            print(f"   JAX: {jax_tensor.shape}")
            mismatches += 1
            continue
        
        # Hash comparison
        pt_hash = sha256_hash(pt_tensor_processed)
        jax_hash = sha256_hash(jax_tensor)
        
        if pt_hash == jax_hash:
            matches += 1
            if matches <= 5:  # Show first few matches
                print(f"✓ {pt_key}")
        else:
            mismatches += 1
            max_diff = np.max(np.abs(pt_tensor_processed - jax_tensor))
            print(f"❌ {pt_key}: hash mismatch (max_diff={max_diff:.2e})")
            
            # Additional debugging for critical weights
            if "embed_tokens" in pt_key or "lm_head" in pt_key:
                print(f"   Critical weight analysis:")
                print(f"   PT hash: {pt_hash[:16]}...")
                print(f"   JAX hash: {jax_hash[:16]}...")
                print(f"   PT stats: mean={np.mean(pt_tensor_processed):.6f}, std={np.std(pt_tensor_processed):.6f}")
                print(f"   JAX stats: mean={np.mean(jax_tensor):.6f}, std={np.std(jax_tensor):.6f}")
    
    return matches, mismatches, skipped

def main():
    model_path = "../weights"
    
    print("="*60)
    print("LIGHTWEIGHT WEIGHT VALIDATION")
    print("="*60)
    
    # Load weights from both frameworks
    pt_weights = load_pytorch_weights(model_path)
    jax_weights = load_jax_weights(model_path)
    
    # Compare weights
    matches, mismatches, skipped = compare_weights(pt_weights, jax_weights)
    
    # Final report
    print("\n" + "="*60)
    print("WEIGHT VALIDATION SUMMARY")
    print("="*60)
    
    total = matches + mismatches + skipped
    print(f"Total weights checked: {total}")
    print(f"✓ Perfect matches: {matches}")
    print(f"❌ Mismatches: {mismatches}")
    print(f"⚠️  Skipped: {skipped}")
    
    success_rate = matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
    print(f"Success rate: {success_rate:.1%}")
    
    if mismatches == 0:
        print("\n🎉 PERFECT WEIGHT PARITY: All weights match bit-perfectly!")
    elif mismatches <= 5:
        print("\n✅ MOSTLY GOOD: Only a few weight mismatches")
    else:
        print("\n⚠️  SIGNIFICANT WEIGHT ISSUES: Many weights don't match")
        print("   This explains the numerical differences in logits.")

if __name__ == "__main__":
    main() 