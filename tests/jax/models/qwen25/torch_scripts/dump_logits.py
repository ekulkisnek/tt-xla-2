#!/usr/bin/env python3
"""
Dump PyTorch Qwen2.5 logits to .npy files for JAX comparison.
Supports single-token, multi-token, and cached generation.
"""
import os
import sys
import torch
import json
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def main():
    parser = argparse.ArgumentParser(description="Dump PyTorch Qwen2.5 logits")
    parser.add_argument("--model_path", type=str, default="../weights", help="Path to model weights")
    parser.add_argument("--prompt_ids", type=str, required=True, help="Space-separated token IDs")
    parser.add_argument("--next_id", type=int, help="Next token ID for cached generation test")
    parser.add_argument("--out", type=str, required=True, help="Output .npy file path")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    # Set memory-efficient environment
    torch.set_grad_enabled(False)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    
    # Parse token IDs
    ids = torch.tensor([[int(i) for i in args.prompt_ids.split()]], dtype=torch.long)
    
    print(f"Loading PyTorch model in {args.dtype}...")
    print(f"Model path: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model.eval()
    
    print(f"Input IDs: {args.prompt_ids}")
    print(f"Input shape: {ids.shape}")
    
    if args.next_id is None:
        # Standard forward pass
        print("Running standard forward pass...")
        with torch.no_grad():
            outputs = model(ids, use_cache=False)
            logits = outputs.logits
    else:
        # Cached generation test
        print(f"Running cached generation test with next_id={args.next_id}...")
        with torch.no_grad():
            # First pass: generate cache
            outputs = model(ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Second pass: use cache with next token
            next_ids = torch.tensor([[args.next_id]], dtype=torch.long)
            outputs = model(next_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.6f}, {logits.max():.6f}]")
    print(f"Logits std: {logits.std():.6f}")
    print(f"Logits mean: {logits.mean():.6f}")
    
    # Sample some logits for debugging
    if logits.numel() > 0:
        flat_logits = logits.flatten()
        sample_indices = torch.linspace(0, flat_logits.size(0) - 1, min(10, flat_logits.size(0))).long()
        sample_logits = flat_logits[sample_indices]
        print(f"Sample logits: {sample_logits.tolist()}")
    
    # Save to numpy
    logits_np = logits.cpu().numpy()
    np.save(args.out, logits_np)
    print(f"Saved to {args.out}")
    print(f"Saved array shape: {logits_np.shape}")
    
    # Cleanup
    del model, logits, logits_np
    if 'outputs' in locals():
        del outputs
    if 'past_key_values' in locals():
        del past_key_values
    gc.collect()
    
    print("PyTorch model cleanup completed.")

if __name__ == "__main__":
    main() 