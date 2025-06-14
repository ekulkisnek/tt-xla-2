#!/usr/bin/env python3
"""
Dump intermediate layer activations from PyTorch for debugging.
"""
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM
import gc

# Hook storage
activations = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            data = output[0].detach().cpu().float().numpy()
        else:
            data = output.detach().cpu().float().numpy()
        activations[name] = data
        print(f"Captured {name}: shape={data.shape}, std={data.std():.6f}")
    return hook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../weights")
    parser.add_argument("--prompt_ids", type=str, required=True)
    parser.add_argument("--layer", type=int, default=0, help="Layer to dump (0-27)")
    parser.add_argument("--out_dir", type=str, default="../tmp", help="Output directory")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    ids = torch.tensor([[int(i) for i in args.prompt_ids.split()]], dtype=torch.long)
    
    print(f"Loading PyTorch model in {args.dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model.eval()
    
    print(f"Hooking layer {args.layer} activations...")
    
    # Hook key components of the specified layer
    embed_layer = model.model.embed_tokens
    layer = model.model.layers[args.layer]
    final_norm = model.model.norm
    lm_head = model.lm_head
    
    # Register hooks for intermediate activations
    embed_handle = embed_layer.register_forward_hook(make_hook("embedding"))
    
    input_norm_handle = layer.input_layernorm.register_forward_hook(make_hook(f"layer{args.layer}_input_norm"))
    attn_handle = layer.self_attn.register_forward_hook(make_hook(f"layer{args.layer}_attention"))
    post_norm_handle = layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer{args.layer}_post_norm"))
    mlp_handle = layer.mlp.register_forward_hook(make_hook(f"layer{args.layer}_mlp"))
    
    if args.layer == 0:
        # For layer 0, also capture the final outputs
        final_norm_handle = final_norm.register_forward_hook(make_hook("final_norm"))
        lm_head_handle = lm_head.register_forward_hook(make_hook("lm_head"))
    
    print(f"Running forward pass with token IDs: {args.prompt_ids}")
    with torch.no_grad():
        outputs = model(ids)
        logits = outputs.logits
    
    print(f"Final logits shape: {logits.shape}")
    print(f"Final logits std: {logits.std():.6f}")
    print(f"Final logits range: [{logits.min():.6f}, {logits.max():.6f}]")
    
    # Save all captured activations
    for name, data in activations.items():
        filename = f"{args.out_dir}/pt_{name}.npy"
        np.save(filename, data)
        print(f"Saved {name}: {data.shape} to {filename}")
    
    # Clean up hooks
    embed_handle.remove()
    input_norm_handle.remove()
    attn_handle.remove()
    post_norm_handle.remove()
    mlp_handle.remove()
    
    if args.layer == 0:
        final_norm_handle.remove()
        lm_head_handle.remove()
    
    # Cleanup
    del model, logits
    gc.collect()
    print("PyTorch cleanup completed.")

if __name__ == "__main__":
    main() 