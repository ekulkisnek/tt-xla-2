#!/usr/bin/env python3
"""
Compare intermediate layer activations between JAX and PyTorch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import argparse
import json
import gc
import os
import simple_inference as si

def extract_layer_activations(model, params, input_ids, layer_idx=0):
    """Extract intermediate activations from JAX model."""
    print(f"Extracting JAX activations for layer {layer_idx}...")
    
    activations = {}
    
    # Get embeddings
    embed_tokens = model.embed_tokens.apply({'params': params['params']['embed_tokens']}, input_ids)
    activations['embedding'] = np.array(embed_tokens)
    print(f"JAX embedding: shape={embed_tokens.shape}, std={float(jnp.std(embed_tokens)):.6f}")
    
    # Process through layers up to the target layer
    hidden_states = embed_tokens
    
    for i in range(layer_idx + 1):
        layer = model.layers[i]
        layer_params = params['params'][f'layers_{i}']
        
        # Input norm
        residual = hidden_states
        hidden_states_norm = layer.input_layernorm.apply(
            {'params': layer_params['input_layernorm']}, 
            hidden_states
        )
        
        if i == layer_idx:
            activations[f'layer{i}_input_norm'] = np.array(hidden_states_norm)
            print(f"JAX layer{i}_input_norm: shape={hidden_states_norm.shape}, std={float(jnp.std(hidden_states_norm)):.6f}")
        
        # Self attention
        batch, seq, _ = hidden_states_norm.shape
        position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
        cos, sin = si.compute_cos_sin_cache(position_ids, layer.self_attn.head_dim, layer.self_attn.rope_theta)
        
        attn_output, _ = layer.self_attn(
            hidden_states_norm,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=None,
            cos=cos,
            sin=sin
        )
        
        if i == layer_idx:
            activations[f'layer{i}_attention'] = np.array(attn_output)
            print(f"JAX layer{i}_attention: shape={attn_output.shape}, std={float(jnp.std(attn_output)):.6f}")
        
        hidden_states = residual + attn_output
        
        # Post attention norm and MLP
        residual = hidden_states
        hidden_states_norm = layer.post_attention_layernorm.apply(
            {'params': layer_params['post_attention_layernorm']}, 
            hidden_states
        )
        
        if i == layer_idx:
            activations[f'layer{i}_post_norm'] = np.array(hidden_states_norm)
            print(f"JAX layer{i}_post_norm: shape={hidden_states_norm.shape}, std={float(jnp.std(hidden_states_norm)):.6f}")
        
        mlp_output = layer.mlp(hidden_states_norm)
        
        if i == layer_idx:
            activations[f'layer{i}_mlp'] = np.array(mlp_output)
            print(f"JAX layer{i}_mlp: shape={mlp_output.shape}, std={float(jnp.std(mlp_output)):.6f}")
        
        hidden_states = residual + mlp_output
    
    # If processing layer 0, also get final norm and lm_head
    if layer_idx == 0:
        # Process through all remaining layers (simplified for debugging)
        for i in range(layer_idx + 1, model.num_layers):
            layer = model.layers[i]
            layer_params = params['params'][f'layers_{i}']
            
            # Simplified layer processing
            residual = hidden_states
            hidden_states_norm = layer.input_layernorm.apply(
                {'params': layer_params['input_layernorm']}, 
                hidden_states
            )
            
            batch, seq, _ = hidden_states_norm.shape
            position_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
            cos, sin = si.compute_cos_sin_cache(position_ids, layer.self_attn.head_dim, layer.self_attn.rope_theta)
            
            attn_output, _ = layer.self_attn(
                hidden_states_norm,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=None,
                cos=cos,
                sin=sin
            )
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states_norm = layer.post_attention_layernorm.apply(
                {'params': layer_params['post_attention_layernorm']}, 
                hidden_states
            )
            mlp_output = layer.mlp(hidden_states_norm)
            hidden_states = residual + mlp_output
        
        # Final norm
        final_norm_output = model.norm.apply({'params': params['params']['norm']}, hidden_states)
        activations['final_norm'] = np.array(final_norm_output)
        print(f"JAX final_norm: shape={final_norm_output.shape}, std={float(jnp.std(final_norm_output)):.6f}")
        
        # LM head
        logits = model.lm_head.apply({'params': params['params']['lm_head']}, final_norm_output)
        activations['lm_head'] = np.array(logits)
        print(f"JAX lm_head: shape={logits.shape}, std={float(jnp.std(logits)):.6f}")
    
    return activations

def compare_activations(jax_acts, ref_dir, layer_idx=0):
    """Compare JAX activations with PyTorch references."""
    print(f"\n{'='*60}")
    print(f"ACTIVATION COMPARISON RESULTS")
    print(f"{'='*60}")
    
    failed_components = []
    
    for name, jax_data in jax_acts.items():
        ref_file = f"{ref_dir}/pt_{name}.npy"
        
        if not os.path.exists(ref_file):
            print(f"⚠️  Missing reference file: {ref_file}")
            continue
        
        ref_data = np.load(ref_file)
        
        if jax_data.shape != ref_data.shape:
            print(f"❌ {name}: Shape mismatch! JAX: {jax_data.shape}, PT: {ref_data.shape}")
            failed_components.append(name)
            continue
        
        diff = np.abs(jax_data - ref_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        scale_ratio = np.std(jax_data) / np.std(ref_data) if np.std(ref_data) > 0 else 0
        
        threshold = 1e-3  # Lenient threshold for intermediate activations
        
        if max_diff < threshold:
            print(f"✅ {name}: max_diff={max_diff:.2e}, scale_ratio={scale_ratio:.3f}")
        else:
            print(f"❌ {name}: max_diff={max_diff:.2e}, scale_ratio={scale_ratio:.3f} (FAIL)")
            failed_components.append(name)
    
    print(f"\n{'='*60}")
    if failed_components:
        print(f"❌ FAILED COMPONENTS: {failed_components}")
        print(f"First failure: {failed_components[0]} - investigate this component!")
    else:
        print(f"✅ ALL COMPONENTS PASSED!")
    
    return failed_components

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../weights")
    parser.add_argument("--ids", type=str, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--ref_dir", type=str, default="../tmp")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    # Set JAX environment
    jax.config.update("jax_enable_x64", False)
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    
    # Parse token IDs
    ids = jnp.array([[int(i) for i in args.ids.split()]], dtype=jnp.int32)
    
    print(f"JAX Layer Activation Comparison")
    print(f"Token IDs: {args.ids}")
    print(f"Layer: {args.layer}")
    print(f"Dtype: {args.dtype}")
    
    # Load config and create model
    config_path = os.path.join(args.model_path, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading JAX model...")
    model = si.Qwen25ForCausalLM(config=config, dtype=dtype)
    params = si.load_params(model, args.model_path, dtype)
    
    # Extract JAX activations
    jax_activations = extract_layer_activations(model, params, ids, args.layer)
    
    # Compare with PyTorch references
    failed_components = compare_activations(jax_activations, args.ref_dir, args.layer)
    
    # Cleanup
    del model, params
    gc.collect()
    jax.clear_caches()
    
    print("JAX cleanup completed.")
    
    # Exit with error code if any components failed
    return len(failed_components) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 