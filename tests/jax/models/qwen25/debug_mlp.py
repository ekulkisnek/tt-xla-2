#!/usr/bin/env python3
"""
Quick test to compare PyTorch vs JAX MLP outputs to check the sqrt(0.5) factor.
"""
import torch
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoModelForCausalLM
import json
import sys
import os
sys.path.append('jax_scripts')
import simple_inference as si

def test_mlp_sqrt_factor():
    """Test if the sqrt(0.5) factor should be present in MLP."""
    
    print("=== MLP SQRT(0.5) FACTOR TEST ===")
    
    # Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = AutoModelForCausalLM.from_pretrained("weights", torch_dtype=torch.bfloat16, device_map="cpu")
    
    # Get first layer MLP
    pt_mlp = pt_model.model.layers[0].mlp
    
    # Create test input
    batch_size, seq_len, hidden_size = 1, 1, 3584
    test_input_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    test_input_torch = torch.tensor(test_input_np, dtype=torch.bfloat16)
    
    # PyTorch MLP forward
    with torch.no_grad():
        pt_output = pt_mlp(test_input_torch)
    pt_output_np = pt_output.detach().cpu().float().numpy()
    
    print(f"PyTorch MLP output std: {pt_output_np.std():.6f}")
    print(f"PyTorch MLP output mean: {pt_output_np.mean():.6f}")
    
    del pt_model, pt_mlp
    
    # Load JAX model
    print("\nLoading JAX model...")
    jax.config.update("jax_enable_x64", False)
    
    with open("weights/config.json", 'r') as f:
        config = json.load(f)
    
    model = si.Qwen25ForCausalLM(config=config, dtype=jnp.bfloat16)
    params = si.load_params(model, "weights", jnp.bfloat16)
    
    # Get first layer MLP params
    mlp_params = params['params']['layers_0']['mlp']
    
    # Create JAX MLP instance and test with original sqrt(0.5)
    mlp = si.QwenMLP(config=config, dtype=jnp.bfloat16)
    test_input_jax = jnp.array(test_input_np, dtype=jnp.bfloat16)
    
    # Apply MLP with sqrt(0.5) factor
    jax_output_with_sqrt = mlp.apply({'params': mlp_params}, test_input_jax)
    jax_output_with_sqrt_np = np.array(jax_output_with_sqrt)
    
    print(f"JAX MLP output (with sqrt(0.5)) std: {float(jax_output_with_sqrt_np.std()):.6f}")
    print(f"JAX MLP output (with sqrt(0.5)) mean: {float(jax_output_with_sqrt_np.mean()):.6f}")
    
    # Calculate scale ratio
    scale_ratio_with_sqrt = float(jax_output_with_sqrt_np.std()) / float(pt_output_np.std())
    print(f"Scale ratio (JAX/PyTorch with sqrt): {scale_ratio_with_sqrt:.3f}")
    
    # Now test without sqrt(0.5) factor
    # Manually compute MLP without sqrt factor
    gate = jax.nn.silu(mlp.gate_proj.apply({'params': mlp_params['gate_proj']}, test_input_jax))
    up = mlp.up_proj.apply({'params': mlp_params['up_proj']}, test_input_jax)
    # Remove the sqrt(0.5) factor
    jax_output_no_sqrt = mlp.down_proj.apply({'params': mlp_params['down_proj']}, gate * up)
    jax_output_no_sqrt_np = np.array(jax_output_no_sqrt)
    
    print(f"JAX MLP output (NO sqrt(0.5)) std: {float(jax_output_no_sqrt_np.std()):.6f}")
    print(f"JAX MLP output (NO sqrt(0.5)) mean: {float(jax_output_no_sqrt_np.mean()):.6f}")
    
    scale_ratio_no_sqrt = float(jax_output_no_sqrt_np.std()) / float(pt_output_np.std())
    print(f"Scale ratio (JAX/PyTorch no sqrt): {scale_ratio_no_sqrt:.3f}")
    
    # Compare differences
    diff_with_sqrt = np.abs(jax_output_with_sqrt_np - pt_output_np)
    diff_no_sqrt = np.abs(jax_output_no_sqrt_np - pt_output_np)
    
    print(f"\n=== COMPARISON ===")
    print(f"Max diff WITH sqrt(0.5): {float(diff_with_sqrt.max()):.2e}")
    print(f"Max diff WITHOUT sqrt(0.5): {float(diff_no_sqrt.max()):.2e}")
    print(f"Mean diff WITH sqrt(0.5): {float(diff_with_sqrt.mean()):.2e}")
    print(f"Mean diff WITHOUT sqrt(0.5): {float(diff_no_sqrt.mean()):.2e}")
    
    # Verdict
    if diff_no_sqrt.max() < diff_with_sqrt.max():
        print("\n🎯 VERDICT: The sqrt(0.5) factor should be REMOVED!")
        print("   JAX implementation is closer to PyTorch WITHOUT the sqrt(0.5) factor.")
        return False  # sqrt factor should NOT be there
    else:
        print("\n✅ VERDICT: The sqrt(0.5) factor should be KEPT!")
        print("   JAX implementation is closer to PyTorch WITH the sqrt(0.5) factor.")
        return True   # sqrt factor should be there

if __name__ == "__main__":
    try:
        should_keep_sqrt = test_mlp_sqrt_factor()
        if not should_keep_sqrt:
            print("\n🚀 ACTION NEEDED: Remove sqrt(0.5) from MLP implementation!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 