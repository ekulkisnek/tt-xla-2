#!/usr/bin/env python3
"""Quick Q matrix diagnostic - check if the problem is weight loading or elsewhere"""

import numpy as np
from safetensors import safe_open

def check_q_weights():
    print("🔍 Quick Q Matrix Diagnostic")
    print("="*50)
    
    # Check what's in the safetensors file vs what we're generating
    safetensors_file = "../instruct_weights/model-00001-of-00004.safetensors"
    q_key = "model.layers.0.self_attn.q_proj.weight"
    
    with safe_open(safetensors_file, framework="np") as f:
        q_weights = f.get_tensor(q_key)
        if hasattr(q_weights, 'dtype') and str(q_weights.dtype) == 'bfloat16':
            q_weights = q_weights.view(np.uint16).astype(np.float32)
        else:
            q_weights = q_weights.astype(np.float32)
        
        print(f"Q weights from safetensors:")
        print(f"  Shape: {q_weights.shape}")
        print(f"  First 3x3 corner:")
        print(f"    {q_weights[:3, :3]}")
        print(f"  Last 3x3 corner:")
        print(f"    {q_weights[-3:, -3:]}")
        
        # What would this look like transposed?
        q_transposed = q_weights.T
        print(f"\nQ weights transposed:")
        print(f"  Shape: {q_transposed.shape}")
        print(f"  First 3x3 corner:")
        print(f"    {q_transposed[:3, :3]}")
        print(f"  Last 3x3 corner:")
        print(f"    {q_transposed[-3:, -3:]}")

if __name__ == "__main__":
    check_q_weights() 