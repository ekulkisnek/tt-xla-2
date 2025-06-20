#!/usr/bin/env python3
"""
PHASE 3: One-token parity test between PyTorch and JAX Qwen2.5-7B-Instruct
"""
import numpy as np
import torch
import jax
import jax.numpy as jnp
import sys
import os
sys.path.append('.')

def test_parity():
    print("🔍 PHASE 3: PyTorch vs JAX Parity Test")
    print("="*50)
    
    # Test prompt
    prompt = "Q: 8-3="
    print(f"Test prompt: '{prompt}'")
    
    # Load PyTorch model
    print("\n📥 Loading PyTorch model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("✅ PyTorch model loaded")
        
        # PyTorch inference
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = pt_model(**inputs)
            pt_logits = outputs.logits[0, -1, :].numpy()  # Last token logits
        
        print(f"PyTorch logits shape: {pt_logits.shape}")
        print(f"Top 5 PyTorch predictions:")
        top_indices = np.argsort(pt_logits)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (id={idx}, logit={pt_logits[idx]:.3f})")
        
        # Clean up PyTorch model to save memory
        del pt_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"❌ PyTorch loading failed: {e}")
        print("Skipping PyTorch comparison...")
        return
    
    # Load JAX model
    print("\n📥 Loading JAX model...")
    try:
        from q25_jax_instruct import Qwen25ForCausalLM, load_params
        import json
        
        config_path = "../instruct_weights/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        params = load_params(model, "../instruct_weights", jnp.float32)
        
        print("✅ JAX model loaded")
        
        # JAX inference
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = jnp.array(inputs["input_ids"])
        
        # Simple forward pass (no attention mask complications)
        outputs = model.apply(
            params,
            input_ids=input_ids,
            return_dict=True
        )
        
        jax_logits = np.array(outputs["logits"][0, -1, :])  # Last token logits
        
        print(f"JAX logits shape: {jax_logits.shape}")
        print(f"Top 5 JAX predictions:")
        top_indices = np.argsort(jax_logits)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. '{token}' (id={idx}, logit={jax_logits[idx]:.3f})")
        
    except Exception as e:
        print(f"❌ JAX loading failed: {e}")
        return
    
    # Compare logits
    print(f"\n📊 Parity Analysis:")
    diff = np.abs(pt_logits - jax_logits)
    rms_diff = np.sqrt(np.mean(diff**2))
    max_diff = np.max(diff)
    
    print(f"  RMS difference: {rms_diff:.6f}")
    print(f"  Max difference: {max_diff:.6f}")
    
    # Check worst mismatches
    worst_indices = np.argsort(diff)[-5:][::-1]
    print(f"  Worst 5 mismatches:")
    for i, idx in enumerate(worst_indices):
        token = tokenizer.decode([idx])
        print(f"    {i+1}. '{token}' (id={idx}): PT={pt_logits[idx]:.3f}, JAX={jax_logits[idx]:.3f}, diff={diff[idx]:.3f})")
    
    # Pass/fail criteria
    print(f"\n🎯 PHASE 3 Results:")
    if rms_diff <= 1e-5:
        print(f"  ✅ PASS: RMS ≤ 1e-5 ({rms_diff:.6f})")
        print(f"  🎯 Weight loading is correct, proceed to PHASE 4")
    elif rms_diff <= 1e-3:
        print(f"  🔶 CLOSE: RMS ≤ 1e-3 ({rms_diff:.6f})")
        print(f"  🔧 Minor numerical differences, may still work")
    else:
        print(f"  ❌ FAIL: RMS > 1e-3 ({rms_diff:.6f})")
        print(f"  🔧 Significant differences - check model implementation")

if __name__ == "__main__":
    test_parity() 