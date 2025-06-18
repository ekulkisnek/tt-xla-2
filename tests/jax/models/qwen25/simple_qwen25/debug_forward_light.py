#!/usr/bin/env python3
import numpy as np, json, argparse, os
from transformers import AutoTokenizer

def test_jax_only(model_path):
    """Test JAX model logits and save them"""
    import jax, jax.numpy as jnp
    from simple_inference import Qwen25ForCausalLM, load_params
    
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok("16 + 7", return_tensors="np")["input_ids"]
    
    print(f"Input: {tok.decode(ids[0])}")
    print(f"Token IDs: {ids[0].tolist()}")
    
    # JAX model
    print("Loading JAX model...")
    cfg = json.load(open(f"{model_path}/config.json"))
    print(f"Config rope_theta: {cfg.get('rope_theta', 'NOT_SET')}")
    
    jx_model = Qwen25ForCausalLM(cfg, dtype=jnp.float32)
    jx_params = load_params(jx_model, model_path, jnp.float32)
    
    # Get logits
    outputs = jx_model.apply(jx_params, input_ids=ids, return_dict=True)
    l_jx = outputs["logits"][0, -1]
    l_jx = np.asarray(l_jx)
    
    print(f"JAX logits shape: {l_jx.shape}")
    print(f"JAX logits range: [{np.min(l_jx):.3f}, {np.max(l_jx):.3f}]")
    
    # Save for comparison
    np.save("jax_logits.npy", l_jx)
    print("Saved JAX logits to jax_logits.npy")
    
    # Test a simple generation step
    print("\nTesting JAX generation step...")
    try:
        next_token = jnp.argmax(l_jx)
        next_token_id = int(next_token)
        next_text = tok.decode([next_token_id])
        print(f"Next token: {next_token_id} -> {repr(next_text)}")
        
        # Test second step
        ids_2 = np.array([[next_token_id]], dtype=np.int32)
        outputs_2 = jx_model.apply(jx_params, input_ids=ids_2, 
                                   past_key_values=outputs["past_key_values"], 
                                   return_dict=True)
        next_token_2 = int(jnp.argmax(outputs_2["logits"][0, -1]))
        next_text_2 = tok.decode([next_token_2])
        print(f"Second token: {next_token_2} -> {repr(next_text_2)}")
        
    except Exception as e:
        print(f"Generation test failed: {e}")

def test_pytorch_only(model_path):
    """Test PyTorch model logits and save them"""
    import torch
    from transformers import AutoModelForCausalLM
    
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok("16 + 7", return_tensors="pt").input_ids
    
    print(f"Input: {tok.decode(ids[0])}")
    print(f"Token IDs: {ids[0].tolist()}")
    
    # PyTorch model
    print("Loading PyTorch model...")
    torch.set_grad_enabled(False)
    pt = AutoModelForCausalLM.from_pretrained(model_path,
                                              torch_dtype=torch.float32,
                                              device_map="cpu",
                                              low_cpu_mem_usage=True)
    
    # Get logits
    l_pt = pt(ids).logits[0, -1].cpu().numpy()
    
    print(f"PyTorch logits shape: {l_pt.shape}")
    print(f"PyTorch logits range: [{np.min(l_pt):.3f}, {np.max(l_pt):.3f}]")
    
    # Save for comparison
    np.save("pytorch_logits.npy", l_pt)
    print("Saved PyTorch logits to pytorch_logits.npy")
    
    # Test generation
    print("\nTesting PyTorch generation...")
    next_token_id = int(torch.argmax(torch.tensor(l_pt)))
    next_text = tok.decode([next_token_id])
    print(f"Next token: {next_token_id} -> {repr(next_text)}")

def compare_logits():
    """Compare saved logits from JAX and PyTorch"""
    if not os.path.exists("jax_logits.npy") or not os.path.exists("pytorch_logits.npy"):
        print("❌ Missing logit files. Run --jax and --pytorch first.")
        return
        
    l_jx = np.load("jax_logits.npy")
    l_pt = np.load("pytorch_logits.npy")
    
    err = np.max(np.abs(l_pt - l_jx))
    print(f"max|Δ| = {err}")
    
    if err <= 1e-4:
        print("✅ PASS: Logits match within tolerance")
    else:
        print(f"❌ FAIL: Logits differ by {err}")
        print("Top 5 PyTorch tokens:", np.argsort(l_pt)[-5:][::-1])
        print("Top 5 JAX tokens:", np.argsort(l_jx)[-5:][::-1])
        
        # Diagnostic hints
        if abs(err - 0.24) < 0.01:
            print("🔍 Likely cause: RoPE theta mismatch (should be 1,000,000 for Qwen2.5)")
        elif 0.01 <= err <= 0.05:
            print("🔍 Likely cause: Causal mask alignment issue")
        elif err > 1.0:
            print("🔍 Likely cause: KV cache shape mismatch or major architectural issue")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--jax", action="store_true", help="Test JAX model")
    p.add_argument("--pytorch", action="store_true", help="Test PyTorch model")
    p.add_argument("--compare", action="store_true", help="Compare saved logits")
    args = p.parse_args()
    
    if args.jax:
        test_jax_only(args.model)
    elif args.pytorch:
        test_pytorch_only(args.model)
    elif args.compare:
        compare_logits()
    else:
        print("Usage: python debug_forward_light.py --model ../weights [--jax|--pytorch|--compare]")

if __name__ == "__main__":
    main() 