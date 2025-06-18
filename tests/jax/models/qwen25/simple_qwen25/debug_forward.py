#!/usr/bin/env python3
import numpy as np, torch, jax, jax.numpy as jnp, json, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params   # your file

def main(p):
    tok = AutoTokenizer.from_pretrained(p.model)
    # --- prompt: "16 + 7" just so it has ≥2 tokens in vocab ---
    ids_pt = tok("16 + 7", return_tensors="pt").input_ids
    ids_jx = ids_pt.detach().cpu().numpy()

    print(f"Input: {tok.decode(ids_pt[0])}")
    print(f"Token IDs: {ids_pt[0].tolist()}")

    # PyTorch ----------------------------------------------------------------
    print("Loading PyTorch model...")
    torch.set_grad_enabled(False)
    pt = AutoModelForCausalLM.from_pretrained(p.model,
                                              torch_dtype=torch.float32,
                                              device_map="cpu")
    l_pt = pt(ids_pt).logits[0, -1].cpu().numpy()

    # JAX --------------------------------------------------------------------
    print("Loading JAX model...")
    cfg = json.load(open(f"{p.model}/config.json"))
    jx_model  = Qwen25ForCausalLM(cfg, dtype=jnp.float32)
    jx_params = load_params(jx_model, p.model, jnp.float32)
    l_jx = jx_model.apply(jx_params, input_ids=ids_jx, return_dict=False)[0][0, -1]
    l_jx = np.asarray(l_jx)

    # diff -------------------------------------------------------------------
    err = np.max(np.abs(l_pt - l_jx))
    print(f"PyTorch logits shape: {l_pt.shape}")
    print(f"JAX logits shape: {l_jx.shape}")
    print(f"PyTorch logits range: [{np.min(l_pt):.3f}, {np.max(l_pt):.3f}]")
    print(f"JAX logits range: [{np.min(l_jx):.3f}, {np.max(l_jx):.3f}]")
    print(f"max|Δ| = {err}")
    
    if err <= 1e-4:
        print("✅ PASS: Logits match within tolerance")
        
        # Greedy 10 tokens ----------------------------------------------------------
        print("\nTesting greedy generation...")
        prompt = "Why do leaves fall?"
        ids_pt = tok(prompt, return_tensors="pt").input_ids
        ids_jx = ids_pt.detach().cpu().numpy()

        pt_ids = pt.generate(ids_pt, max_new_tokens=10, temperature=0.0, top_k=1)[0]
        jx_out = []
        pkv = None
        cur   = ids_jx
        for i in range(10):
            lgts, pkv = jx_model.apply(jx_params, input_ids=cur, past_key_values=pkv, return_dict=False)
            nxt = int(np.argmax(np.asarray(lgts[0,-1])))
            jx_out.append(nxt)
            cur = np.array([[nxt]], dtype=np.int32)
            print(f"  Step {i}: token={nxt}, text={repr(tok.decode([nxt]))}")

        pt_text = tok.decode(pt_ids[ids_pt.size(1):])
        jx_text = tok.decode(jx_out)
        
        print(f"PyTorch: {repr(pt_text)}")
        print(f"JAX:     {repr(jx_text)}")
        
        if pt_text == jx_text:
            print("✅ PASS: Greedy generation matches!")
        else:
            print("❌ FAIL: Generation diverged")
            
    else:
        print(f"❌ FAIL: Logits differ by {err}")
        print("Top 5 PyTorch logits:", np.argsort(l_pt)[-5:][::-1])
        print("Top 5 JAX logits:", np.argsort(l_jx)[-5:][::-1])
        
        # Check specific error patterns
        if abs(err - 0.24) < 0.01:
            print("🔍 Likely cause: RoPE theta mismatch (should be 1,000,000 for Qwen2.5)")
        elif 0.01 <= err <= 0.05:
            print("🔍 Likely cause: Causal mask alignment issue")
        elif err > 1.0:
            print("🔍 Likely cause: KV cache shape mismatch or major architectural issue")
            
    # Always test logits comparison regardless of pass/fail
    try:
        np.testing.assert_allclose(l_pt, l_jx, atol=1e-4)
    except AssertionError as e:
        print(f"Assertion failed: {e}")

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--model", required=True)
    main(a.parse_args()) 