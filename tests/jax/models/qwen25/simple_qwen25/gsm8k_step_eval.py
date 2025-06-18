#!/usr/bin/env python3
"""
Low-overhead GSM8K evaluator for JAX Qwen2.5-7B.
• Starts with exactly one example unless --count or --indices is given.
• Optional PyTorch parity check (disable with --no_torch).
"""
import os, re, json, argparse, random, time, functools, numpy as np, datasets, jax, jax.numpy as jnp
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params  # your existing code

# ---------- tiny helper ----------
def extract_num(txt: str):
    m = re.search(r"####\s*(-?\d+)", txt)
    return m.group(1) if m else None

# ---------- JAX greedy generator ----------
def build_greedy_fn(model):
    @jax.jit
    def _step(params, input_ids, pkv):
        """Single greedy decode step."""
        outputs = model.apply(params, input_ids=input_ids, past_key_values=pkv, return_dict=True)
        logits = outputs["logits"]
        next_pkv = outputs["past_key_values"]
        next_id = jnp.argmax(logits[:, -1, :], axis=-1)
        return next_id, next_pkv
    return _step

# ---------- optional PyTorch baseline ----------
def build_torch_baseline(load: bool):
    if not load:
        return None, None
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    return model, tok

def torch_answer(torch_model, tok, prompt, max_new=256):
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids
    out = torch_model.generate(ids, max_new_tokens=max_new, temperature=0.0, top_k=1)[0][ids.shape[1]:]
    return tok.decode(out, skip_special_tokens=True)

# ---------- main ----------
def main(args):
    # 1. Dataset sample list --------------------------------------------------
    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        start = 0 if args.offset is None else args.offset
        indices = list(range(start, start + args.count))
    print(f"🔹 Evaluating indices: {indices}")

    # 2. Tokeniser (shared) ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.jax_path)

    # 3. JAX model + params ---------------------------------------------------
    print("🔹 Loading JAX model …")
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.jax_path, "config.json")))
    model = Qwen25ForCausalLM(config=cfg, dtype=jnp.bfloat16)
    params = load_params(model, args.jax_path, jnp.bfloat16)
    greedy = build_greedy_fn(model)
    print(f"    done in {time.time()-t0:.1f}s")

    # Warm up JIT compilation with a dummy call
    print("🔹 Warming up JAX compilation...")
    dummy_ids = np.array([[1, 2, 3]], dtype=np.int32)
    try:
        _, _ = greedy(params, dummy_ids, None)
        print("    JIT compilation completed")
    except Exception as e:
        print(f"    Warmup failed: {e}")

    # 4. Optional PyTorch baseline -------------------------------------------
    torch_model, torch_tok = build_torch_baseline(not args.no_torch)

    # 5. Loop over problems ---------------------------------------------------
    correct = both_match = 0
    for idx in indices:
        ex = test[idx]
        
        print(f"\n📝 Problem {idx}: {ex['question'][:80]}...")
        
        # Use a simpler prompt format that works better with the model
        prompt = f"{ex['question']}\n\nSolution:"
        # JAX forward ---------------------------------------------------------
        input_ids = tok(prompt, return_tensors="np")["input_ids"]
        collected = []
        pkv = None  # Start with None, model will handle initialization
        
        gen_start = time.time()
        timeout_per_step = 2.0  # Max 2 seconds per token
        
        for step in range(args.max_steps):
            step_start = time.time()
            try:
                next_id, pkv = greedy(params, input_ids, pkv)
                next_id = int(next_id[0])
                collected.append(next_id)
                input_ids = np.array([[next_id]], dtype=np.int32)
                
                step_time = time.time() - step_start
                if step_time > timeout_per_step:
                    print(f"⚠️ Step {step} took {step_time:.1f}s (timeout)")
                
                # Show progress every 20 tokens
                if step % 20 == 0 and step > 0:
                    partial_text = tok.decode(collected[-20:])
                    print(f"    Step {step}: ...{partial_text}")
                
                # Check for stopping conditions
                token_text = tok.decode([next_id])
                if token_text.endswith("####") or next_id == tok.eos_token_id:
                    print(f"✓ Stopped at step {step}: found #### or EOS")
                    break
                    
                # Emergency brake: if generation is taking too long
                total_time = time.time() - gen_start
                if total_time > 60:  # Max 1 minute per problem
                    print(f"⚠️ Generation timeout after {total_time:.1f}s")
                    break
                    
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                break
        
        gen_time = time.time() - gen_start
        print(f"⏱️ Generated {len(collected)} tokens in {gen_time:.1f}s ({len(collected)/gen_time:.1f} tok/s)")
        
        jax_txt = tok.decode(collected)
        jax_ans = extract_num(jax_txt)
        gold    = ex["answer"].split("####")[1].strip()
        jax_ok  = (jax_ans == gold)
        if jax_ok: correct += 1

        # Show a sample of the generated text
        print(f"📄 Generated: {jax_txt[:150]}...")
        if "####" in jax_txt:
            answer_part = jax_txt.split("####")[-1][:50]
            print(f"🎯 Found answer section: {answer_part}")

        # PyTorch parity ------------------------------------------------------
        if torch_model:
            print("🔄 Running PyTorch baseline...")
            torch_start = time.time()
            t_txt = torch_answer(torch_model, torch_tok, prompt, args.max_steps)
            t_ans = extract_num(t_txt)
            both_match += int(jax_ans == t_ans)
            torch_time = time.time() - torch_start
            print(f"⏱️ PyTorch took {torch_time:.1f}s")

        # Report single line --------------------------------------------------
        flag = "✅" if jax_ok else "❌"
        print(f"{flag} idx={idx:<4} gold={gold:<6} jax={jax_ans} "
              f"{'(torch='+str(t_ans)+')' if torch_model else ''}")

    # 6. Summary --------------------------------------------------------------
    n = len(indices)
    print(f"\n🏆 FINAL RESULTS:")
    print(f"JAX accuracy: {correct}/{n} ({100*correct/n:.1f}%)")
    if torch_model:
        print(f"JAX==PyTorch: {both_match}/{n}")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jax_path", required=True,
                   help="Folder with Qwen2.5-7B safetensors + config.json")
    p.add_argument("--count", type=int, default=1,
                   help="How many consecutive problems to run (ignored if --indices given)")
    p.add_argument("--offset", type=int, default=None,
                   help="Start index if using --count (default 0)")
    p.add_argument("--indices", type=str, default="",
                   help="Comma-separated list of explicit indices (overrides --count/--offset)")
    p.add_argument("--max_steps", type=int, default=64,
                   help="Decoder budget per problem before giving up (reduced default)")
    p.add_argument("--no_torch", action="store_true",
                   help="Skip PyTorch baseline (saves RAM/time)")
    main(p.parse_args()) 