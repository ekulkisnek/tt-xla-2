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

    # 4. Optional PyTorch baseline -------------------------------------------
    torch_model, torch_tok = build_torch_baseline(not args.no_torch)

    # 5. Loop over problems ---------------------------------------------------
    correct = both_match = 0
    for idx in indices:
        ex = test[idx]
        # Use a simpler prompt format that works better with the model
        prompt = f"{ex['question']}\n\nSolution:"
        # JAX forward ---------------------------------------------------------
        input_ids = tok(prompt, return_tensors="np")["input_ids"]
        collected = []
        pkv = None  # Start with None, model will handle initialization
        
        for step in range(args.max_steps):
            try:
                next_id, pkv = greedy(params, input_ids, pkv)
                next_id = int(next_id[0])
                collected.append(next_id)
                input_ids = np.array([[next_id]], dtype=np.int32)
                
                # Optional debug output
                # if step < 5:
                #     token_text = tok.decode([next_id])
                #     print(f"Step {step}: token_id={next_id}, text={repr(token_text)}")
                
                if tok.decode([next_id]).endswith("####") or next_id == tok.eos_token_id:
                    print(f"Stopping at step {step}: EOS or #### found")
                    break
            except Exception as e:
                print(f"Error during generation: {e}")
                break
        jax_txt = tok.decode(collected)
        jax_ans = extract_num(jax_txt)
        gold    = ex["answer"].split("####")[1].strip()
        jax_ok  = (jax_ans == gold)
        if jax_ok: correct += 1
        
        # Debug output
        print(f"Generated text: {repr(jax_txt[:200])}...")
        print(f"Question: {ex['question'][:100]}...")

        # PyTorch parity ------------------------------------------------------
        if torch_model:
            t_txt = torch_answer(torch_model, torch_tok, prompt, args.max_steps)
            t_ans = extract_num(t_txt)
            both_match += int(jax_ans == t_ans)

        # Report single line --------------------------------------------------
        flag = "✅" if jax_ok else "❌"
        print(f"{flag}  idx={idx:<4}  gold={gold:<6}  jax={jax_ans}  "
              f"{'(torch '+t_ans+')' if torch_model else ''}")

    # 6. Summary --------------------------------------------------------------
    n = len(indices)
    print(f"\nJAX accuracy {correct}/{n}  ({100*correct/n:.1f} %)")
    if torch_model:
        print(f"JAX == PyTorch on {both_match}/{n}")

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
    p.add_argument("--max_steps", type=int, default=256,
                   help="Decoder budget per problem before giving up")
    p.add_argument("--no_torch", action="store_true",
                   help="Skip PyTorch baseline (saves RAM/time)")
    main(p.parse_args()) 