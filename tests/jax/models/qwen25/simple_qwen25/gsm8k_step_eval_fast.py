#!/usr/bin/env python3
"""
FAST GSM8K evaluator for JAX Qwen2.5-7B.
Uses optimized single-step generation with better caching.
"""
import os, re, json, argparse, time, numpy as np, datasets, jax, jax.numpy as jnp
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params

# ---------- helper functions ----------
def extract_num(txt: str):
    m = re.search(r"####\s*(-?\d+)", txt)
    return m.group(1) if m else None

# ---------- optimized generation ----------
def build_optimized_generator(model, params):
    """Build a more efficient generator that minimizes recompilation"""
    
    @jax.jit
    def forward_pass(input_ids, past_key_values):
        """Single forward pass - this gets compiled once"""
        outputs = model.apply(params, input_ids=input_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        next_pkv = outputs["past_key_values"] 
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        return next_token, next_pkv, logits[:, -1, :]
    
    return forward_pass

def main(args):
    # Dataset
    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        start = 0 if args.offset is None else args.offset
        indices = list(range(start, start + args.count))
    print(f"🔹 Fast evaluating indices: {indices}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.jax_path)

    # Model loading
    print("🔹 Loading JAX model...")
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.jax_path, "config.json")))
    model = Qwen25ForCausalLM(config=cfg, dtype=jnp.bfloat16)
    params = load_params(model, args.jax_path, jnp.bfloat16)
    print(f"    Model loaded in {time.time()-t0:.1f}s")

    # Build optimized generator
    print("🔹 Compiling generation function...")
    forward_pass = build_optimized_generator(model, params)
    
    # Warm up compilation
    dummy_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    try:
        _, _, _ = forward_pass(dummy_ids, None)
        print("    Generation function compiled ✓")
    except Exception as e:
        print(f"    Compilation failed: {e}")
        return

    # Evaluation loop
    correct = 0
    for idx in indices:
        ex = test[idx]
        print(f"\n📝 Problem {idx}: {ex['question'][:80]}...")
        
        # Prepare prompt  
        prompt = f"{ex['question']}\n\nSolution:"
        input_ids = tok(prompt, return_tensors="np")["input_ids"]
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        
        # Fast generation
        collected_tokens = []
        pkv = None
        current_ids = input_ids
        
        total_start = time.time()
        for step in range(args.max_tokens):
            step_start = time.time()
            
            try:
                next_token, pkv, logits = forward_pass(current_ids, pkv)
                next_token_id = int(next_token[0])
                collected_tokens.append(next_token_id)
                
                step_time = time.time() - step_start
                
                # Progress indicator every 10 tokens
                if step % 10 == 0:
                    speed = (step + 1) / (time.time() - total_start)
                    print(f"  Step {step}: {speed:.1f} tok/s")
                
                # Check stopping conditions
                if step_time > 3.0:  # If any step takes > 3s, something's wrong
                    print(f"⚠️  Step {step} took {step_time:.1f}s, stopping")
                    break
                    
                token_text = tok.decode([next_token_id])
                if "####" in token_text or next_token_id == tok.eos_token_id:
                    print(f"✓ Found answer marker at step {step}")
                    break
                
                # Update input for next iteration (just the new token)
                current_ids = jnp.array([[next_token_id]], dtype=jnp.int32)
                
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                break
        
        total_time = time.time() - total_start
        
        # Process results
        if collected_tokens:
            jax_txt = tok.decode(collected_tokens)
            jax_ans = extract_num(jax_txt)
            
            print(f"⏱️  {len(collected_tokens)} tokens in {total_time:.1f}s ({len(collected_tokens)/total_time:.1f} tok/s)")
            print(f"📄 {jax_txt[:150]}...")
            if "####" in jax_txt:
                answer_part = jax_txt.split("####")[-1][:30].strip()
                print(f"🎯 Answer: {answer_part}")
        else:
            jax_ans = None
            jax_txt = ""
            
        gold = ex["answer"].split("####")[1].strip()
        jax_ok = (jax_ans == gold)
        if jax_ok: correct += 1

        # Report
        flag = "✅" if jax_ok else "❌"
        print(f"{flag} idx={idx:<4} gold={gold:<6} jax={jax_ans}")

    # Summary
    n = len(indices)
    print(f"\n🏆 FINAL RESULTS:")
    print(f"JAX accuracy: {correct}/{n} ({100*correct/n:.1f}%)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jax_path", required=True, help="Path to model weights")
    p.add_argument("--count", type=int, default=1, help="Number of problems")
    p.add_argument("--offset", type=int, default=None, help="Start index")
    p.add_argument("--indices", type=str, default="", help="Specific indices")
    p.add_argument("--max_tokens", type=int, default=80, help="Max tokens per problem")
    main(p.parse_args()) 