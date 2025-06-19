#!/usr/bin/env python3
"""
Simple GSM8K evaluator using the working validation approach from simple_inference.py
"""
import os, re, json, argparse, time, numpy as np, datasets, jax, jax.numpy as jnp
from transformers import AutoTokenizer
from simple_inference import Qwen25ForCausalLM, load_params

def extract_num(txt: str):
    m = re.search(r"####\s*(-?\d+)", txt)
    return m.group(1) if m else None

def simple_generate(model, params, tokenizer, prompt, max_tokens=256):
    """Simple greedy generation - exact copy of validation_level_3 with longer max_tokens"""
    try:
        # Simple greedy generation - match validation_level_3 exactly
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        generated_tokens = []
        current_ids = input_ids
        past_key_values = None
        
        for step in range(max_tokens):
            # Create position_ids for longer generation (this is what validation_level_3 is missing)
            if step == 0:
                # First step: positions start from 0
                position_ids = np.arange(current_ids.shape[1], dtype=np.int32)[None, :]
            else:
                # Subsequent steps: increment position
                position_ids = np.array([[position_ids[0, -1] + 1]], dtype=np.int32)
            
            # Forward pass with position_ids for RoPE
            outputs = model.apply(
                params,
                input_ids=current_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_dict=True
            )
            
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            
            # Greedy selection
            next_token = jnp.argmax(logits[0, -1, :])
            generated_tokens.append(int(next_token))
            
            # Update for next iteration
            current_ids = jnp.array([[int(next_token)]], dtype=jnp.int32)
            
            # Check for early stopping
            if int(next_token) == tokenizer.eos_token_id:
                break
                
            # Check if we found the answer marker - stop after #### appears
            if "####" in tokenizer.decode(generated_tokens):
                break
        
        # Decode result
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text
        
    except Exception as e:
        print(f"Generation error: {e}")
        return ""

def main(args):
    # Dataset
    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        start = 0 if args.offset is None else args.offset
        indices = list(range(start, start + args.count))
    print(f"🔹 Evaluating indices: {indices}")

    # Load model (same as simple_inference.py)
    print("🔹 Loading model...")
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.jax_path, "config.json")))
    model = Qwen25ForCausalLM(config=cfg, dtype=jnp.bfloat16)
    params = load_params(model, args.jax_path, jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.jax_path)
    print(f"    Model loaded in {time.time()-t0:.1f}s")

    # Test that generation works
    print("🔹 Testing generation...")
    test_result = simple_generate(model, params, tokenizer, "What is 2+2?", max_tokens=10)
    print(f"    Test output: {test_result}")

    # Evaluation loop
    correct = 0
    for idx in indices:
        ex = test[idx]
        print(f"\n📝 Problem {idx}: {ex['question'][:60]}...")
        
        # Correct GSM8K prompt template
        prompt = f"Q: {ex['question']}\n\nA: Let's think step by step.\n"
        
        # Generate
        start_time = time.time()
        generated_text = simple_generate(model, params, tokenizer, prompt, max_tokens=args.max_tokens)
        gen_time = time.time() - start_time
        
        # Extract answer
        jax_ans = extract_num(generated_text)
        gold = ex["answer"].split("####")[1].strip()
        jax_ok = (jax_ans == gold)
        if jax_ok: correct += 1
        
        # Report
        flag = "✅" if jax_ok else "❌"
        print(f"⏱️  Generated in {gen_time:.1f}s")
        print(f"📄 {generated_text[:100]}...")
        if "####" in generated_text:
            answer_part = generated_text.split("####")[-1][:20].strip()
            print(f"🎯 Answer: {answer_part}")
        print(f"{flag} idx={idx} gold={gold} jax={jax_ans}")

    # Summary
    n = len(indices)
    print(f"\n🏆 RESULTS: {correct}/{n} ({100*correct/n:.1f}%)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jax_path", required=True)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--offset", type=int, default=None)
    p.add_argument("--indices", type=str, default="")
    p.add_argument("--max_tokens", type=int, default=256)
    main(p.parse_args()) 