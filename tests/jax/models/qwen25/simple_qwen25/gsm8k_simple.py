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
    """High-quality greedy generation - uses same logic as simple_inference.py but greedy"""
    try:
        # Tokenize input (same as simple_inference.py)
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        # Create attention mask - use 4D format for Qwen model (CRITICAL!)
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        attention_mask = np.ones((batch_size, 1, 1, seq_length), dtype=np.int32)
        
        # Position IDs - make sure to match the input_ids length
        position_ids = np.arange(input_ids.shape[1], dtype=np.int32)[None, :]
        
        # Initialize generation state (same as simple_inference.py)
        state = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": None,
        }
        
        # Track generated text
        generated_text = ""
        
        # Generate tokens (same logic as simple_inference.py)
        for _ in range(max_tokens):
            # Forward pass
            outputs = model.apply(
                params,
                input_ids=state["input_ids"],
                attention_mask=state["attention_mask"],
                position_ids=state["position_ids"],
                past_key_values=state["past_key_values"],
                return_dict=True
            )
            
            # Get logits and past key values
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            
            # Greedy selection (temperature=0 equivalent)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)
            
            # Update state (same as simple_inference.py)
            state["input_ids"] = next_token[:, None]
            state["attention_mask"] = np.ones((batch_size, 1, 1, 1), dtype=np.int32)
            state["position_ids"] = np.array([[state["position_ids"][0, -1] + 1]], dtype=np.int32)
            state["past_key_values"] = past_key_values
            
            # Decode token
            token = tokenizer.decode(next_token[0])
            generated_text += token
            
            # Check for early stopping
            if next_token[0] == tokenizer.eos_token_id:
                break
                
            # Check if we found the answer marker - stop after #### appears
            if "####" in generated_text:
                break
        
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