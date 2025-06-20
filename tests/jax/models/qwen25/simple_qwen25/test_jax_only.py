#!/usr/bin/env python3
"""
Test JAX Qwen2.5-7B-Instruct on GSM8K problems (separate from PyTorch)
"""
import jax.numpy as jnp
import numpy as np
import re
import json
import sys
import os
sys.path.append('.')

def extract_answer(text):
    """Extract numerical answer from generated text"""
    patterns = [
        r"(?:the answer is|answer:|=)\s*(\d+)",
        r"####\s*(\d+)",
        r"\$(\d+)",
        r"(\d+)\s*(?:dollars?|apples?|stickers?|candies?|books?|toys?|items?)",
        r"(?:^|\s)(\d+)(?:\s|$|\.)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return None

def test_jax_instruct():
    print("🚀 Testing JAX Qwen2.5-7B-INSTRUCT")
    print("="*60)
    
    # Load JAX model
    try:
        from q25_jax_instruct import Qwen25ForCausalLM, load_params, generate_text
        from transformers import AutoTokenizer
        import json
        
        print("📥 Loading JAX model from instruct_weights...")
        print("📝 Confirming this is the INSTRUCT model...")
        
        # Load tokenizer (instruct version)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        
        config_path = "../instruct_weights/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"📋 Config vocab_size: {config['vocab_size']}")
        print(f"📋 Config model_type: {config['model_type']}")
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        params = load_params(model, "../instruct_weights", jnp.float32)
        
        print("✅ JAX INSTRUCT model loaded")
        
    except Exception as e:
        print(f"❌ JAX loading failed: {e}")
        return
    
    # Same test problems as PyTorch
    problems = [
        {
            "question": "Janet has 5 apples. She eats 2 apples. How many apples does she have left?",
            "answer": 3
        },
        {
            "question": "Tom bought 8 books for $3 each. How much did he spend in total?",
            "answer": 24
        },
        {
            "question": "What is 2 + 2?",
            "answer": 4
        },
        {
            "question": "What is 8 - 3?",
            "answer": 5
        },
        {
            "question": "Sarah has 15 stickers. She gives 7 to her friend. How many stickers does Sarah have now?",
            "answer": 8
        }
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n🧮 Problem {i+1}/5")
        print(f"Question: {problem['question']}")
        print(f"Expected: {problem['answer']}")
        
        # Use instruct chat template (same as PyTorch)
        messages = [{"role": "user", "content": f"Solve this step by step:\n{problem['question']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Prompt preview: {prompt[:100]}...")
        
        # Generate with same settings as PyTorch
        try:
            response = generate_text(
                model, params, tokenizer, 
                prompt=prompt,
                max_tokens=50,  # Same as PyTorch
                temperature=0.1,
                use_chat_template=False  # Already applied
            )
            print(f"Generated: {response}")
            
        except Exception as e:
            print(f"Generation failed: {e}")
            response = ""
        
        # Extract answer
        extracted = extract_answer(response)
        correct = extracted == problem['answer']
        
        print(f"Extracted answer: {extracted}")
        print(f"Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        
        results.append({
            'problem': i+1,
            'question': problem['question'],
            'expected': problem['answer'],
            'generated': response,
            'extracted': extracted,
            'correct': correct
        })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print(f"🏆 JAX INSTRUCT RESULTS: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"{'='*60}")
    
    # Save results for comparison
    with open('jax_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("💾 Results saved to jax_results.json")
    
    # Compare with PyTorch if available
    try:
        with open('pytorch_results.json', 'r') as f:
            pytorch_results = json.load(f)
        
        print(f"\n📊 QUICK COMPARISON")
        print(f"{'='*40}")
        print(f"PyTorch: {sum(1 for r in pytorch_results if r['correct'])}/{len(pytorch_results)}")
        print(f"JAX:     {correct_count}/{total_count}")
        
        # Check if answers match
        matches = 0
        for i in range(min(len(pytorch_results), len(results))):
            if pytorch_results[i]['extracted'] == results[i]['extracted']:
                matches += 1
        
        print(f"Agreement: {matches}/{min(len(pytorch_results), len(results))} problems")
        
    except FileNotFoundError:
        print("📝 Run test_pytorch_only.py first for comparison")

if __name__ == "__main__":
    test_jax_instruct() 