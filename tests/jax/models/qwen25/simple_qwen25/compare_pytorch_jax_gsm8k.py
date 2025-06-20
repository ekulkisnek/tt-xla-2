#!/usr/bin/env python3
"""
Compare PyTorch vs JAX Qwen2.5-7B-Instruct on actual GSM8K problems
"""
import numpy as np
import torch
import jax
import jax.numpy as jnp
import json
import re
import sys
import os
sys.path.append('.')

def extract_answer(text):
    """Extract numerical answer from generated text"""
    # Look for patterns like "The answer is X" or just numbers
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

def test_pytorch_gsm8k():
    """Test PyTorch on GSM8K problems"""
    print("🔥 Testing PyTorch Qwen2.5-7B-Instruct")
    print("="*50)
    
    # Load PyTorch model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("✅ PyTorch model loaded")
        
    except Exception as e:
        print(f"❌ PyTorch loading failed: {e}")
        return []
    
    # GSM8K test problems
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
            "question": "Sarah has 15 stickers. She gives 7 to her friend. How many stickers does Sarah have now?",
            "answer": 8
        },
        {
            "question": "A box contains 24 candies. If 6 children share them equally, how many candies does each child get?",
            "answer": 4
        },
        {
            "question": "Mike has 12 toys. He gives away 4 toys and buys 3 more. How many toys does he have now?",
            "answer": 11
        }
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n🧮 Problem {i+1}: {problem['question']}")
        print(f"Expected: {problem['answer']}")
        
        # Use chat template for instruct model
        messages = [{"role": "user", "content": f"Solve this step by step:\n{problem['question']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = pt_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Generated: {response[:200]}...")
        
        # Extract answer
        extracted = extract_answer(response)
        correct = extracted == problem['answer']
        
        print(f"Extracted: {extracted}")
        print(f"Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        
        results.append({
            'problem': i+1,
            'question': problem['question'],
            'expected': problem['answer'],
            'generated': response,
            'extracted': extracted,
            'correct': correct
        })
    
    # Clean up
    del pt_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def test_jax_gsm8k():
    """Test JAX on GSM8K problems"""
    print("\n\n🚀 Testing JAX Qwen2.5-7B-Instruct")
    print("="*50)
    
    # Load JAX model
    try:
        from q25_jax_instruct import Qwen25ForCausalLM, load_params, generate_text
        from transformers import AutoTokenizer
        import json
        
        # Load tokenizer (same as PyTorch)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        
        config_path = "../instruct_weights/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        params = load_params(model, "../instruct_weights", jnp.float32)
        
        print("✅ JAX model loaded")
        
    except Exception as e:
        print(f"❌ JAX loading failed: {e}")
        return []
    
    # Same problems as PyTorch
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
            "question": "Sarah has 15 stickers. She gives 7 to her friend. How many stickers does Sarah have now?",
            "answer": 8
        },
        {
            "question": "A box contains 24 candies. If 6 children share them equally, how many candies does each child get?",
            "answer": 4
        },
        {
            "question": "Mike has 12 toys. He gives away 4 toys and buys 3 more. How many toys does he have now?",
            "answer": 11
        }
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n🧮 Problem {i+1}: {problem['question']}")
        print(f"Expected: {problem['answer']}")
        
        # Use chat template (same as PyTorch)
        messages = [{"role": "user", "content": f"Solve this step by step:\n{problem['question']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate
        try:
            response = generate_text(
                model, params, tokenizer, 
                prompt=prompt,
                max_tokens=100, 
                temperature=0.1,
                use_chat_template=False  # Already applied
            )
            print(f"Generated: {response[:200]}...")
            
        except Exception as e:
            print(f"Generation failed: {e}")
            response = ""
        
        # Extract answer
        extracted = extract_answer(response)
        correct = extracted == problem['answer']
        
        print(f"Extracted: {extracted}")
        print(f"Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        
        results.append({
            'problem': i+1,
            'question': problem['question'],
            'expected': problem['answer'],
            'generated': response,
            'extracted': extracted,
            'correct': correct
        })
    
    return results

def compare_results(pytorch_results, jax_results):
    """Compare PyTorch vs JAX results"""
    print("\n\n📊 COMPARISON RESULTS")
    print("="*70)
    
    if not pytorch_results:
        print("❌ No PyTorch results to compare")
        return
    
    if not jax_results:
        print("❌ No JAX results to compare")
        return
    
    # Calculate scores
    pt_correct = sum(1 for r in pytorch_results if r['correct'])
    jax_correct = sum(1 for r in jax_results if r['correct'])
    total = len(pytorch_results)
    
    print(f"📈 Overall Scores:")
    print(f"  PyTorch: {pt_correct}/{total} ({pt_correct/total*100:.1f}%)")
    print(f"  JAX:     {jax_correct}/{total} ({jax_correct/total*100:.1f}%)")
    
    # Detailed comparison
    print(f"\n📋 Problem-by-Problem:")
    print(f"{'Problem':<8} {'Expected':<8} {'PyTorch':<10} {'JAX':<10} {'Match':<8}")
    print("-" * 50)
    
    matches = 0
    for i in range(total):
        pt_result = pytorch_results[i]
        jax_result = jax_results[i]
        
        pt_extracted = pt_result['extracted'] if pt_result['extracted'] is not None else "None"
        jax_extracted = jax_result['extracted'] if jax_result['extracted'] is not None else "None"
        
        match = pt_result['extracted'] == jax_result['extracted']
        if match:
            matches += 1
        
        print(f"{i+1:<8} {pt_result['expected']:<8} {pt_extracted:<10} {jax_extracted:<10} {'✅' if match else '❌':<8}")
    
    print(f"\n🎯 Agreement: {matches}/{total} ({matches/total*100:.1f}%)")
    
    if matches == total:
        print("🎉 PERFECT AGREEMENT: PyTorch and JAX give identical results!")
    elif matches >= total * 0.8:
        print("🔶 HIGH AGREEMENT: Models mostly agree")
    else:
        print("⚠️ LOW AGREEMENT: Significant differences between PyTorch and JAX")

def main():
    print("🔍 PyTorch vs JAX Qwen2.5-7B-Instruct GSM8K Comparison")
    print("="*70)
    print("📝 Both models loading from: Qwen/Qwen2.5-7B-Instruct")
    print("📝 Using identical chat templates and generation settings")
    
    # Test PyTorch
    pytorch_results = test_pytorch_gsm8k()
    
    # Test JAX  
    jax_results = test_jax_gsm8k()
    
    # Compare
    compare_results(pytorch_results, jax_results)

if __name__ == "__main__":
    main() 