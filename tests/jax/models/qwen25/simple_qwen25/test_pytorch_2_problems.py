#!/usr/bin/env python3
"""
Test PyTorch Qwen2.5-7B-Instruct on 2 GSM8K problems for direct comparison with JAX
"""
import torch
import re
import json
import time

def extract_answer(text):
    """Extract numerical answer from generated text"""
    patterns = [
        r"the answer is (\d+)",
        r"answer is (\d+)",
        r"answer: (\d+)", 
        r"####\s*(\d+)",
        r"therefore[,\s]*(?:the answer is\s*)?(\d+)",
        r"so[,\s]*(?:the answer is\s*)?(\d+)",
        r"result is (\d+)",
        r"equals? (\d+)",
        r"\$(\d+)",
        r"(\d+)\s*(?:dollars?|apples?|stickers?|candies?|books?|toys?|items?|people?|minutes?|hours?|days?)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[-1])
    
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return int(numbers[-1])
    
    return None

def test_pytorch_2_problems():
    print("🔥 Testing PyTorch Qwen2.5-7B-INSTRUCT on 2 GSM8K Problems")
    print("="*70)
    
    # Load PyTorch model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"📥 Loading model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        print("✅ PyTorch INSTRUCT model loaded")
        
    except Exception as e:
        print(f"❌ PyTorch loading failed: {e}")
        return
    
    # Same 2 problems as JAX test
    problems = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": 72
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": 10
        }
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n{'='*70}")
        print(f"🧮 Problem {i+1}/2")
        print(f"{'='*70}")
        print(f"Question: {problem['question']}")
        print(f"Expected Answer: {problem['answer']}")
        
        # Use instruct chat template
        messages = [{"role": "user", "content": f"Solve this math problem step by step:\n\n{problem['question']}\n\nShow your work and clearly state the final answer."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate with more tokens for complete reasoning
        inputs = tokenizer(prompt, return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            outputs = pt_model.generate(
                **inputs,
                max_new_tokens=250,  # Increased tokens
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n🤖 Generated Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print(f"⏱️  Generation time: {generation_time:.1f}s")
        
        # Extract answer
        extracted = extract_answer(response)
        correct = extracted == problem['answer']
        
        print(f"\n📊 Analysis:")
        print(f"  Extracted Answer: {extracted}")
        print(f"  Expected Answer: {problem['answer']}")
        print(f"  Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
        
        results.append({
            'problem': i+1,
            'question': problem['question'],
            'expected': problem['answer'],
            'generated': response,
            'extracted': extracted,
            'correct': correct,
            'time': generation_time
        })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"\n{'='*70}")
    print(f"🏆 PYTORCH FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Score: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"Average generation time: {avg_time:.1f}s per problem")
    print(f"\nProblem-by-problem:")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"  {r['problem']}: {status} Expected {r['expected']}, Got {r['extracted']} ({r['time']:.1f}s)")
    
    # Save results for comparison with JAX
    with open('pytorch_2_problems_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to pytorch_2_problems_results.json")
    
    # Clean up memory
    del pt_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"🧹 Memory cleaned up")
    
    return results

if __name__ == "__main__":
    test_pytorch_2_problems() 