#!/usr/bin/env python3
"""
Test PyTorch Qwen2.5-7B-Instruct on REAL GSM8K problems with full generation
"""
import torch
import re
import json

def extract_answer(text):
    """Extract numerical answer from generated text"""
    # More comprehensive patterns for GSM8K
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
    
    # Try patterns in order of specificity
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[-1])  # Take the last match (usually the final answer)
    
    # Fallback: extract any number near the end
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return int(numbers[-1])
    
    return None

def test_pytorch_real_gsm8k():
    print("🔥 Testing PyTorch Qwen2.5-7B-INSTRUCT on Real GSM8K")
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
    
    # Real GSM8K problems (from the actual dataset)
    problems = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": 72
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": 10
        },
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose. How much more money does Betty need to buy the wallet?",
            "answer": 35
        },
        {
            "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
            "answer": 42
        },
        {
            "question": "James writes a 3-page letter to 2 different friends. He then writes a 5-page letter to 2 other friends. How many pages did he write in total?",
            "answer": 16
        },
        {
            "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more purple flowers than yellow. Twenty-five percent of flowers are neither yellow nor purple. How many flowers does Mark have in his garden in total?",
            "answer": 35
        },
        {
            "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many slices does he eat that day?",
            "answer": 48
        }
    ]
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n{'='*70}")
        print(f"🧮 Problem {i+1}/7")
        print(f"{'='*70}")
        print(f"Question: {problem['question']}")
        print(f"Expected Answer: {problem['answer']}")
        
        # Use instruct chat template
        messages = [{"role": "user", "content": f"Solve this math problem step by step:\n\n{problem['question']}\n\nShow your work and clearly state the final answer."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate with more tokens for complete reasoning
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = pt_model.generate(
                **inputs,
                max_new_tokens=200,  # More tokens for full reasoning
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n🤖 Generated Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
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
            'correct': correct
        })
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"\n{'='*70}")
    print(f"🏆 PYTORCH FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Score: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"\nProblem-by-problem:")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"  {r['problem']}: {status} Expected {r['expected']}, Got {r['extracted']}")
    
    # Save results for comparison with JAX
    with open('pytorch_real_gsm8k_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to pytorch_real_gsm8k_results.json")
    
    # Clean up memory
    del pt_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"🧹 Memory cleaned up")

if __name__ == "__main__":
    test_pytorch_real_gsm8k() 