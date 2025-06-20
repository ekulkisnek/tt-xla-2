#!/usr/bin/env python3
"""
Test PyTorch Qwen2.5-7B-Instruct on GSM8K problems (separate from JAX)
"""
import torch
import re
import json

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

def test_pytorch_instruct():
    print("🔥 Testing PyTorch Qwen2.5-7B-INSTRUCT")
    print("="*60)
    
    # Load PyTorch model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"📥 Loading model: {model_id}")
        print("📝 Confirming this is the INSTRUCT model...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        # Verify it's the instruct model by checking config
        print(f"📋 Model name: {pt_model.config.name_or_path if hasattr(pt_model.config, 'name_or_path') else 'N/A'}")
        print(f"📋 Model type: {pt_model.config.model_type}")
        print("✅ PyTorch INSTRUCT model loaded")
        
    except Exception as e:
        print(f"❌ PyTorch loading failed: {e}")
        return
    
    # Test problems
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
        
        # Use instruct chat template
        messages = [{"role": "user", "content": f"Solve this step by step:\n{problem['question']}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Prompt preview: {prompt[:100]}...")
        
        # Generate with conservative settings
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = pt_model.generate(
                **inputs,
                max_new_tokens=50,  # Shorter to avoid memory issues
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Generated: {response}")
        
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
    print(f"🏆 PYTORCH INSTRUCT RESULTS: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"{'='*60}")
    
    # Save results for comparison
    with open('pytorch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("💾 Results saved to pytorch_results.json")

if __name__ == "__main__":
    test_pytorch_instruct() 