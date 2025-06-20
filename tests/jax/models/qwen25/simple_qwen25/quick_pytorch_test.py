#!/usr/bin/env python3
"""
Quick PyTorch test for comparison with JAX
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_pytorch_simple():
    print("🤖 Testing PyTorch on Simple Math")
    print("="*50)
    
    try:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Loading PyTorch model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        questions = ["What is 2+2?", "What is 5-3?"]
        
        for i, question in enumerate(questions):
            print(f"\n{'='*30}")
            print(f"Question {i+1}: {question}")
            print(f"{'='*30}")
            
            # Use same chat template as JAX
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = pt_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"PyTorch response: '{response}'")
        
        del pt_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pytorch_simple() 