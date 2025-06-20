#!/usr/bin/env python3
"""
PyTorch reference test to compare with JAX implementation
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def test_pytorch_reference():
    """Test the official PyTorch model"""
    
    # Note: This requires significant memory and GPU
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print("🔹 Testing PyTorch Reference Model")
    print(f"Model: {model_name}")
    print("="*60)
    
    try:
        # Load model and tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("Loading model... (this may take a while)")
        # Use CPU since we don't have GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu",  # Change to "auto" if you have GPU
            trust_remote_code=True
        )
        
        # Test the same problem
        prompt = "8 - 3 = "
        print(f"\n🧮 Testing: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"Input tokens: {input_ids}")
        
        # Get logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get top predictions
        last_logits = logits[0, -1, :]
        top_k_values, top_k_indices = torch.topk(last_logits, k=5)
        
        print("\nPyTorch reference top 5 predictions:")
        for i, (logit, token_id) in enumerate(zip(top_k_values, top_k_indices)):
            token = tokenizer.decode([int(token_id)])
            print(f"  {i+1}. Token {int(token_id)}: '{token}' (logit: {float(logit):.3f})")
        
        # Also test generation
        print("\n🎯 Testing generation:")
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,  # Greedy
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: '{generated_text}'")
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        print("This might be due to memory constraints or missing GPU.")
        print("The model requires ~14GB RAM minimum.")

if __name__ == "__main__":
    test_pytorch_reference() 