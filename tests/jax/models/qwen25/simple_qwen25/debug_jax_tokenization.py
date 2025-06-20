#!/usr/bin/env python3
"""
Debug JAX tokenization and input processing
"""
import sys
sys.path.append('.')
from transformers import AutoTokenizer
import numpy as np

def debug_tokenization():
    print("🔍 Debugging JAX Tokenization Issues")
    print("="*50)
    
    model_path = "../instruct_weights"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test inputs that were failing
    test_prompts = [
        "Solve this math problem step by step:\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nShow your work and clearly state the final answer.",
        "Hello",
        "2+2="
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*50}")
        print(f"🧮 Test {i+1}: {prompt[:50]}...")
        print(f"{'='*50}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        print(f"📊 Tokenization Results:")
        print(f"  Input length: {len(prompt)} chars")
        print(f"  Token count: {input_ids.shape[1]}")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Token IDs (first 10): {input_ids[0][:10].tolist()}")
        print(f"  Token IDs (last 10): {input_ids[0][-10:].tolist()}")
        
        # Decode back to check
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"  Decoded back: {repr(decoded[:100])}...")
        
        # Check for round-trip accuracy
        matches = decoded.strip() == prompt.strip()
        print(f"  Round-trip match: {'✅' if matches else '❌'}")
        
        if not matches:
            print(f"  🚨 MISMATCH DETECTED!")
            print(f"    Original: {repr(prompt[:100])}")
            print(f"    Decoded:  {repr(decoded[:100])}")
        
        # Check special tokens
        special_tokens = [
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id
        ]
        print(f"  Special tokens: EOS={special_tokens[0]}, BOS={special_tokens[1]}, PAD={special_tokens[2]}, UNK={special_tokens[3]}")
        
        # Check if any special tokens appear in input
        contains_special = any(token_id in special_tokens for token_id in input_ids[0])
        print(f"  Contains special tokens: {'⚠️  YES' if contains_special else '✅ NO'}")
        
        # Vocabulary size check
        max_token = int(input_ids.max())
        vocab_size = tokenizer.vocab_size
        print(f"  Max token ID: {max_token}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Valid range: {'✅' if max_token < vocab_size else '❌ OUT OF RANGE!'}")

if __name__ == "__main__":
    debug_tokenization() 