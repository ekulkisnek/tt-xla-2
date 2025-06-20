#!/usr/bin/env python3
"""
Compare base model vs instruct model on same problem
"""
import sys
sys.path.append('../baseline_single')
from q25_jax import Qwen25ForCausalLM as BaseQwen25, load_params as base_load_params
sys.path.append('.')
from q25_jax_instruct import Qwen25ForCausalLM as InstructQwen25, load_params as instruct_load_params
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp
import json
import os

def compare_models():
    """Compare base vs instruct on same problem"""
    
    # Test problem
    prompt = "8 - 3 = "
    
    print("🔄 COMPARING BASE vs INSTRUCT MODEL")
    print(f"Problem: '{prompt}'")
    print("="*60)
    
    # Test Base Model
    print("\n🔹 TESTING BASE MODEL")
    print("-" * 30)
    
    try:
        base_path = "../weights"
        base_config_path = os.path.join(base_path, "config.json") 
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
        
        base_model = BaseQwen25(config=base_config, dtype=jnp.bfloat16)
        base_tokenizer = AutoTokenizer.from_pretrained(base_path)
        base_params = base_load_params(base_model, base_path, jnp.bfloat16)
        
        # Test base model prediction
        inputs = base_tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        outputs = base_model.apply(
            base_params,
            input_ids=input_ids,
            return_dict=True
        )
        
        logits = outputs["logits"]
        last_logits = logits[0, -1, :]
        top_k_logits, top_k_indices = jax.lax.top_k(last_logits, k=5)
        
        print("Base model top 5 predictions:")
        for i, (logit, token_id) in enumerate(zip(top_k_logits, top_k_indices)):
            token = base_tokenizer.decode([int(token_id)])
            print(f"  {i+1}. Token {int(token_id)}: '{token}' (logit: {float(logit):.3f})")
        
        del base_model, base_params  # Free memory
        
    except Exception as e:
        print(f"Base model test failed: {e}")
    
    # Test Instruct Model  
    print("\n🔹 TESTING INSTRUCT MODEL")
    print("-" * 30)
    
    try:
        instruct_path = "../instruct_weights"
        instruct_config_path = os.path.join(instruct_path, "config.json")
        with open(instruct_config_path, 'r') as f:
            instruct_config = json.load(f)
        
        instruct_model = InstructQwen25(config=instruct_config, dtype=jnp.bfloat16)
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_path)
        instruct_params = instruct_load_params(instruct_model, instruct_path, jnp.bfloat16)
        
        # Test instruct model prediction
        inputs = instruct_tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        outputs = instruct_model.apply(
            instruct_params,
            input_ids=input_ids,
            return_dict=True
        )
        
        logits = outputs["logits"]
        last_logits = logits[0, -1, :]
        top_k_logits, top_k_indices = jax.lax.top_k(last_logits, k=5)
        
        print("Instruct model top 5 predictions:")
        for i, (logit, token_id) in enumerate(zip(top_k_logits, top_k_indices)):
            token = instruct_tokenizer.decode([int(token_id)])
            print(f"  {i+1}. Token {int(token_id)}: '{token}' (logit: {float(logit):.3f})")
            
    except Exception as e:
        print(f"Instruct model test failed: {e}")
    
    print("\n" + "="*60)
    print("💡 If models give different predictions, the issue is instruct-specific.")
    print("💡 If they're the same, the issue is in our JAX implementation.")

if __name__ == "__main__":
    compare_models() 