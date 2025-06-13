#!/usr/bin/env python3
"""
Qwen2.5-7b Full Model Inference Script
This script attempts to load the full 7B model with memory optimizations.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc
import psutil
import os

def check_memory():
    """Check available system memory"""
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent}%)")
    return memory.available / (1024**3)  # Return available GB

def load_model_with_optimizations():
    """Load the Qwen2.5-7b model with aggressive memory optimizations"""
    print("Checking system memory...")
    available_gb = check_memory()
    
    if available_gb < 8:
        print(f"Warning: Only {available_gb:.1f} GB RAM available. 7B model typically needs 14+ GB.")
        print("Falling back to smaller model...")
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    else:
        print("Attempting to load Qwen2.5-7B-Instruct...")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    try:
        print(f"Loading {model_name}...")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Aggressive memory optimization for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # Additional memory optimizations
            offload_folder="./offload",  # Offload to disk if needed
            offload_state_dict=True,
        )
        
        print(f"Successfully loaded {model_name}")
        print("Model loaded with memory optimizations")
        
        # Check memory after loading
        print("\nMemory usage after loading:")
        check_memory()
        
        return model, tokenizer, model_name
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Trying fallback model...")
        
        # Fallback to smaller model
        try:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"Successfully loaded fallback model: {model_name}")
            return model, tokenizer, model_name
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            return None, None, None

def generate_response(model, tokenizer, question, max_new_tokens=256):
    """Generate a response with memory monitoring"""
    
    print(f"\nMemory before generation:")
    check_memory()
    
    # Prepare the conversation
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful AI assistant. Provide detailed and thoughtful answers."},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt")
    input_length = model_inputs.input_ids.shape[1]
    
    print(f"Input length: {input_length} tokens")
    print("Generating response...")
    
    start_time = time.time()
    
    # Generate with memory optimization
    with torch.no_grad():
        # Use smaller batch size and shorter sequences for memory efficiency
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for efficiency
        )
    
    generation_time = time.time() - start_time
    
    # Extract generated text
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    tokens_generated = len(generated_ids[0])
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Generated {tokens_generated} tokens ({tokens_per_second:.1f} tokens/sec)")
    
    # Clean up
    del model_inputs, generated_ids
    gc.collect()
    
    return response

def main():
    """Main inference function"""
    
    print("="*80)
    print("QWEN2.5-7B MODEL INFERENCE")
    print("="*80)
    
    # Load model
    model, tokenizer, model_name = load_model_with_optimizations()
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    print(f"\nUsing model: {model_name}")
    
    # Test with one complex question first
    question = """Explain the relationship between artificial intelligence and consciousness. 
    What are the main philosophical arguments about whether AI can achieve true consciousness, 
    and what would be the implications if it could?"""
    
    print("\n" + "="*60)
    print("COMPLEX QUESTION:")
    print(question)
    print("\n" + "="*60)
    print("RESPONSE:")
    
    try:
        response = generate_response(model, tokenizer, question, max_new_tokens=512)
        print(response)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print("This might be due to insufficient memory for the 7B model.")
    
    finally:
        # Clean up
        del model, tokenizer
        gc.collect()
        
        print(f"\nFinal memory state:")
        check_memory()
    
    print("\nInference completed!")

if __name__ == "__main__":
    main() 