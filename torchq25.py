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
    
    # Use local weights from weights/ folder
    model_path = "weights/Qwen2.5-7B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return None, None, None
    
    print(f"Loading model from local path: {model_path}")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # Set generation config for deterministic output
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=500,  # Increased to 500 tokens
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        print(f"Successfully loaded model from {model_path}")
        print("Model loaded with memory optimizations")
        
        # Check memory after loading
        print("\nMemory usage after loading:")
        check_memory()
        
        return model, tokenizer, model_path
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None, None

def generate_response(model, tokenizer, question, max_new_tokens=500):
    """Generate a response with memory monitoring and real-time token display"""
    
    print(f"\nMemory before generation:")
    check_memory()
    
    # Prepare the conversation with same system prompt as JAX version
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt", padding=True)
    input_length = model_inputs.input_ids.shape[1]
    
    print(f"Input length: {input_length} tokens")
    print("Starting generation with real-time token display...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Manual generation loop for real-time display
    input_ids = model_inputs.input_ids.clone()
    generated_tokens = []
    current_text = ""
    
    print("Generating tokens:")
    for i in range(max_new_tokens):
        # Generate next token
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
        
        # Add to input_ids for next iteration
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        generated_tokens.append(next_token.item())
        
        # Decode the new token and add to current text
        new_token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        current_text += new_token_text
        
        # Show the numbered token
        print(f"{i+1}: {new_token_text}")
        
        # Check for end of sequence
        if next_token.item() == tokenizer.eos_token_id:
            print(f"\n[EOS token reached]")
            break
        
        # Check for end of response markers
        if "<|im_end|>" in new_token_text or "<|endoftext|>" in new_token_text:
            print(f"\n[End marker reached]")
            break
    
    generation_time = time.time() - start_time
    
    # Decode the full generated sequence
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    tokens_generated = len(generated_tokens)
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Generated {tokens_generated} tokens ({tokens_per_second:.1f} tokens/sec)")
    print(f"Final response length: {len(response)} characters")
    
    print(f"\n" + "=" * 60)
    print("COMPLETE RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    # Clean up
    del model_inputs, input_ids, generated_tokens
    gc.collect()
    
    return response

def main():
    """Main inference function"""
    
    print("="*80)
    print("QWEN2.5-7B MODEL INFERENCE - GSM8K MATH QUESTIONS")
    print("="*80)
    
    # Load model
    model, tokenizer, model_name = load_model_with_optimizations()
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    print(f"\nUsing model: {model_name}")
    
    # GSM8K-style math questions
    math_questions = [
        "Question: Sam scores 80 on the first test and 90 on the second. What score does he need on the third test to have an average of 85?",
        
        "Question: A bus has 40 passengers. At the first stop, 1/5 get off and 8 get on. How many passengers are there now?",
        
        "Question: There are 4 apples and 5 oranges in a bowl. John adds 3 more apples and twice as many oranges as the original number of apples. How many fruits are there now?",
        
        "Question: Peter has $100. He buys a shirt for $25, pants for $35, and then finds $10. How much does he have left?",
        
        "Question: A recipe requires 2 cups of flour for 12 cookies. How many cups are needed for 30 cookies?",
        
        "Question: In a class of 30 students, 40% are girls. How many boys are there?",
        
        "Question: A ticket costs $8. There are 5 friends going to the movie. They have a coupon for $10 off the total. How much do they pay in total?",
        
        "Question: A craft store makes a third of its sales in the fabric section, a quarter of its sales in the jewelry section, and the rest in the stationery section. They made 36 sales today. How many sales were in the stationery section?",
        
        "Question: Katy makes coffee using teaspoons of sugar and cups of water in the ratio of 7:13. If she used a total of 120 teaspoons of sugar and cups of water, calculate the number of teaspoonfuls of sugar she used.",
        
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    ]
    
    for i, question in enumerate(math_questions, 1):
        print(f"\n" + "="*60)
        print(f"MATH QUESTION {i}:")
        print(question)
        print("\n" + "="*60)
        print("RESPONSE:")
        
        try:
            response = generate_response(model, tokenizer, question, max_new_tokens=500)
            print(response)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            print("This might be due to insufficient memory for the 7B model.")
        
        print(f"\n" + "-"*60)
    
    # Clean up
    del model, tokenizer
    gc.collect()
    
    print(f"\nFinal memory state:")
    check_memory()
    
    print("\nMath inference completed!")

if __name__ == "__main__":
    main() 