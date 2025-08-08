#!/usr/bin/env python3
"""
Qwen2.5-7b Full Model Inference Script for GSM8K Evaluation
This script loads the full 7B model and evaluates all 1319 GSM8K questions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import gc
import psutil
import os
import re
import json

def extract_boxed_answer(text):
    """Extract the final answer from GSM8K format text."""
    # First try to find boxed format \boxed{...}
    match = re.search(r'\\boxed{([0-9]+)}', text)
    if match:
        return int(match.group(1))
    
    # Then try GSM8K format #### number
    match = re.search(r'####\s*([0-9]+)', text)
    if match:
        return int(match.group(1))
    
    # Finally try to find any number at the end of the text
    match = re.search(r'([0-9]+)\s*$', text.strip())
    if match:
        return int(match.group(1))
    
    return None

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
            max_new_tokens=500,  # Maximum 500 tokens
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

def generate_response_for_gsm8k(model, tokenizer, question, max_new_tokens=500):
    """Generate a response for GSM8K evaluation with proper stopping logic"""
    
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
    
    # Manual generation loop with proper stopping logic
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
        
        # Show the numbered token in real-time
        print(f"{i+1}: {new_token_text}", end="", flush=True)
        
        # Check for end of sequence
        if next_token.item() == tokenizer.eos_token_id:
            print(f"\n[EOS token reached]")
            break
        
        # Check for end of response markers (like in test_gsm8k.py)
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

def evaluate_gsm8k_full(model, tokenizer, start_index=0, max_samples=None):
    """Evaluate all GSM8K questions"""
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Get all questions or limit if specified
    if max_samples:
        test_data = dataset[start_index:start_index + max_samples]
    else:
        test_data = dataset[start_index:]
    
    total_questions = len(test_data)
    print(f"Evaluating {total_questions} questions starting from index {start_index}")
    
    # Handle the dataset structure properly
    questions = test_data["question"]
    answers = test_data["answer"]
    
    correct = 0
    total_time = 0
    total_tokens = 0
    results = []
    
    for i in range(len(questions)):
        print(f"\n{'='*80}")
        print(f"Processing question {i+1}/{len(questions)} (Global index: {start_index + i})")
        print(f"{'='*80}")
        
        # Use actual GSM8K questions
        prompt = questions[i]
        target = extract_boxed_answer(answers[i])
        
        print(f"Question: {prompt}")
        print(f"Target answer: {target}")
        
        start_time = time.time()
        
        try:
            # Generate response
            output = generate_response_for_gsm8k(model, tokenizer, prompt, max_new_tokens=500)
            predicted = extract_boxed_answer(output)
            
            generation_time = time.time() - start_time
            total_time += generation_time
            
            # Count tokens (rough estimate)
            output_tokens = len(output.split())
            total_tokens += output_tokens
            
            # Check correctness
            is_correct = predicted == target
            if is_correct:
                correct += 1
                print(f"✅ CORRECT! Predicted: {predicted} | Target: {target}")
            else:
                print(f"❌ WRONG! Predicted: {predicted} | Target: {target}")
            
            # Store result
            result = {
                "index": start_index + i,
                "question": prompt,
                "target": target,
                "predicted": predicted,
                "correct": is_correct,
                "response": output,
                "generation_time": generation_time,
                "tokens_generated": output_tokens
            }
            results.append(result)
            
            print(f"Current accuracy: {correct}/{i+1} ({correct/(i+1)*100:.1f}%)")
            print(f"Average time per question: {total_time/(i+1):.2f} seconds")
            
        except Exception as e:
            print(f"Error during generation: {e}")
            result = {
                "index": start_index + i,
                "question": prompt,
                "target": target,
                "predicted": None,
                "correct": False,
                "response": f"ERROR: {str(e)}",
                "generation_time": 0,
                "tokens_generated": 0
            }
            results.append(result)
        
        # Memory cleanup between questions
        gc.collect()
        
        # Save intermediate results every 50 questions
        if (i + 1) % 50 == 0:
            intermediate_results = {
                "total_processed": i + 1,
                "correct": correct,
                "accuracy": correct / (i + 1) * 100,
                "total_time": total_time,
                "avg_time_per_question": total_time / (i + 1),
                "total_tokens": total_tokens,
                "results": results
            }
            
            with open(f"gsm8k_results_intermediate_{start_index}_{i+1}.json", "w") as f:
                json.dump(intermediate_results, f, indent=2)
            print(f"Saved intermediate results to gsm8k_results_intermediate_{start_index}_{i+1}.json")
    
    # Final results
    accuracy = correct / total_questions * 100
    avg_time_per_question = total_time / total_questions if total_questions > 0 else 0
    
    final_results = {
        "total_questions": total_questions,
        "start_index": start_index,
        "correct": correct,
        "accuracy": accuracy,
        "total_time": total_time,
        "avg_time_per_question": avg_time_per_question,
        "total_tokens": total_tokens,
        "results": results
    }
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS:")
    print(f"GSM8K Accuracy ({total_questions} questions): {accuracy:.2f}%")
    print(f"Correct: {correct}/{total_questions}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per question: {avg_time_per_question:.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"{'='*80}")
    
    # Save final results
    with open(f"gsm8k_results_final_{start_index}_{total_questions}.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved final results to gsm8k_results_final_{start_index}_{total_questions}.json")
    
    return accuracy, final_results

def main():
    """Main inference function for GSM8K evaluation"""
    
    print("="*80)
    print("QWEN2.5-7B MODEL INFERENCE - FULL GSM8K EVALUATION")
    print("="*80)
    
    # Load model
    model, tokenizer, model_name = load_model_with_optimizations()
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    print(f"\nUsing model: {model_name}")
    
    # Evaluate all GSM8K questions
    # You can modify these parameters as needed
    start_index = 0  # Start from the beginning
    max_samples = None  # Set to a number to limit samples, or None for all 1319
    
    if max_samples:
        print(f"Evaluating {max_samples} questions starting from index {start_index}")
    else:
        print(f"Evaluating all questions starting from index {start_index}")
    
    try:
        accuracy, results = evaluate_gsm8k_full(model, tokenizer, start_index, max_samples)
        print(f"\nEvaluation completed with {accuracy:.2f}% accuracy!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
    
    # Clean up
    del model, tokenizer
    gc.collect()
    
    print(f"\nFinal memory state:")
    check_memory()
    
    print("\nGSM8K evaluation completed!")

if __name__ == "__main__":
    main() 