#!/usr/bin/env python3
"""
Run greedy generation with PyTorch model for Gate 4 comparison.
"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def main():
    parser = argparse.ArgumentParser(description="Run PyTorch greedy generation")
    parser.add_argument("--model_path", type=str, default="../weights", help="Path to model weights")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=64, help="Maximum tokens to generate")
    parser.add_argument("--out", type=str, required=True, help="Output text file")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    
    # Set memory-efficient environment
    torch.set_grad_enabled(False)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    
    print(f"Loading PyTorch model and tokenizer...")
    print(f"Model path: {args.model_path}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model.eval()
    
    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"Input token IDs: {input_ids.tolist()}")
    print(f"Input length: {input_ids.shape[1]} tokens")
    
    # Greedy generation
    print("Running greedy generation...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,  # Greedy decoding
            temperature=1.0,  # Doesn't matter for greedy
            top_p=1.0,        # Doesn't matter for greedy
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Generated length: {outputs.shape[1]} tokens")
    print(f"New tokens generated: {outputs.shape[1] - input_ids.shape[1]}")
    
    # Save to file
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    
    print(f"Generated text saved to {args.out}")
    print("=" * 60)
    print("GENERATED TEXT:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    
    # Cleanup
    del model, tokenizer, outputs, inputs
    gc.collect()
    
    print("PyTorch cleanup completed.")

if __name__ == "__main__":
    main() 