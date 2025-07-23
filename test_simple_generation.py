#!/usr/bin/env python3
"""Simple text generation test that works around Generator class issues."""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import qwen_nnx

def simple_generation_test():
    """Test complex math problems using direct model calls instead of Generator."""
    
    print("🧮 Testing Complex Math with Simple Generation")
    print("=" * 60)
    
    # Load model and tokenizer
    model_path = '../qwen25-7b/qwen25_7b_instruct_weights'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = qwen_nnx.QwenModel.load_from_hf_pt_model(model_path, dtype=jnp.bfloat16)
    
    def generate_simple(prompt, max_tokens=50):
        """Simple generation using model directly (like working examples)."""
        
        # Tokenize input
        input_ids = jnp.array(tokenizer.encode(prompt))[None, :]  # Add batch dim
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask
        attention_mask = jnp.ones((batch_size, seq_len))
        
        generated_tokens = []
        current_tokens = input_ids
        
        for i in range(max_tokens):
            # Forward pass through model
            logits = model(current_tokens, attention_mask=attention_mask)
            
            # Get next token (greedy decoding)
            next_token = jnp.argmax(logits[0, -1, :])
            generated_tokens.append(int(next_token))
            
            # Add to sequence for next iteration
            next_token_batch = jnp.array([[next_token]])
            current_tokens = jnp.concatenate([current_tokens, next_token_batch], axis=1)
            
            # Update attention mask
            attention_mask = jnp.ones((batch_size, current_tokens.shape[1]))
            
            # Stop if we hit EOS token
            if int(next_token) == tokenizer.eos_token_id:
                break
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "prompt": "3+1=",
            "description": "Simple addition",
            "expected": "4"
        },
        {
            "prompt": "What is 5+3? Answer:",
            "description": "Simple addition question",
            "expected": "8"
        },
        {
            "prompt": "Calculate 2*4=",
            "description": "Simple multiplication",
            "expected": "8"
        },
        {
            "prompt": "Q: If I have 10 apples and eat 3, how many are left?\nA:",
            "description": "Simple word problem",
            "expected": "7"
        },
        {
            "prompt": "Solve: 15 - 6 = ",
            "description": "Simple subtraction",
            "expected": "9"
        }
    ]
    
    print(f"Testing {len(test_cases)} problems...\n")
    
    correct_count = 0
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test {i}/{len(test_cases)}: {test_case['description']}")
        print(f"Prompt: '{test_case['prompt']}'")
        
        try:
            # Generate response
            generated = generate_simple(test_case['prompt'], max_tokens=20)
            print(f"Generated: '{generated}'")
            
            # Check if expected answer is in the generated text
            expected = str(test_case['expected'])
            is_correct = expected in generated
            
            if is_correct:
                correct_count += 1
                print(f"✅ CORRECT! Found expected answer: {expected}")
            else:
                print(f"❌ INCORRECT. Expected: {expected}")
            
            results.append({
                'test': test_case['description'],
                'prompt': test_case['prompt'],
                'expected': expected,
                'generated': generated,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'test': test_case['description'],
                'prompt': test_case['prompt'],
                'expected': test_case['expected'],
                'generated': f"ERROR: {str(e)}",
                'correct': False
            })
        
        print("-" * 60)
    
    # Summary
    accuracy = correct_count / len(test_cases) * 100
    print(f"\n📊 RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"Tests passed: {correct_count}/{len(test_cases)}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 60:
        print(f"\n🎉 SUCCESS! Model working with {accuracy:.1f}% accuracy!")
        print("📈 Now testing more complex problems...")
        test_complex_problems(model, tokenizer, generate_simple)
    else:
        print(f"\n🔧 Basic generation needs improvement: {accuracy:.1f}% accuracy")
        
    return results, accuracy

def test_complex_problems(model, tokenizer, generate_func):
    """Test more complex mathematical problems."""
    
    print("\n" + "=" * 60)
    print("🔬 TESTING COMPLEX MATHEMATICAL PROBLEMS")
    print("=" * 60)
    
    complex_cases = [
        {
            "prompt": "Question: Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4. How many can she sell?\nAnswer: Step by step:",
            "description": "GSM8K-style word problem",
            "expected": "9"  # 16 - 3 - 4 = 9
        },
        {
            "prompt": "Q: A robe takes 2 blue fiber and half that much white fiber. Total bolts?\nA:",
            "description": "Fraction calculation",
            "expected": "3"  # 2 + 1 = 3
        },
        {
            "prompt": "Math problem: 25 - 7 - 5 = ",
            "description": "Multi-step subtraction",
            "expected": "13"
        },
        {
            "prompt": "Calculate: 3 * 4 + 2 = ",
            "description": "Order of operations",
            "expected": "14"
        }
    ]
    
    correct_count = 0
    
    for i, test_case in enumerate(complex_cases, 1):
        print(f"\n🧮 Complex Test {i}/{len(complex_cases)}: {test_case['description']}")
        print(f"Prompt: {test_case['prompt'][:70]}...")
        
        try:
            generated = generate_func(test_case['prompt'], max_tokens=30)
            print(f"Generated: {generated[:100]}...")
            
            expected = str(test_case['expected'])
            is_correct = expected in generated
            
            if is_correct:
                correct_count += 1
                print(f"✅ CORRECT! Found: {expected}")
            else:
                print(f"❌ INCORRECT. Expected: {expected}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    complex_accuracy = correct_count / len(complex_cases) * 100
    print(f"\n📊 COMPLEX PROBLEMS SUMMARY:")
    print(f"Complex accuracy: {complex_accuracy:.1f}% ({correct_count}/{len(complex_cases)})")
    
    if complex_accuracy >= 50:
        print("🎉 EXCELLENT! Model handles complex problems well!")
    elif complex_accuracy >= 25:
        print("📈 GOOD PROGRESS! Some complex problems working.")
    else:
        print("🔧 NEEDS WORK: Focus on prompt engineering for complex problems.")
    
    return complex_accuracy

if __name__ == "__main__":
    results, accuracy = simple_generation_test()
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    if accuracy >= 80:
        print("🌟 OUTSTANDING: Model excels at mathematical reasoning!")
    elif accuracy >= 60:
        print("✅ SUCCESS: Model demonstrates solid math capabilities!")
    elif accuracy >= 40:
        print("📈 PROGRESS: Model shows math potential, needs refinement.")
    else:
        print("🔧 DEVELOPMENT NEEDED: Focus on basic math generation.") 