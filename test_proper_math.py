#!/usr/bin/env python3
"""
Test proper math prompting using patterns from working GSM8K implementations
"""
import subprocess
import sys

def test_with_proper_prompting():
    """Test using the exact prompting patterns from working gsm8k implementations"""
    
    print("🧮 TESTING PROPER MATH PROMPTING")
    print("Using patterns from working gsm8k_simple.py and gsm8k_instruct_test.py")
    print("="*70)
    
    # Test cases using EXACT patterns from working implementations
    test_cases = [
        {
            "prompt": "Q: What is 2+2?\n\nA: Let's think step by step.\n",
            "description": "GSM8K format from gsm8k_simple.py",
            "expected": "4"
        },
        {
            "prompt": "Solve this math problem step by step:\n\nWhat is 2+2?\n\nShow your work and provide the final numerical answer.",
            "description": "Instruction format from gsm8k_instruct_test.py", 
            "expected": "4"
        },
        {
            "prompt": "Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?\n\nA: Let's solve this step by step.\n",
            "description": "Full GSM8K problem format",
            "expected": "18" 
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {case['description']}")
        print(f"Expected answer: {case['expected']}")
        print(f"Prompt: {case['prompt'][:100]}...")
        print("Generated: ", end="", flush=True)
        
        try:
            result = subprocess.run([
                "python", "qwen25_tp_final_fixed.py",
                "--model_path", "qwen25_7b_instruct_weights",
                "--prompt", case['prompt'],
                "--max_tokens", "50",
                "--tp", "1"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Extract generated text from the output
                lines = result.stdout.strip().split('\n')
                generated = ""
                for line in lines:
                    if "Generating text with prompt:" in line:
                        # Look for the actual generated content
                        continue
                    elif line and not line.startswith("2025-") and "INFO" not in line:
                        generated = line.strip()
                        break
                
                print(f"'{generated}'")
                
                # Check if it contains the expected answer
                if case['expected'] in generated:
                    print("✅ Contains expected answer!")
                elif any(char.isdigit() for char in generated):
                    print("⚠️  Contains numbers (partially working)")
                else:
                    print("❌ No expected answer found")
                    
            elif result.returncode == 137:
                print("❌ Memory killed")
            else:
                print(f"❌ Error (code {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Exception: {e}")

def test_chat_template():
    """Test if we can use chat template format like the working q25_jax_instruct.py"""
    print(f"\n🗨️  TESTING CHAT TEMPLATE FORMAT")
    print("="*70)
    
    # Try to run with chat template (like q25_jax_instruct.py does)
    chat_prompt = "What is 2+2? Please solve step by step."
    
    print(f"Testing with chat template format...")
    print(f"Prompt: {chat_prompt}")
    print("Generated: ", end="", flush=True)
    
    try:
        # First try to see if we have a working q25_jax_instruct.py equivalent
        result = subprocess.run([
            "python", "qwen25_tp_final_fixed.py", 
            "--model_path", "qwen25_7b_instruct_weights",
            "--prompt", chat_prompt,
            "--max_tokens", "20",
            "--tp", "1"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            generated = ""
            for line in lines:
                if line and not line.startswith("2025-") and "INFO" not in line and "Generating" not in line:
                    generated = line.strip()
                    break
            print(f"'{generated}'")
        else:
            print(f"Failed with code {result.returncode}")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("🎯 PROPER MATH TESTING")
    print("Using exact prompting patterns from working implementations")
    print()
    
    test_with_proper_prompting()
    test_chat_template()
    
    print("\n" + "="*70)
    print("📊 DIAGNOSIS")
    print("The issue is likely:")
    print("1. ❌ Raw prompts like '2+2=' don't work with instruct models")
    print("2. ❌ Missing proper chat templates (<|im_start|>/<|im_end|>)")
    print("3. ❌ Need 'step by step' reasoning prompts")
    print("4. ❌ Model needs proper instruction format to activate reasoning")
    print()
    print("Next: Implement chat template support in our model!")

if __name__ == "__main__":
    main() 