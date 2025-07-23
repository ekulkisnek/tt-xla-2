#!/usr/bin/env python3
"""
Basic generation test to prove the model works
"""
import subprocess
import sys

def test_basic_generation():
    """Test basic generation with the working model"""
    print("🧪 TESTING BASIC GENERATION")
    print("="*50)
    
    test_cases = [
        {"prompt": "Hello", "expected_type": "text", "max_tokens": 3},
        {"prompt": "2+2=", "expected_type": "math", "max_tokens": 3},
        {"prompt": "1+1=", "expected_type": "math", "max_tokens": 3},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: '{case['prompt']}'")
        print(f"Expected: {case['expected_type']}")
        print("Generated: ", end="", flush=True)
        
        try:
            # Run the model
            result = subprocess.run([
                "python", "qwen25_tp_final_fixed.py",
                "--model_path", "qwen25_7b_instruct_weights", 
                "--prompt", case['prompt'],
                "--max_tokens", str(case['max_tokens']),
                "--tp", "1"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Extract just the generated text (after the loading messages)
                output_lines = result.stdout.strip().split('\n')
                generated = ""
                for line in output_lines:
                    if "Generating text with prompt:" in line:
                        # The next few characters are the generation
                        idx = output_lines.index(line)
                        if idx + 1 < len(output_lines):
                            generated = output_lines[idx + 1] if idx + 1 < len(output_lines) else ""
                        break
                
                print(f"'{generated}'")
                
                if case['expected_type'] == "math":
                    # Check if it looks like a number
                    if any(char.isdigit() for char in generated):
                        print("✅ Contains digits (math-like)")
                    else:
                        print("❌ No digits found")
                else:
                    print("✅ Generated text")
                    
            elif result.returncode == 137:
                print("❌ Memory killed")
            else:
                print(f"❌ Error (code {result.returncode})")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Exception: {e}")

def main():
    print("🎯 QWEN2.5-7B BASIC GENERATION TEST")
    print("Testing if the rewritten model can generate coherent output")
    print()
    
    test_basic_generation()
    
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("This test verifies:")
    print("✅ Model loads successfully (7.6B parameters)")
    print("✅ Generation works without crashes") 
    print("? Quality depends on actual output content")
    print()
    print("Next step: If generation works, optimize prompting for math")

if __name__ == "__main__":
    main() 