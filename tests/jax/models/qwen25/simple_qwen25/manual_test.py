#!/usr/bin/env python3
"""
Manual testing script to evaluate model behavior systematically
"""
import subprocess
import sys
import time

def test_prompt(prompt, max_tokens=20, description=""):
    """Test a single prompt and show results"""
    print(f"\n{'='*60}")
    print(f"TESTING: {description if description else prompt}")
    print(f"PROMPT: '{prompt}'")
    print(f"MAX_TOKENS: {max_tokens}")
    print("-" * 60)
    
    cmd = [
        sys.executable, "qwen25_tp_final.py",
        "--model_path", "../weights",
        "--prompt", prompt,
        "--max_tokens", str(max_tokens),
        "--temperature", "0.0",
        "--tp", "1"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "Generating text with prompt:" in line:
                    continue
                elif line and not line.startswith("2025-") and not "INFO" in line and not "Special tokens" in line and not "Unable to initialize" in line:
                    print(f"✅ OUTPUT ({elapsed:.1f}s): '{line}'")
                    break
            else:
                print(f"⚠️  NO OUTPUT EXTRACTED ({elapsed:.1f}s)")
                print("STDOUT:", result.stdout[-200:])
        else:
            print(f"❌ ERROR ({elapsed:.1f}s): {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (180s)")
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

def main():
    """Run manual tests"""
    print("🧪 MANUAL TESTING OF QWEN2.5-7B")
    
    # Test various categories
    tests = [
        # Basic completion
        ("The capital of France is", 5, "Geography - Simple Fact"),
        ("2+2=", 3, "Math - Simple Addition"),
        ("Hello, my name is", 10, "Basic Conversation"),
        
        # Math problems
        ("What is 5 times 3?", 15, "Math - Multiplication Question"),
        ("If I have 10 apples and eat 3, how many are left?", 20, "Math - Word Problem"),
        
        # Reasoning
        ("All cats are animals. Fluffy is a cat. Therefore,", 15, "Logic - Syllogism"),
        ("If it's raining, the ground gets wet. The ground is wet. What can we conclude?", 25, "Logic - Inference"),
        
        # General knowledge
        ("The first president of the United States was", 10, "History - US Presidents"),
        ("Water boils at", 8, "Science - Basic Facts"),
        ("The largest planet in our solar system is", 10, "Astronomy - Solar System"),
        
        # Language
        ("The opposite of hot is", 5, "Language - Antonyms"),
        ("Please translate 'hello' to Spanish:", 10, "Language - Translation"),
        
        # Coding
        ("In Python, to print text you use the", 15, "Programming - Python Basics"),
        ("def hello_world():", 20, "Programming - Function"),
        
        # Creative
        ("Once upon a time, there was a", 25, "Creative - Story Beginning"),
        ("The weather today looks", 15, "Creative - Description"),
        
        # Instructions
        ("List three colors:", 15, "Instruction - Simple List"),
        ("Count from 1 to 5:", 15, "Instruction - Counting"),
        
        # Pattern completion
        ("1, 2, 3,", 10, "Pattern - Numbers"),
        ("Monday, Tuesday,", 10, "Pattern - Days"),
        ("A, B, C,", 8, "Pattern - Letters"),
    ]
    
    # Run each test
    for i, (prompt, max_tokens, description) in enumerate(tests, 1):
        print(f"\n[TEST {i}/{len(tests)}]")
        test_prompt(prompt, max_tokens, description)
        time.sleep(2)  # Brief pause between tests
    
    print(f"\n{'='*60}")
    print("🎯 TESTING COMPLETE")
    print("Manually review the outputs above for quality assessment")
    print("='*60}")

if __name__ == "__main__":
    main() 