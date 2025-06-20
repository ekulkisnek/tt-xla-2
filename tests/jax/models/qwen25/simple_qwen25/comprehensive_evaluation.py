#!/usr/bin/env python3
"""
Comprehensive evaluation of qwen25_tp_final.py across different prompt types
Tests reasoning, math, coding, general knowledge, and instruction following
"""
import subprocess
import sys
import time
import json
from typing import List, Dict, Tuple

def run_model_test(prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> Tuple[bool, str, float]:
    """Run a single test with the model"""
    cmd = [
        sys.executable, "qwen25_tp_final.py",
        "--model_path", "../weights",
        "--prompt", prompt,
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--tp", "1"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Extract generated text from output
            lines = result.stdout.strip().split('\n')
            generated_text = ""
            for line in lines:
                if "Generating text with prompt:" in line:
                    continue
                elif line and not line.startswith("2025-") and not "INFO" in line and not "Special tokens" in line and not "Unable to initialize" in line:
                    generated_text = line
                    break
            
            return True, generated_text, elapsed
        else:
            return False, f"Error: {result.stderr}", elapsed
            
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", 300
    except Exception as e:
        return False, f"Exception: {e}", 0

def main():
    """Run comprehensive evaluation"""
    print("🧪 COMPREHENSIVE EVALUATION OF QWEN2.5-7B TENSOR PARALLEL")
    print("=" * 80)
    
    # Test categories with various prompt types
    test_categories = {
        "🧮 MATH & REASONING (GSM8K-style)": [
            ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 cakes using 4 eggs each. She sells the remainder at the farmers' market for $2 per fresh duck egg. How much money does she make?", 100),
            ("A train travels 120 miles in 2 hours. How fast is the train going in miles per hour?", 30),
            ("If 5 apples cost $3, how much do 8 apples cost?", 30),
            ("Tom has 3 times as many marbles as Jack. If Jack has 12 marbles, how many marbles do they have together?", 40),
            ("A rectangle has length 8 and width 5. What is its perimeter?", 30),
            ("Sarah saves $15 per week. How much will she save in 6 weeks?", 30),
        ],
        
        "💭 LOGICAL REASONING": [
            ("All birds can fly. Penguins are birds. Therefore,", 20),
            ("If it's raining, then the ground is wet. The ground is wet. Therefore,", 30),
            ("A is bigger than B. B is bigger than C. Therefore,", 20),
            ("Every cat has whiskers. Fluffy is a cat. Therefore,", 20),
        ],
        
        "🌍 GENERAL KNOWLEDGE": [
            ("The capital of France is", 10),
            ("The largest planet in our solar system is", 10),
            ("Who wrote Romeo and Juliet?", 15),
            ("What is the chemical symbol for gold?", 10),
            ("The Great Wall of China was built in", 15),
            ("Mount Everest is located in", 15),
        ],
        
        "💻 CODING & TECHNICAL": [
            ("Write a Python function to calculate factorial:", 80),
            ("In Python, to create a list you use", 20),
            ("What does API stand for?", 20),
            ("The time complexity of binary search is", 15),
            ("def hello():\n    print(", 15),
        ],
        
        "📚 INSTRUCTION FOLLOWING": [
            ("Explain in simple terms what photosynthesis is.", 60),
            ("List three benefits of exercise.", 40),
            ("Translate 'Hello, how are you?' to Spanish:", 20),
            ("Write a haiku about nature.", 40),
            ("Give me a recipe for scrambled eggs.", 80),
        ],
        
        "🎭 CREATIVE & OPEN-ENDED": [
            ("Once upon a time, in a magical forest,", 100),
            ("The future of artificial intelligence will", 60),
            ("If I could travel anywhere in the world, I would", 50),
            ("The most important invention in human history is", 40),
        ],
        
        "🔤 LANGUAGE & GRAMMAR": [
            ("Complete the sentence: The quick brown fox", 20),
            ("The past tense of 'run' is", 10),
            ("What is the opposite of 'hot'?", 10),
            ("A group of lions is called a", 10),
        ],
        
        "🧩 PATTERN COMPLETION": [
            ("2, 4, 6, 8,", 15),
            ("Monday, Tuesday, Wednesday,", 15),
            ("A, B, C, D,", 10),
            ("Red, Orange, Yellow,", 15),
        ]
    }
    
    # Run all tests
    all_results = {}
    total_tests = sum(len(prompts) for prompts in test_categories.values())
    current_test = 0
    
    for category, prompts in test_categories.items():
        print(f"\n{category}")
        print("-" * 60)
        
        category_results = []
        
        for prompt, max_tokens in prompts:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Test with greedy decoding
            success, output, elapsed = run_model_test(prompt, max_tokens, temperature=0.0)
            
            result = {
                "prompt": prompt,
                "success": success,
                "output": output,
                "elapsed": elapsed,
                "max_tokens": max_tokens
            }
            
            category_results.append(result)
            
            if success:
                print(f"  ✅ SUCCESS ({elapsed:.1f}s): '{output[:80]}{'...' if len(output) > 80 else ''}'")
            else:
                print(f"  ❌ FAILED ({elapsed:.1f}s): {output}")
            
            # Brief pause between tests
            time.sleep(1)
        
        all_results[category] = category_results
    
    # Generate comprehensive analysis
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    total_success = sum(sum(1 for r in results if r["success"]) for results in all_results.values())
    total_failed = total_tests - total_success
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {total_success} ({total_success/total_tests*100:.1f}%)")
    print(f"   Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    
    # Category breakdown
    print(f"\n📈 CATEGORY BREAKDOWN:")
    for category, results in all_results.items():
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        avg_time = sum(r["elapsed"] for r in results if r["success"]) / max(successful, 1)
        
        print(f"   {category}")
        print(f"      Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"      Avg Response Time: {avg_time:.1f}s")
        print()
    
    # Detailed analysis by category
    print("\n🔍 DETAILED ANALYSIS BY CATEGORY:")
    print("=" * 60)
    
    for category, results in all_results.items():
        print(f"\n{category}")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            status = "✅" if result["success"] else "❌"
            print(f"{i}. {status} Input: '{result['prompt'][:60]}{'...' if len(result['prompt']) > 60 else ''}'")
            print(f"   Output: '{result['output'][:100]}{'...' if len(result['output']) > 100 else ''}'")
            print(f"   Time: {result['elapsed']:.1f}s")
            print()
    
    # Math problem analysis
    print("\n🧮 MATHEMATICAL REASONING ANALYSIS:")
    print("=" * 60)
    
    math_category = "🧮 MATH & REASONING (GSM8K-style)"
    if math_category in all_results:
        math_results = all_results[math_category]
        
        print("Analyzing mathematical reasoning capabilities...")
        print()
        
        for i, result in enumerate(math_results, 1):
            print(f"Problem {i}: {result['prompt']}")
            print(f"Model Output: {result['output']}")
            
            # Basic analysis of mathematical content
            output_lower = result['output'].lower()
            has_number = any(char.isdigit() for char in result['output'])
            has_calculation_words = any(word in output_lower for word in ['calculate', 'multiply', 'add', 'subtract', 'divide', 'equals', 'total', 'answer'])
            
            print(f"Contains Numbers: {'Yes' if has_number else 'No'}")
            print(f"Contains Math Language: {'Yes' if has_calculation_words else 'No'}")
            print("-" * 40)
    
    # Save results to JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Quality assessment
    print("\n🎯 QUALITY ASSESSMENT:")
    print("=" * 60)
    
    # Check for common issues
    all_outputs = [r["output"] for results in all_results.values() for r in results if r["success"]]
    
    # Language mixing detection
    mixed_language_count = 0
    for output in all_outputs:
        if any(ord(char) > 127 for char in output):  # Non-ASCII characters
            mixed_language_count += 1
    
    # Repetition detection
    repetitive_count = 0
    for output in all_outputs:
        words = output.split()
        if len(words) > 3:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.7:  # Less than 70% unique words
                repetitive_count += 1
    
    # Short/truncated responses
    short_response_count = sum(1 for output in all_outputs if len(output.strip()) < 3)
    
    print(f"Mixed Language Responses: {mixed_language_count}/{len(all_outputs)} ({mixed_language_count/len(all_outputs)*100:.1f}%)")
    print(f"Repetitive Responses: {repetitive_count}/{len(all_outputs)} ({repetitive_count/len(all_outputs)*100:.1f}%)")
    print(f"Very Short Responses: {short_response_count}/{len(all_outputs)} ({short_response_count/len(all_outputs)*100:.1f}%)")
    
    print(f"\n✅ Evaluation complete! Check {results_file} for full results.")
    
    return all_results

if __name__ == "__main__":
    main() 