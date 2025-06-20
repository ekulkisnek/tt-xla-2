#!/usr/bin/env python3
"""
Quick comprehensive evaluation with shorter prompts and reduced timeouts
"""
import subprocess
import sys
import time
import json
from typing import List, Dict, Tuple

def run_model_test(prompt: str, max_tokens: int = 15, temperature: float = 0.0, timeout: int = 120) -> Tuple[bool, str, float]:
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
            return False, f"Error: {result.stderr[:100]}", elapsed
            
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", timeout
    except Exception as e:
        return False, f"Exception: {e}", 0

def main():
    """Run quick comprehensive evaluation"""
    print("🧪 QUICK COMPREHENSIVE EVALUATION OF QWEN2.5-7B")
    print("=" * 60)
    
    # Test categories with shorter, simpler prompts
    test_categories = {
        "🧮 MATH": [
            ("2+2=", 5),
            ("5*3=", 5),
            ("10-4=", 5),
            ("12/3=", 5),
            ("What is 7+8?", 10),
        ],
        
        "💭 LOGIC": [
            ("All cats are animals. Fluffy is a cat. Therefore,", 15),
            ("If A>B and B>C, then", 10),
            ("True or False: 5>3", 5),
        ],
        
        "🌍 KNOWLEDGE": [
            ("The capital of France is", 5),
            ("2+2 equals", 5),
            ("The sun is a", 5),
            ("Water boils at", 8),
            ("The first president of the USA was", 10),
        ],
        
        "💻 TECHNICAL": [
            ("In Python, print(", 8),
            ("HTML stands for", 10),
            ("def add(a,b):", 15),
        ],
        
        "📚 INSTRUCTION": [
            ("List 3 colors:", 15),
            ("Count to 5:", 15),
            ("Say hello in Spanish:", 10),
        ],
        
        "🎭 CREATIVE": [
            ("Once upon a time", 20),
            ("The weather today is", 15),
            ("My favorite color is", 10),
        ],
        
        "🔤 LANGUAGE": [
            ("The opposite of hot is", 5),
            ("A, B, C,", 5),
            ("Yesterday, today,", 5),
        ]
    }
    
    # Run all tests
    all_results = {}
    total_tests = sum(len(prompts) for prompts in test_categories.values())
    current_test = 0
    
    for category, prompts in test_categories.items():
        print(f"\n{category}")
        print("-" * 40)
        
        category_results = []
        
        for prompt, max_tokens in prompts:
            current_test += 1
            print(f"[{current_test}/{total_tests}] '{prompt[:30]}{'...' if len(prompt) > 30 else ''}'", end=" ")
            
            # Test with greedy decoding and shorter timeout
            success, output, elapsed = run_model_test(prompt, max_tokens, temperature=0.0, timeout=120)
            
            result = {
                "prompt": prompt,
                "success": success,
                "output": output,
                "elapsed": elapsed,
                "max_tokens": max_tokens
            }
            
            category_results.append(result)
            
            if success:
                print(f"✅ ({elapsed:.1f}s): '{output[:50]}{'...' if len(output) > 50 else ''}'")
            else:
                print(f"❌ ({elapsed:.1f}s): {output[:30]}")
        
        all_results[category] = category_results
    
    # Analysis
    print("\n" + "=" * 60)
    print("📊 RESULTS ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    total_success = sum(sum(1 for r in results if r["success"]) for results in all_results.values())
    total_failed = total_tests - total_success
    
    print(f"\n🎯 OVERALL:")
    print(f"   Success: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    print(f"   Failed: {total_failed}/{total_tests} ({total_failed/total_tests*100:.1f}%)")
    
    # Category breakdown
    print(f"\n📈 BY CATEGORY:")
    for category, results in all_results.items():
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        
        print(f"   {category}: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Detailed results
    print(f"\n🔍 DETAILED RESULTS:")
    for category, results in all_results.items():
        print(f"\n{category}:")
        for i, result in enumerate(results, 1):
            status = "✅" if result["success"] else "❌"
            print(f"  {i}. {status} '{result['prompt']}' → '{result['output']}'")
    
    # Quality assessment
    successful_outputs = [r["output"] for results in all_results.values() for r in results if r["success"]]
    
    if successful_outputs:
        print(f"\n🎯 QUALITY METRICS:")
        
        # Language mixing
        mixed_lang = sum(1 for output in successful_outputs if any(ord(c) > 127 for c in output))
        print(f"   Mixed Language: {mixed_lang}/{len(successful_outputs)} ({mixed_lang/len(successful_outputs)*100:.1f}%)")
        
        # Very short responses
        short = sum(1 for output in successful_outputs if len(output.strip()) < 3)
        print(f"   Very Short: {short}/{len(successful_outputs)} ({short/len(successful_outputs)*100:.1f}%)")
        
        # Coherent responses (rough heuristic)
        coherent = sum(1 for output in successful_outputs if len(output.split()) > 1 and not output.startswith(('.', '月', '#')))
        print(f"   Appears Coherent: {coherent}/{len(successful_outputs)} ({coherent/len(successful_outputs)*100:.1f}%)")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"quick_eval_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return all_results

if __name__ == "__main__":
    main() 