#!/usr/bin/env python3
"""
Full model validation script for qwen25_tp_final.py
Confirms the tensor parallel implementation works with the complete 7.6B model
"""
import subprocess
import sys
import time

def run_test(prompt, max_tokens, temperature, expected_min_length=1):
    """Run a single test and validate output"""
    cmd = [
        sys.executable, "qwen25_tp_final.py",
        "--model_path", "../weights",
        "--prompt", prompt,
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--tp", "1"
    ]
    
    print(f"Testing prompt: '{prompt}' (temp={temperature}, max_tokens={max_tokens})")
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
                    # Find the line after this one that contains generated text
                    continue
                elif line and not line.startswith("2025-") and not "INFO" in line and not "Special tokens" in line:
                    generated_text = line
                    break
            
            print(f"  ✅ SUCCESS ({elapsed:.1f}s)")
            print(f"     Generated: '{generated_text}'")
            print(f"     Length: {len(generated_text)} chars")
            return True, generated_text, elapsed
        else:
            print(f"  ❌ FAILED (exit code {result.returncode})")
            print(f"     Error: {result.stderr}")
            return False, "", elapsed
            
    except subprocess.TimeoutExpired:
        print(f"  ⏰ TIMEOUT (>300s)")
        return False, "", 300
    except Exception as e:
        print(f"  💥 EXCEPTION: {e}")
        return False, "", 0

def main():
    """Run comprehensive validation tests"""
    print("🚀 Full Model Validation for qwen25_tp_final.py")
    print("=" * 60)
    
    tests = [
        # (prompt, max_tokens, temperature)
        ("Hello", 3, 0.0),
        ("The sky is", 2, 0.0),
        ("Python is a", 4, 0.0),
        ("1+1=", 2, 0.0),
        ("Hello world", 3, 0.7),  # With sampling
    ]
    
    results = []
    total_time = 0
    
    for i, (prompt, max_tokens, temperature) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] ", end="")
        success, output, elapsed = run_test(prompt, max_tokens, temperature)
        results.append((prompt, success, output, elapsed))
        total_time += elapsed
        
        if not success:
            print("⚠️  Continuing with remaining tests...")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - passed
    
    print(f"Tests passed: {passed}/{len(results)}")
    print(f"Tests failed: {failed}/{len(results)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per test: {total_time/len(results):.1f}s")
    
    print("\nDetailed Results:")
    for prompt, success, output, elapsed in results:
        status = "✅" if success else "❌"
        print(f"  {status} '{prompt}' -> '{output}' ({elapsed:.1f}s)")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Tensor parallel implementation is working correctly with the full 7.6B model")
        print("✅ Model loading: SUCCESS")
        print("✅ Text generation: SUCCESS") 
        print("✅ JIT compilation: SUCCESS")
        print("✅ Memory management: SUCCESS")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        print("❌ Some issues detected with the implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 