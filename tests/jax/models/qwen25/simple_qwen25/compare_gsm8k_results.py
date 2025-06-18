#!/usr/bin/env python3
"""
Compare GSM8K results between JAX and PyTorch implementations.
Usage: python compare_gsm8k_results.py jax_gsm8k.jsonl pytorch_gsm8k.jsonl
"""
import json
import argparse
import sys

def load_results(file_path):
    """Load results from JSONL file"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def compare_results(jax_results, pytorch_results):
    """Compare JAX and PyTorch results"""
    if len(jax_results) != len(pytorch_results):
        print(f"ERROR: Different number of results! JAX: {len(jax_results)}, PyTorch: {len(pytorch_results)}")
        return
    
    total_problems = len(jax_results)
    jax_correct = sum(1 for r in jax_results if r["correct"])
    pytorch_correct = sum(1 for r in pytorch_results if r["correct"])
    
    both_correct = 0
    both_wrong = 0
    jax_only_correct = 0
    pytorch_only_correct = 0
    answer_mismatches = 0
    
    print(f"=== GSM8K PARITY ANALYSIS ===")
    print(f"Total problems: {total_problems}")
    print(f"JAX accuracy: {jax_correct/total_problems*100:.2f}% ({jax_correct}/{total_problems})")
    print(f"PyTorch accuracy: {pytorch_correct/total_problems*100:.2f}% ({pytorch_correct}/{total_problems})")
    print()
    
    mismatched_problems = []
    
    for i, (jax_r, pt_r) in enumerate(zip(jax_results, pytorch_results)):
        jax_pred = jax_r["predicted"]
        pt_pred = pt_r["predicted"]
        gt = jax_r["ground_truth"]
        
        # Check if predictions match
        if jax_pred != pt_pred:
            answer_mismatches += 1
            mismatched_problems.append({
                "id": i,
                "jax_pred": jax_pred,
                "pytorch_pred": pt_pred,
                "ground_truth": gt,
                "jax_correct": jax_r["correct"],
                "pytorch_correct": pt_r["correct"]
            })
        
        # Count agreement patterns
        if jax_r["correct"] and pt_r["correct"]:
            both_correct += 1
        elif not jax_r["correct"] and not pt_r["correct"]:
            both_wrong += 1
        elif jax_r["correct"] and not pt_r["correct"]:
            jax_only_correct += 1
        elif not jax_r["correct"] and pt_r["correct"]:
            pytorch_only_correct += 1
    
    print(f"Agreement Analysis:")
    print(f"  Both correct: {both_correct}")
    print(f"  Both wrong: {both_wrong}")
    print(f"  JAX only correct: {jax_only_correct}")
    print(f"  PyTorch only correct: {pytorch_only_correct}")
    print(f"  Answer mismatches: {answer_mismatches}")
    print()
    
    # Calculate agreement rate
    agreement_rate = (both_correct + both_wrong) / total_problems * 100
    print(f"Agreement rate: {agreement_rate:.2f}%")
    
    # Score difference
    score_diff = abs(jax_correct - pytorch_correct) / total_problems * 100
    print(f"Score difference: {score_diff:.2f}%")
    
    # Parity status
    if score_diff <= 0.1:
        print("✅ PARITY STATUS: PASS (≤0.1% difference)")
    elif score_diff <= 1.0:
        print("⚠️  PARITY STATUS: CLOSE (≤1.0% difference)")
    else:
        print("❌ PARITY STATUS: FAIL (>1.0% difference)")
    
    print()
    
    # Show some mismatched examples
    if mismatched_problems:
        print(f"=== FIRST 10 MISMATCHED PROBLEMS ===")
        for prob in mismatched_problems[:10]:
            print(f"Problem {prob['id']}:")
            print(f"  JAX: {prob['jax_pred']} ({'✓' if prob['jax_correct'] else '✗'})")
            print(f"  PyTorch: {prob['pytorch_pred']} ({'✓' if prob['pytorch_correct'] else '✗'})")
            print(f"  Ground Truth: {prob['ground_truth']}")
            print()
    
    # Save detailed comparison
    comparison_file = "gsm8k_comparison.json"
    comparison_data = {
        "summary": {
            "total_problems": total_problems,
            "jax_accuracy": jax_correct/total_problems*100,
            "pytorch_accuracy": pytorch_correct/total_problems*100,
            "agreement_rate": agreement_rate,
            "score_difference": score_diff,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "jax_only_correct": jax_only_correct,
            "pytorch_only_correct": pytorch_only_correct,
            "answer_mismatches": answer_mismatches
        },
        "mismatched_problems": mismatched_problems
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Detailed comparison saved to {comparison_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare GSM8K results between JAX and PyTorch")
    parser.add_argument("jax_results", help="JAX results JSONL file")
    parser.add_argument("pytorch_results", help="PyTorch results JSONL file")
    
    args = parser.parse_args()
    
    print("Loading results...")
    jax_results = load_results(args.jax_results)
    pytorch_results = load_results(args.pytorch_results)
    
    compare_results(jax_results, pytorch_results)

if __name__ == "__main__":
    main() 