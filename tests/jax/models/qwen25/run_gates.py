#!/usr/bin/env python3
"""
Run the five-gate parity ladder systematically.
Master orchestration script for proving JAX/PyTorch equivalence.
"""
import os
import sys
import subprocess
import argparse
import tempfile
import shutil
import time

def run_command(cmd, env_vars=None, cwd=None):
    """Run a command with optional environment variables and working directory."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    print(f"Running: {cmd}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    result = subprocess.run(cmd, shell=True, env=env, cwd=cwd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    
    print(f"✅ Command completed successfully")
    return True

def gate_0(dtype="float32"):
    """G-0: Weight loading validation."""
    print(f"\n{'='*80}")
    print(f"GATE 0: Weight Loading Validation")
    print(f"{'='*80}")
    print("This gate validates that weights are loaded correctly (not random initialization).")
    
    # Set JAX environment
    jax_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_NUM_GENEROUS_MEMORY_BLOCKS": "0"
    }
    
    # Just run a quick inference to trigger weight loading validation
    cmd = f"python simple_inference.py --model_path ../weights --prompt 'test' --max_tokens 1 --dtype {dtype}"
    return run_command(cmd, jax_env, "jax_scripts")

def gate_1():
    """G-1: Single token logits comparison."""
    print(f"\n{'='*80}")
    print(f"GATE 1: Single Token Logits")
    print(f"{'='*80}")
    print("Testing basic forward pass with single token.")
    print("Pass threshold: max Δ < 1e-5 (float32)")
    
    # PyTorch: dump reference
    print("\n--- Phase 1: Generate PyTorch reference ---")
    pt_env = {
        "TORCH_CPU_ALLOC_CONF": "max_split_size_mb:128",
        "PYTORCH_MPS_CACHE_DISABLE": "1"
    }
    
    cmd = "python dump_logits.py --prompt_ids '1' --out ../tmp/pt_g1.npy --dtype float32"
    if not run_command(cmd, pt_env, "torch_scripts"):
        return False
    
    time.sleep(2)  # Brief pause for cleanup
    
    # JAX: compare
    print("\n--- Phase 2: Compare with JAX ---")
    jax_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_NUM_GENEROUS_MEMORY_BLOCKS": "0"
    }
    
    cmd = "python cmp_logits.py --ids '1' --ref ../tmp/pt_g1.npy --dtype float32"
    return run_command(cmd, jax_env, "jax_scripts")

def gate_2():
    """G-2: 16-token logits comparison."""
    print(f"\n{'='*80}")
    print(f"GATE 2: 16-Token Logits (RoPE & MLP)")
    print(f"{'='*80}")
    print("Testing longer sequences with RoPE and MLP layers.")
    print("Pass threshold: max Δ < 1e-4 (float32)")
    
    # Use a 16-token sequence
    ids = "1 42 600 17 5 8 9 2 11 15 20 25 30 35 40 45"
    
    # PyTorch: dump reference
    print("\n--- Phase 1: Generate PyTorch reference ---")
    pt_env = {
        "TORCH_CPU_ALLOC_CONF": "max_split_size_mb:128",
        "PYTORCH_MPS_CACHE_DISABLE": "1"
    }
    
    cmd = f"python dump_logits.py --prompt_ids '{ids}' --out ../tmp/pt_g2.npy --dtype float32"
    if not run_command(cmd, pt_env, "torch_scripts"):
        return False
    
    time.sleep(2)  # Brief pause for cleanup
    
    # JAX: compare
    print("\n--- Phase 2: Compare with JAX ---")
    jax_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_NUM_GENEROUS_MEMORY_BLOCKS": "0"
    }
    
    cmd = f"python cmp_logits.py --ids '{ids}' --ref ../tmp/pt_g2.npy --dtype float32"
    return run_command(cmd, jax_env, "jax_scripts")

def gate_3():
    """G-3: Cached step validation."""
    print(f"\n{'='*80}")
    print(f"GATE 3: Cached Step (KV + Mask)")
    print(f"{'='*80}")
    print("Testing KV cache functionality and causal masking.")
    print("Pass threshold: max Δ < 1e-4 (float32)")
    
    # PyTorch: dump cached reference
    print("\n--- Phase 1: Generate PyTorch cached reference ---")
    pt_env = {
        "TORCH_CPU_ALLOC_CONF": "max_split_size_mb:128",
        "PYTORCH_MPS_CACHE_DISABLE": "1"
    }
    
    cmd = "python dump_logits.py --prompt_ids '1 42 600' --next_id 9 --out ../tmp/pt_g3.npy --dtype float32"
    if not run_command(cmd, pt_env, "torch_scripts"):
        return False
    
    time.sleep(2)  # Brief pause for cleanup
    
    # JAX: compare with cache
    print("\n--- Phase 2: Compare with JAX cached ---")
    jax_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_NUM_GENEROUS_MEMORY_BLOCKS": "0"
    }
    
    cmd = "python cmp_logits.py --ids '1 42 600 9' --ref ../tmp/pt_g3.npy --use_cache --dtype float32"
    return run_command(cmd, jax_env, "jax_scripts")

def gate_4():
    """G-4: Greedy generation comparison."""
    print(f"\n{'='*80}")
    print(f"GATE 4: Greedy 64-Token Generation")
    print(f"{'='*80}")
    print("Testing end-to-end text generation with greedy decoding.")
    print("Pass threshold: Byte-identical text output")
    
    prompt = "The meaning of life"
    
    # PyTorch: generate reference
    print("\n--- Phase 1: Generate PyTorch reference text ---")
    pt_env = {
        "TORCH_CPU_ALLOC_CONF": "max_split_size_mb:128",
        "PYTORCH_MPS_CACHE_DISABLE": "1"
    }
    
    cmd = f"python run_greedy.py --prompt '{prompt}' --max_tokens 64 --out ../tmp/pt_g4.txt --dtype float32"
    if not run_command(cmd, pt_env, "torch_scripts"):
        return False
    
    time.sleep(2)  # Brief pause for cleanup
    
    # JAX: generate and save to file
    print("\n--- Phase 2: Generate JAX text ---")
    jax_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_NUM_GENEROUS_MEMORY_BLOCKS": "0"
    }
    
    # Create a custom script to capture JAX output to file
    jax_gen_script = f"""
import sys
sys.path.append('.')
import simple_inference as si
import subprocess
import os

# Redirect stdout to file
with open('../tmp/jax_g4.txt', 'w') as f:
    result = subprocess.run([
        'python', 'simple_inference.py', 
        '--model_path', '../weights',
        '--prompt', '{prompt}',
        '--temperature', '0.0',
        '--top_p', '0.0', 
        '--top_k', '0',
        '--max_tokens', '64',
        '--dtype', 'float32'
    ], capture_output=True, text=True, env=os.environ.copy())
    
    f.write(result.stdout)
    print("JAX generation completed")
    print("STDOUT:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
"""
    
    # Write and run the JAX generation script
    with open("jax_scripts/temp_gen.py", "w") as f:
        f.write(jax_gen_script)
    
    if not run_command("python temp_gen.py", jax_env, "jax_scripts"):
        return False
    
    # Clean up temp script
    try:
        os.remove("jax_scripts/temp_gen.py")
    except:
        pass
    
    # Compare files
    print("\n--- Phase 3: Compare generated texts ---")
    print("Comparing PyTorch and JAX generated texts...")
    
    try:
        with open("tmp/pt_g4.txt", "r") as f:
            pt_text = f.read().strip()
        with open("tmp/jax_g4.txt", "r") as f:
            jax_text = f.read().strip()
        
        print(f"PyTorch text ({len(pt_text)} chars):")
        print("-" * 40)
        print(pt_text)
        print("-" * 40)
        
        print(f"JAX text ({len(jax_text)} chars):")
        print("-" * 40)
        print(jax_text)
        print("-" * 40)
        
        if pt_text == jax_text:
            print("✅ TEXTS ARE IDENTICAL")
            return True
        else:
            print("❌ TEXTS ARE DIFFERENT")
            # Show character-by-character diff for first difference
            for i, (c1, c2) in enumerate(zip(pt_text, jax_text)):
                if c1 != c2:
                    print(f"First difference at position {i}: PT='{c1}' vs JAX='{c2}'")
                    break
            return False
    
    except Exception as e:
        print(f"Error comparing files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run five-gate parity ladder")
    parser.add_argument("--gate", type=int, choices=[0,1,2,3,4], help="Run specific gate only")
    parser.add_argument("--continue_on_fail", action="store_true", help="Continue even if a gate fails")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Model dtype")
    args = parser.parse_args()
    
    # Ensure we're in the right directory and tmp exists
    if not os.path.exists("weights"):
        print("ERROR: weights directory not found. Run this script from the qwen25 directory.")
        sys.exit(1)
    
    os.makedirs("tmp", exist_ok=True)
    
    # Verify scripts exist
    required_files = [
        "torch_scripts/dump_logits.py",
        "torch_scripts/run_greedy.py", 
        "jax_scripts/cmp_logits.py",
        "jax_scripts/simple_inference.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Required file {file} not found")
            sys.exit(1)
    
    gates = [gate_0, gate_1, gate_2, gate_3, gate_4]
    gate_names = ["G-0 Weight Loading", "G-1 Single Token", "G-2 Multi Token", "G-3 Cached", "G-4 Generation"]
    
    print(f"{'='*80}")
    print(f"QWEN2.5 FIVE-GATE PARITY LADDER")
    print(f"{'='*80}")
    print(f"Model dtype: {args.dtype}")
    print(f"Continue on failure: {args.continue_on_fail}")
    
    if args.gate is not None:
        # Run specific gate
        print(f"Running specific gate: {gate_names[args.gate]}")
        if args.gate == 0:
            success = gates[args.gate](args.dtype)
        else:
            success = gates[args.gate]()
        result_emoji = "✅" if success else "❌"
        print(f"\n{result_emoji} {gate_names[args.gate]}: {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    
    # Run all gates in sequence
    results = []
    for i, gate_func in enumerate(gates):
        print(f"\n{'='*80}")
        print(f"STARTING GATE {i}: {gate_names[i]}")
        print(f"{'='*80}")
        
        if i == 0:
            success = gate_func(args.dtype)
        else:
            success = gate_func()
        results.append(success)
        
        result_emoji = "✅" if success else "❌"
        print(f"\n{result_emoji} GATE {i} ({gate_names[i]}): {'PASSED' if success else 'FAILED'}")
        
        if not success:
            print(f"\n❌ GATE {i} FAILED!")
            if not args.continue_on_fail:
                print("Stopping at first failure. Fix this gate before proceeding.")
                print("\nRECOMMENDED NEXT STEPS:")
                if i == 0:
                    print("- Check weight loading in simple_inference.py")
                    print("- Verify embedding std is different from initialization")
                elif i == 1:
                    print("- Compare single token forward pass step by step")
                    print("- Check embedding, first layer, RMSNorm")
                elif i == 2:
                    print("- Check RoPE implementation")
                    print("- Verify MLP and attention scaling")
                elif i == 3:
                    print("- Debug KV cache handling")
                    print("- Check causal mask logic")
                elif i == 4:
                    print("- Verify greedy sampling (temperature=0)")
                    print("- Check tokenizer consistency")
                sys.exit(1)
            else:
                print("Continuing to next gate...")
        else:
            print(f"✅ GATE {i} PASSED!")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    
    total_passed = sum(results)
    for i, (result, name) in enumerate(zip(results, gate_names)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"Gate {i} ({name}): {status}")
    
    print(f"\nOverall: {total_passed}/{len(results)} gates passed")
    
    if total_passed == len(results):
        print(f"\n🎉 ALL GATES PASSED! JAX MODEL IS PROVABLY EQUIVALENT TO PYTORCH!")
        print(f"Next steps:")
        print(f"- Switch to bfloat16 for production use")
        print(f"- Run qualitative evaluation on various prompts") 
        print(f"- Set up CI/CD with these parity tests")
    else:
        print(f"\n❌ {len(results) - total_passed} gate(s) failed. Fix failing gates for full parity.")
    
    sys.exit(0 if total_passed == len(results) else 1)

if __name__ == "__main__":
    main() 