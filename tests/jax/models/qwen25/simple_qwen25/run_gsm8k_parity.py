#!/usr/bin/env python3
"""
Master script for GSM8K parity validation between JAX and PyTorch Qwen2.5-7B.

This script:
1. Runs PyTorch evaluation on GSM8K
2. Runs JAX evaluation on GSM8K  
3. Compares results and reports parity status
4. Provides detailed analysis of any discrepancies

Usage:
python run_gsm8k_parity.py --model_path ../weights [--quick_test]
"""
import os
import sys
import time
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gsm8k_parity")

def run_command(cmd, description):
    """Run a command and log results"""
    logger.info(f"🔄 {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        logger.info(f"✅ {description} completed in {duration:.1f}s")
    else:
        logger.error(f"❌ {description} failed!")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"{description} failed")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="GSM8K parity validation between JAX and PyTorch")
    parser.add_argument("--model_path", required=True, help="Path to Qwen2.5-7B model weights")
    parser.add_argument("--quick_test", action="store_true", help="Run on small subset for testing")
    parser.add_argument("--jax_dtype", default="float32", choices=["float32", "bfloat16"], 
                       help="JAX model dtype")
    parser.add_argument("--skip_pytorch", action="store_true", help="Skip PyTorch evaluation")
    parser.add_argument("--skip_jax", action="store_true", help="Skip JAX evaluation")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)
    
    config_path = model_path / "config.json"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info("🚀 Starting GSM8K Parity Validation")
    logger.info(f"Model path: {model_path.absolute()}")
    logger.info(f"JAX dtype: {args.jax_dtype}")
    logger.info(f"Quick test: {args.quick_test}")
    
    # Determine output files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("gsm8k_results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pytorch_output = output_dir / "pytorch_gsm8k.jsonl"
    jax_output = output_dir / "jax_gsm8k.jsonl"
    
    script_dir = Path(__file__).parent
    
    # Step 1: Run PyTorch evaluation
    if not args.skip_pytorch:
        pytorch_script = script_dir / "gsm8k_pytorch.py"
        if not pytorch_script.exists():
            logger.error(f"PyTorch script not found: {pytorch_script}")
            sys.exit(1)
        
        pytorch_cmd = [
            sys.executable, str(pytorch_script),
            "--model_path", str(model_path),
            "--output_path", str(pytorch_output)
        ]
        
        run_command(pytorch_cmd, "PyTorch GSM8K evaluation")
    else:
        logger.info("⏭️  Skipping PyTorch evaluation")
    
    # Step 2: Run JAX evaluation
    if not args.skip_jax:
        jax_script = script_dir / "gsm8k_jax.py"
        if not jax_script.exists():
            logger.error(f"JAX script not found: {jax_script}")
            sys.exit(1)
        
        jax_cmd = [
            sys.executable, str(jax_script),
            "--model_path", str(model_path),
            "--output_path", str(jax_output),
            "--dtype", args.jax_dtype
        ]
        
        run_command(jax_cmd, "JAX GSM8K evaluation")
    else:
        logger.info("⏭️  Skipping JAX evaluation")
    
    # Step 3: Compare results
    if not args.skip_pytorch and not args.skip_jax:
        if pytorch_output.exists() and jax_output.exists():
            compare_script = script_dir / "compare_gsm8k_results.py"
            if not compare_script.exists():
                logger.error(f"Comparison script not found: {compare_script}")
                sys.exit(1)
            
            compare_cmd = [
                sys.executable, str(compare_script),
                str(jax_output), str(pytorch_output)
            ]
            
            # Change to output directory for comparison files
            original_cwd = os.getcwd()
            os.chdir(output_dir)
            
            try:
                run_command(compare_cmd, "Comparing JAX vs PyTorch results")
            finally:
                os.chdir(original_cwd)
            
            logger.info(f"📊 Results saved to: {output_dir.absolute()}")
        else:
            logger.warning("Cannot compare - missing result files")
    else:
        logger.info("⏭️  Skipping comparison (missing evaluations)")
    
    # Step 4: Summary
    logger.info("✨ GSM8K Parity Validation Complete!")
    logger.info(f"📁 Check results in: {output_dir.absolute()}")
    
    if output_dir.exists():
        comparison_file = output_dir / "gsm8k_comparison.json"
        if comparison_file.exists():
            logger.info(f"📋 Detailed comparison: {comparison_file}")
            
            # Quick summary
            import json
            with open(comparison_file, 'r') as f:
                comparison = json.load(f)
            
            summary = comparison["summary"]
            logger.info(f"🎯 JAX Accuracy: {summary['jax_accuracy']:.2f}%")
            logger.info(f"🎯 PyTorch Accuracy: {summary['pytorch_accuracy']:.2f}%")
            logger.info(f"📏 Score Difference: {summary['score_difference']:.2f}%")
            
            if summary['score_difference'] <= 0.1:
                logger.info("✅ PARITY: PASS (≤0.1% difference)")
            elif summary['score_difference'] <= 1.0:
                logger.info("⚠️  PARITY: CLOSE (≤1.0% difference)")
            else:
                logger.info("❌ PARITY: FAIL (>1.0% difference)")

if __name__ == "__main__":
    main() 