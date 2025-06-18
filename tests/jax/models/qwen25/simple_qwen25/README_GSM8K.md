# GSM8K Parity Validation Framework

This directory contains a comprehensive framework for validating numerical parity between JAX and PyTorch implementations of Qwen2.5-7B on the GSM8K benchmark.

## Overview

The framework implements the single-goal playbook from your requirements:

> **"The JAX build must reach (±0.1%) the exact GSM8K accuracy that the official PyTorch Qwen-2-5-7B-Instruct build attains on the GSM8K test split."**

## Files

### Core Evaluation Scripts
- `gsm8k_jax.py` - JAX implementation for GSM8K evaluation
- `gsm8k_pytorch.py` - PyTorch implementation for GSM8K evaluation  
- `compare_gsm8k_results.py` - Results comparison and parity analysis
- `run_gsm8k_parity.py` - Master orchestrator script

### Supporting Files
- `simple_inference.py` - Base JAX model implementation (imported by gsm8k_jax.py)

## Key Features

### 1. Lock-Down Reference Implementation
- **Model version**: Fixed Qwen2.5-7B-Instruct weights
- **Prompt template**: Identical `AutoTokenizer.apply_chat_template()` usage
- **Generation params**: 
  - `temperature=0.0` (pure greedy)
  - `max_new_tokens=256`
  - `repetition_penalty=1.1` 
  - `do_sample=False` (PyTorch) / greedy argmax (JAX)
- **Answer extraction**: Standard GSM8K regex `r"####\s*(-?\d[\d,]*)"`

### 2. Deterministic JAX Implementation
- Forced deterministic ops: `jax.config.update('jax_deterministic_ops', True)`
- Single device execution: `--xla_force_host_platform_device_count=1`
- Deterministic PRNG: Fixed seed, no time-based randomness
- dtype support: float32 (for exact parity) and bfloat16 (for memory efficiency)

### 3. Evaluation Harness
- Memory-efficient dataset processing (100 problems per batch)
- Identical prompt formatting across frameworks
- Comprehensive error handling and logging
- Detailed per-problem result tracking

### 4. Parity Analysis
- Problem-level agreement analysis
- Score difference calculation
- Pass/fail status based on ±0.1% threshold
- Detailed mismatch investigation

## Usage

### Quick Start
```bash
# Run full parity validation
python run_gsm8k_parity.py --model_path ../weights

# Run with float32 for maximum precision
python run_gsm8k_parity.py --model_path ../weights --jax_dtype float32

# Run individual evaluations
python gsm8k_pytorch.py --model_path ../weights --output_path pytorch_results.jsonl
python gsm8k_jax.py --model_path ../weights --output_path jax_results.jsonl --dtype float32

# Compare existing results
python compare_gsm8k_results.py jax_results.jsonl pytorch_results.jsonl
```

### Expected Output Structure
```
gsm8k_results/20240124_143022/
├── pytorch_gsm8k.jsonl          # PyTorch predictions
├── jax_gsm8k.jsonl              # JAX predictions  
├── gsm8k_comparison.json        # Detailed parity analysis
```

## Requirements

### Dependencies
```bash
pip install datasets torch transformers jax flax safetensors psutil
```

### System Requirements
- 64GB+ RAM (for 7B model)
- CPU execution (optimized for memory constraints)
- ~2-4 hours runtime for full GSM8K test set (1,319 problems)

## Parity Thresholds

| Score Difference | Status | Description |
|-----------------|--------|-------------|
| ≤ 0.1% | ✅ PASS | Perfect parity achieved |
| ≤ 1.0% | ⚠️ CLOSE | Near parity, investigate |
| > 1.0% | ❌ FAIL | Significant divergence |

## Troubleshooting

### Common Issues

1. **Memory OOM with float32**
   - Use `--jax_dtype bfloat16` 
   - Chunked evaluation handles memory efficiently

2. **Different tokenizer behavior**
   - Both scripts use identical `apply_chat_template()` calls
   - Prompt format is logged for verification

3. **Generation parameter mismatches** 
   - JAX uses pure greedy argmax
   - PyTorch uses `do_sample=False`
   - Both achieve deterministic generation

4. **Answer extraction inconsistencies**
   - Identical regex pattern: `r"####\s*(-?\d[\d,]*)"`
   - Comma removal: `answer.replace(",", "")`

### Debugging Workflow

1. Check first 5 problem outputs in logs
2. Examine `gsm8k_comparison.json` for mismatch patterns
3. Compare prompt formatting between frameworks
4. Verify model weight loading consistency
5. Run with `--skip_pytorch` or `--skip_jax` to isolate issues

## Architecture

```
run_gsm8k_parity.py
├── gsm8k_pytorch.py
│   ├── Load HF transformers model
│   ├── Deterministic generation  
│   └── GSM8K evaluation loop
├── gsm8k_jax.py  
│   ├── Load custom JAX model (simple_inference.py)
│   ├── Deterministic generation
│   └── GSM8K evaluation loop
└── compare_gsm8k_results.py
    ├── Load both result files
    ├── Problem-by-problem comparison
    └── Parity analysis report
```

## Success Criteria

The framework achieves its goal when:
1. JAX accuracy ≥ (PyTorch accuracy - 0.1%)
2. No systematic biases in mismatch patterns  
3. Identical responses on majority of problems
4. Reproducible results across runs

This framework provides the foundation for achieving and maintaining numerical parity between JAX and PyTorch implementations on GSM8K. 