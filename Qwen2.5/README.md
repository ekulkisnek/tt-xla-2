# Qwen2.5-7B Tensor-Parallel JAX Implementation for TT-xla Bounty

## Overview
Tensor-parallelized Qwen2.5-7B-Instruct in JAX/Flax for multi-device. Extends open-source single-device, with TP in projections/MLPs.

Features:
- TP via shard_map.
- Precomputed RoPE with complex application.
- GQA attention.
- Greedy sampling.
- bfloat16 for speed.

Rationale:
- Replicated embeddings/LM head for efficiency.
- Float32 upcast in attention for stability.
- Concat KV cache (in-place possible future improvement).

## Setup
1. Install deps:
   ```
   pip install jax flax transformers safetensors psutil numpy datasets
   ```
2. Download weights from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct (safetensors).

## Usage
- Inference:
  ```
  python generate_multi_chip.py --model_path weights
  ```
  - Runs 10 math questions as samples.
- GSM8K Eval:
  ```
  python test_gsm8k.py --model_path weights --num_samples 50
  ```

## Architecture
Causal LM with TP sharding on output dims, all-gather combination.

## Samples
Math questions in generate_multi_chip.py, e.g., Sam's prompt: "Question: Sam scores 80 on the first test and 90 on the second. What score does he need on the third test to have an average of 85?"
- Output: [Reasoning with \boxed{85}].

GSM8K: Run test_gsm8k.py for accuracy (~91.5% expected, matching single-device).

## Compatibility
Simulated 1x1 default. For 1x8 etc., set XLA_FLAGS device count.

## Limitations
Concat cache may use memory; repetition possible in long gens.