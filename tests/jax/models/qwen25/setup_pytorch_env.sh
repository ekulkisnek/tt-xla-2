#!/bin/bash
# PyTorch environment setup for memory-efficient CPU inference
export TORCH_CPU_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_MPS_CACHE_DISABLE=1
export OMP_NUM_THREADS=8

echo "PyTorch environment configured:"
echo "  TORCH_CPU_ALLOC_CONF=${TORCH_CPU_ALLOC_CONF}"
echo "  PYTORCH_MPS_CACHE_DISABLE=${PYTORCH_MPS_CACHE_DISABLE}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}" 