#!/usr/bin/env python3
"""
Stage 5: Multi-Mesh Benchmarking Script
Measures tokens/s and scaling performance across different mesh configurations
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import os
import sys

# Add parent directory to path to import stage3
sys.path.append('..')
from stage3 import (
    create_mesh, TensorParallelDense, QwenAttention, QwenMLP, 
    QwenDecoderLayer, Qwen25ForCausalLM
)

class PerformanceBenchmark:
    """Benchmark class for measuring TP scaling performance"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def benchmark_mesh(self, mesh_shape, num_tokens=50, warmup_tokens=5):
        """Benchmark a specific mesh configuration"""
        model_parallel, data_parallel = mesh_shape
        total_devices = model_parallel * data_parallel
        
        print(f"\n=== Benchmarking Mesh {model_parallel}×{data_parallel} ({total_devices} devices) ===")
        
        # Set up virtual devices if needed
        if len(jax.devices()) < total_devices:
            print(f"Creating {total_devices} virtual devices...")
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={total_devices}"
            jax.config.update('jax_platform_name', 'cpu')
        
        try:
            if total_devices == 1:
                # Single device benchmark
                return self._benchmark_single_device(num_tokens, warmup_tokens)
            else:
                # Multi-device benchmark
                mesh = create_mesh(model_parallel=model_parallel, data_parallel=data_parallel)
                return self._benchmark_with_mesh(mesh, num_tokens, warmup_tokens)
                
        except Exception as e:
            print(f"❌ Benchmark failed for mesh {mesh_shape}: {e}")
            return None
    
    def _benchmark_single_device(self, num_tokens, warmup_tokens):
        """Benchmark single device performance"""
        print("Setting up single device model...")
        model = Qwen25ForCausalLM(config=self.config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        
        # Initialize
        input_ids = jnp.array([[1, 2]])
        params = model.init(key, input_ids)
        
        # Warmup
        print(f"Warmup: {warmup_tokens} tokens...")
        current_ids = input_ids
        past_kv = None
        
        for i in range(warmup_tokens):
            if i == 0:
                outputs = model.apply(params, current_ids, return_dict=True)
            else:
                outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
            
            logits = outputs["logits"]
            past_kv = outputs["past_key_values"]
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            current_ids = next_token
        
        # Actual benchmark
        print(f"Benchmarking: {num_tokens} tokens...")
        start_time = time.time()
        
        for i in range(num_tokens):
            outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
            logits = outputs["logits"]
            past_kv = outputs["past_key_values"]
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            current_ids = next_token
        
        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_sec = num_tokens / total_time
        
        result = {
            'mesh_shape': (1, 1),
            'total_devices': 1,
            'tokens': num_tokens,
            'time': total_time,
            'tokens_per_sec': tokens_per_sec,
            'relative_speedup': 1.0
        }
        
        print(f"✅ Single device: {tokens_per_sec:.2f} tokens/s")
        return result
    
    def _benchmark_with_mesh(self, mesh, num_tokens, warmup_tokens):
        """Benchmark with tensor parallel mesh"""
        mesh_shape = mesh.devices.shape
        total_devices = mesh.size
        
        print(f"Setting up TP model with mesh {mesh_shape}...")
        
        with mesh:
            model = Qwen25ForCausalLM(config=self.config, dtype=jnp.float32)
            key = jax.random.PRNGKey(42)
            
            # Initialize
            input_ids = jnp.array([[1, 2]])
            params = model.init(key, input_ids)
            
            # Warmup
            print(f"Warmup: {warmup_tokens} tokens...")
            current_ids = input_ids
            past_kv = None
            
            for i in range(warmup_tokens):
                if i == 0:
                    outputs = model.apply(params, current_ids, return_dict=True)
                else:
                    outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                
                logits = outputs["logits"]
                past_kv = outputs["past_key_values"]
                next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                current_ids = next_token
            
            # Actual benchmark
            print(f"Benchmarking: {num_tokens} tokens...")
            start_time = time.time()
            
            for i in range(num_tokens):
                outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
                logits = outputs["logits"]
                past_kv = outputs["past_key_values"]
                next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                current_ids = next_token
            
            end_time = time.time()
            total_time = end_time - start_time
            tokens_per_sec = num_tokens / total_time
            
            result = {
                'mesh_shape': mesh_shape,
                'total_devices': total_devices,
                'tokens': num_tokens,
                'time': total_time,
                'tokens_per_sec': tokens_per_sec,
                'relative_speedup': None  # Will be calculated later
            }
            
            print(f"✅ TP {mesh_shape}: {tokens_per_sec:.2f} tokens/s")
            return result
    
    def run_scaling_benchmark(self):
        """Run complete scaling benchmark across multiple mesh configurations"""
        print("Stage 5: Multi-Mesh Scaling Benchmark")
        print("=" * 50)
        
        # Define mesh configurations to test
        mesh_configs = [
            (1, 1),   # Single device baseline
            (2, 1),   # 1×2 TP
            (4, 1),   # 1×4 TP  
            (8, 1),   # 1×8 TP
        ]
        
        # If we have enough virtual devices, also test data parallel
        if True:  # For virtual device testing
            mesh_configs.extend([
                (2, 2),   # 2×2 TP+DP
                (4, 2),   # 2×4 DP+TP
            ])
        
        results = []
        baseline_tokens_per_sec = None
        
        for mesh_shape in mesh_configs:
            try:
                result = self.benchmark_mesh(mesh_shape, num_tokens=20, warmup_tokens=3)
                if result:
                    # Calculate relative speedup
                    if baseline_tokens_per_sec is None:
                        baseline_tokens_per_sec = result['tokens_per_sec']
                        result['relative_speedup'] = 1.0
                    else:
                        result['relative_speedup'] = result['tokens_per_sec'] / baseline_tokens_per_sec
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"❌ Failed to benchmark mesh {mesh_shape}: {e}")
                continue
        
        # Print summary
        self._print_scaling_summary(results)
        return results
    
    def _print_scaling_summary(self, results):
        """Print scaling benchmark summary"""
        print("\n" + "=" * 60)
        print("SCALING BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"{'Mesh':<10} {'Devices':<8} {'Tokens/s':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("-" * 60)
        
        for result in results:
            mesh_str = f"{result['mesh_shape'][0]}×{result['mesh_shape'][1]}"
            devices = result['total_devices']
            tokens_per_sec = result['tokens_per_sec']
            speedup = result['relative_speedup']
            efficiency = speedup / devices if devices > 1 else 1.0
            
            print(f"{mesh_str:<10} {devices:<8} {tokens_per_sec:<12.2f} {speedup:<10.2f} {efficiency:<12.2f}")
        
        # Analyze results
        print("\n" + "=" * 60)
        print("ANALYSIS:")
        
        # Check for expected scaling
        for result in results:
            devices = result['total_devices']
            speedup = result['relative_speedup']
            efficiency = speedup / devices if devices > 1 else 1.0
            
            if devices == 1:
                print(f"✅ Baseline: {result['tokens_per_sec']:.2f} tokens/s")
            elif devices <= 4:
                expected_min = 0.8 * devices  # 80% efficiency minimum
                if speedup >= expected_min:
                    print(f"✅ {result['mesh_shape'][0]}×{result['mesh_shape'][1]}: Good scaling ({efficiency:.1%} efficiency)")
                else:
                    print(f"⚠️ {result['mesh_shape'][0]}×{result['mesh_shape'][1]}: Below expected scaling ({efficiency:.1%} efficiency)")
            else:
                expected_min = 0.7 * devices  # 70% efficiency for larger meshes
                if speedup >= expected_min:
                    print(f"✅ {result['mesh_shape'][0]}×{result['mesh_shape'][1]}: Acceptable scaling ({efficiency:.1%} efficiency)")
                else:
                    print(f"⚠️ {result['mesh_shape'][0]}×{result['mesh_shape'][1]}: Poor scaling ({efficiency:.1%} efficiency)")

def main():
    """Main benchmarking function"""
    # Use smaller config for faster benchmarking
    config = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "vocab_size": 100,
        "rms_norm_eps": 1e-5
    }
    
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_scaling_benchmark()
    
    # Check Stage 5 gates
    print("\n" + "=" * 60)
    print("STAGE 5 GATE VERIFICATION:")
    
    # Find relevant results
    baseline = next((r for r in results if r['total_devices'] == 1), None)
    tp_4 = next((r for r in results if r['mesh_shape'] == (4, 1)), None)
    tp_8 = next((r for r in results if r['mesh_shape'] == (8, 1)), None)
    
    gates_passed = 0
    total_gates = 0
    
    if baseline and tp_4:
        total_gates += 1
        efficiency_4 = tp_4['relative_speedup'] / 4
        if efficiency_4 >= 0.7:  # 70% efficiency threshold
            print(f"✅ Gate 5-A: 1×4 mesh shows good scaling ({efficiency_4:.1%} efficiency)")
            gates_passed += 1
        else:
            print(f"❌ Gate 5-A: 1×4 mesh poor scaling ({efficiency_4:.1%} efficiency)")
    
    if baseline and tp_8:
        total_gates += 1
        efficiency_8 = tp_8['relative_speedup'] / 8
        if efficiency_8 >= 0.6:  # 60% efficiency threshold for 8 devices
            print(f"✅ Gate 5-B: 1×8 mesh shows acceptable scaling ({efficiency_8:.1%} efficiency)")
            gates_passed += 1
        else:
            print(f"❌ Gate 5-B: 1×8 mesh poor scaling ({efficiency_8:.1%} efficiency)")
    
    # Overall Stage 5 result
    if gates_passed == total_gates and total_gates > 0:
        print("\n🎉 Stage 5 Complete - Multi-mesh bring-up successful!")
        print("Ready for Stage 6: Performance hardening!")
    else:
        print(f"\n⚠️ Stage 5 partial: {gates_passed}/{total_gates} gates passed")
        print("Consider tuning or investigating scaling issues")

if __name__ == "__main__":
    main() 