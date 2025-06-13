#!/usr/bin/env python3
"""
Comprehensive PyTorch vs JAX Qwen2.5 Parity Validation Script
Implements the 6-stage validation ladder for perfect parity verification.
"""

import os
import sys
import json
import hashlib
import time
import gc
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jax
import jax.numpy as jnp
from simple_inference import Qwen25ForCausalLM, load_params

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ParityValidator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = {}
        
        print("="*80)
        print("QWEN2.5 PYTORCH vs JAX PARITY VALIDATION")
        print("="*80)
        
        # Load configuration
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Define test inputs
        self.test_ids_1 = np.array([[1]], dtype=np.int32)  # Single token
        self.test_ids_16 = np.array([[1,42,600,17,5,8,9,2,13,21,55,77,88,99,100,101]], dtype=np.int32)  # 16 tokens
        
        print(f"Test inputs prepared:")
        print(f"  Single token: {self.test_ids_1.tolist()}")
        print(f"  16 tokens: {self.test_ids_16.tolist()}")
    
    def sha256_hash(self, arr: np.ndarray) -> str:
        """Compute SHA-256 hash of numpy array."""
        h = hashlib.sha256()
        h.update(arr.tobytes())
        return h.hexdigest()
    
    def load_pytorch_model(self) -> torch.nn.Module:
        """Load PyTorch model in float32."""
        print("\nLoading PyTorch model (float32)...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        print("✓ PyTorch model loaded")
        return model
    
    def load_jax_model(self) -> Tuple[Any, Dict]:
        """Load JAX model in float32."""
        print("\nLoading JAX model (float32)...")
        jax_model = Qwen25ForCausalLM(config=self.config, dtype=jnp.float32)
        jax_params = load_params(jax_model, self.model_path, jnp.float32)
        print("✓ JAX model loaded")
        return jax_model, jax_params
    
    def stage_1_weight_hashing(self, pt_model: torch.nn.Module, jax_model: Any, jax_params: Dict) -> bool:
        """Stage 1: Hash every weight tensor for bit-perfect comparison."""
        print("\n" + "="*60)
        print("STAGE 1: WEIGHT HASHING VALIDATION")
        print("="*60)
        
        pt_state = pt_model.state_dict()
        mismatches = []
        matches = []
        skipped = []
        
        # Key mapping for comparison
        key_mappings = {
            # Direct mappings
            "model.embed_tokens.weight": ("embed_tokens", "embedding"),
            "model.norm.weight": ("norm", "scale"), 
            "lm_head.weight": ("lm_head", "kernel"),
        }
        
        # Add layer mappings
        for i in range(self.config["num_hidden_layers"]):
            layer_prefix = f"model.layers.{i}"
            jax_prefix = f"layers_{i}"
            
            # Layer norms
            key_mappings[f"{layer_prefix}.input_layernorm.weight"] = (jax_prefix, "input_layernorm", "scale")
            key_mappings[f"{layer_prefix}.post_attention_layernorm.weight"] = (jax_prefix, "post_attention_layernorm", "scale")
            
            # Attention projections
            for proj in ["q", "k", "v", "o"]:
                key_mappings[f"{layer_prefix}.self_attn.{proj}_proj.weight"] = (jax_prefix, "self_attn", f"{proj}_proj", "kernel")
                if f"{layer_prefix}.self_attn.{proj}_proj.bias" in pt_state:
                    key_mappings[f"{layer_prefix}.self_attn.{proj}_proj.bias"] = (jax_prefix, "self_attn", f"{proj}_proj", "bias")
            
            # MLP projections
            for proj in ["gate", "up", "down"]:
                key_mappings[f"{layer_prefix}.mlp.{proj}_proj.weight"] = (jax_prefix, "mlp", f"{proj}_proj", "kernel")
        
        print(f"Comparing {len(key_mappings)} weight tensors...")
        
        for pt_key, jax_path in key_mappings.items():
            if pt_key not in pt_state:
                skipped.append(pt_key)
                continue
                
            # Get PyTorch tensor
            pt_tensor = pt_state[pt_key].cpu().numpy()
            
            # Get JAX tensor
            try:
                jax_tensor = jax_params["params"]
                for path_part in jax_path:
                    jax_tensor = jax_tensor[path_part]
                jax_tensor = np.array(jax_tensor)
            except KeyError:
                print(f"⚠️  JAX tensor not found for {pt_key} -> {jax_path}")
                mismatches.append(pt_key)
                continue
            
            # Handle transpose for weight matrices
            if "weight" in pt_key and ("proj" in pt_key or "lm_head" in pt_key):
                if "layernorm" not in pt_key and "norm.weight" not in pt_key:
                    pt_tensor = pt_tensor.T  # Transpose PyTorch to match JAX convention
            
            # Compare shapes
            if pt_tensor.shape != jax_tensor.shape:
                print(f"❌ Shape mismatch {pt_key}: PT={pt_tensor.shape} vs JAX={jax_tensor.shape}")
                mismatches.append(pt_key)
                continue
            
            # Hash comparison
            pt_hash = self.sha256_hash(pt_tensor)
            jax_hash = self.sha256_hash(jax_tensor)
            
            if pt_hash == jax_hash:
                matches.append(pt_key)
                print(f"✓ {pt_key}")
            else:
                max_diff = np.max(np.abs(pt_tensor - jax_tensor))
                print(f"❌ {pt_key}: hash mismatch (max_diff={max_diff:.2e})")
                mismatches.append(pt_key)
        
        print(f"\nStage 1 Results:")
        print(f"  ✓ Matches: {len(matches)}")
        print(f"  ❌ Mismatches: {len(mismatches)}")
        print(f"  ⚠️  Skipped: {len(skipped)}")
        
        if skipped:
            print(f"  Skipped keys: {skipped}")
        
        success = len(mismatches) == 0
        self.results["stage_1"] = {
            "success": success,
            "matches": len(matches),
            "mismatches": len(mismatches),
            "skipped": len(skipped)
        }
        
        return success
    
    def stage_2_single_token(self, pt_model: torch.nn.Module, jax_model: Any, jax_params: Dict) -> bool:
        """Stage 2: Single token forward pass comparison."""
        print("\n" + "="*60)
        print("STAGE 2: SINGLE TOKEN FORWARD PASS")
        print("="*60)
        
        # PyTorch forward pass
        with torch.no_grad():
            pt_logits = pt_model(torch.tensor(self.test_ids_1)).logits.cpu().numpy()
        
        # JAX forward pass
        jax_out = jax_model.apply(
            jax_params,
            input_ids=jnp.array(self.test_ids_1),
            return_dict=True
        )
        jax_logits = np.array(jax_out["logits"])
        
        # Compare
        max_diff = np.max(np.abs(pt_logits - jax_logits))
        mean_diff = np.mean(np.abs(pt_logits - jax_logits))
        
        print(f"Logits shape: PT={pt_logits.shape}, JAX={jax_logits.shape}")
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        
        # Success criterion
        success = max_diff < 1e-6
        
        if success:
            print("✓ Stage 2 PASSED: Single token forward pass matches perfectly!")
        else:
            print("❌ Stage 2 FAILED: Significant difference in logits")
            
            # Additional debugging
            print(f"PT logits range: [{np.min(pt_logits):.3f}, {np.max(pt_logits):.3f}]")
            print(f"JAX logits range: [{np.min(jax_logits):.3f}, {np.max(jax_logits):.3f}]")
            
            # Show first few logits for debugging
            print(f"PT first 5 logits: {pt_logits[0,0,:5]}")
            print(f"JAX first 5 logits: {jax_logits[0,0,:5]}")
        
        self.results["stage_2"] = {
            "success": success,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff)
        }
        
        return success
    
    def stage_3_rope_test(self, pt_model: torch.nn.Module, jax_model: Any, jax_params: Dict) -> bool:
        """Stage 3: 16-token forward pass (RoPE test)."""
        print("\n" + "="*60)
        print("STAGE 3: 16-TOKEN FORWARD PASS (RoPE TEST)")
        print("="*60)
        
        # PyTorch forward pass
        with torch.no_grad():
            pt_logits = pt_model(torch.tensor(self.test_ids_16)).logits.cpu().numpy()
        
        # JAX forward pass
        jax_out = jax_model.apply(
            jax_params,
            input_ids=jnp.array(self.test_ids_16),
            return_dict=True
        )
        jax_logits = np.array(jax_out["logits"])
        
        # Compare
        max_diff = np.max(np.abs(pt_logits - jax_logits))
        mean_diff = np.mean(np.abs(pt_logits - jax_logits))
        
        print(f"Logits shape: PT={pt_logits.shape}, JAX={jax_logits.shape}")
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        
        # Success criterion (slightly looser for longer sequences)
        success = max_diff < 1e-4
        
        if success:
            print("✓ Stage 3 PASSED: 16-token forward pass matches!")
        else:
            print("❌ Stage 3 FAILED: Significant difference in 16-token logits")
            
            # Additional debugging
            print(f"PT logits range: [{np.min(pt_logits):.3f}, {np.max(pt_logits):.3f}]")
            print(f"JAX logits range: [{np.min(jax_logits):.3f}, {np.max(jax_logits):.3f}]")
        
        self.results["stage_3"] = {
            "success": success,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff)
        }
        
        return success
    
    def stage_4_kv_cache(self, pt_model: torch.nn.Module, jax_model: Any, jax_params: Dict) -> bool:
        """Stage 4: KV cache validation."""
        print("\n" + "="*60)
        print("STAGE 4: KV CACHE VALIDATION")
        print("="*60)
        
        # Use a shorter prompt for cache testing
        prompt_ids = self.test_ids_16[:, :8]  # First 8 tokens
        next_token_ids = self.test_ids_16[:, 8:9]  # 9th token
        
        print(f"Prompt: {prompt_ids.tolist()}")
        print(f"Next token: {next_token_ids.tolist()}")
        
        # PyTorch: Full pass + cached pass
        with torch.no_grad():
            pt_out1 = pt_model(torch.tensor(prompt_ids), use_cache=True)
            pt_logits1 = pt_out1.logits.cpu().numpy()
            pt_cache = pt_out1.past_key_values
            
            pt_out2 = pt_model(
                torch.tensor(next_token_ids),
                past_key_values=pt_cache,
                use_cache=True
            )
            pt_logits2 = pt_out2.logits.cpu().numpy()
        
        # JAX: Full pass + cached pass
        jax_out1 = jax_model.apply(
            jax_params,
            input_ids=jnp.array(prompt_ids),
            return_dict=True
        )
        jax_logits1 = np.array(jax_out1["logits"])
        jax_cache = jax_out1["past_key_values"]
        
        jax_out2 = jax_model.apply(
            jax_params,
            input_ids=jnp.array(next_token_ids),
            past_key_values=jax_cache,
            return_dict=True
        )
        jax_logits2 = np.array(jax_out2["logits"])
        
        # Compare cached step logits
        max_diff = np.max(np.abs(pt_logits2 - jax_logits2))
        
        print(f"Cached logits shape: PT={pt_logits2.shape}, JAX={jax_logits2.shape}")
        print(f"Max difference in cached step: {max_diff:.2e}")
        
        # Compare cache shapes
        cache_shapes_match = True
        if pt_cache is not None and jax_cache is not None:
            for i, (pt_kv, jax_kv) in enumerate(zip(pt_cache, jax_cache)):
                pt_k, pt_v = pt_kv
                jax_k, jax_v = jax_kv
                
                if pt_k.shape != jax_k.shape or pt_v.shape != jax_v.shape:
                    print(f"❌ Layer {i} cache shape mismatch:")
                    print(f"  PT: k={pt_k.shape}, v={pt_v.shape}")
                    print(f"  JAX: k={jax_k.shape}, v={jax_v.shape}")
                    cache_shapes_match = False
        
        success = max_diff < 1e-4 and cache_shapes_match
        
        if success:
            print("✓ Stage 4 PASSED: KV cache validation successful!")
        else:
            print("❌ Stage 4 FAILED: Cache validation failed")
        
        self.results["stage_4"] = {
            "success": success,
            "max_diff": float(max_diff),
            "cache_shapes_match": cache_shapes_match
        }
        
        return success
    
    def stage_5_greedy_decode(self, pt_model: torch.nn.Module, jax_model: Any, jax_params: Dict) -> bool:
        """Stage 5: Greedy decode text matching."""
        print("\n" + "="*60)
        print("STAGE 5: GREEDY DECODE (BYTE-FOR-BYTE)")
        print("="*60)
        
        prompt = "The relationship between AI"
        max_new_tokens = 16
        
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")
        
        # PyTorch greedy generation
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            pt_outputs = pt_model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                temperature=0.0,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        pt_generated = pt_outputs[0][inputs.input_ids.shape[1]:]
        pt_text = self.tokenizer.decode(pt_generated, skip_special_tokens=True)
        
        # JAX greedy generation
        jax_inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = jax_inputs["input_ids"]
        
        generated_tokens = []
        current_ids = input_ids
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Forward pass
            outputs = jax_model.apply(
                jax_params,
                input_ids=jnp.array(current_ids),
                past_key_values=past_key_values,
                return_dict=True
            )
            
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]
            
            # Greedy selection
            next_token = jnp.argmax(logits[0, -1, :])
            generated_tokens.append(int(next_token))
            
            # Update for next iteration
            current_ids = np.array([[int(next_token)]], dtype=np.int32)
            
            # Check for EOS
            if int(next_token) == self.tokenizer.eos_token_id:
                break
        
        jax_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\nPyTorch generated: '{pt_text}'")
        print(f"JAX generated:     '{jax_text}'")
        
        text_matches = pt_text == jax_text
        
        if text_matches:
            print("✓ Stage 5 PASSED: Byte-for-byte text match!")
        else:
            print("❌ Stage 5 FAILED: Generated text differs")
            
            # Character-by-character comparison
            min_len = min(len(pt_text), len(jax_text))
            for i in range(min_len):
                if pt_text[i] != jax_text[i]:
                    print(f"  First difference at position {i}: '{pt_text[i]}' vs '{jax_text[i]}'")
                    break
        
        self.results["stage_5"] = {
            "success": text_matches,
            "pt_text": pt_text,
            "jax_text": jax_text,
            "tokens_generated": len(generated_tokens)
        }
        
        return text_matches
    
    def run_quick_test(self) -> Dict:
        """Run just Stage 2 and Stage 5 for initial comparison."""
        print("Loading models...")
        pt_model = self.load_pytorch_model()
        jax_model, jax_params = self.load_jax_model()
        
        # Run key stages
        stages = [
            ("Stage 2: Single Token", self.stage_2_single_token),
            ("Stage 5: Greedy Decode", self.stage_5_greedy_decode),
        ]
        
        all_passed = True
        for stage_name, stage_func in stages:
            try:
                success = stage_func(pt_model, jax_model, jax_params)
                if not success:
                    all_passed = False
                    print(f"\n⚠️  {stage_name} failed")
            except Exception as e:
                print(f"\n❌ {stage_name} crashed: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        # Cleanup
        del pt_model, jax_model, jax_params
        gc.collect()
        
        # Final report
        print("\n" + "="*80)
        print("QUICK PARITY TEST REPORT")
        print("="*80)
        
        for stage, result in self.results.items():
            status = "✓ PASS" if result["success"] else "❌ FAIL"
            print(f"{stage.upper()}: {status}")
        
        if all_passed:
            print("\n🎉 QUICK TEST PASSED! Core functionality matches!")
        else:
            print("\n⚠️  Quick test failed. Need to debug core issues first.")
        
        return self.results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch vs JAX Parity Validation")
    parser.add_argument("--model_path", type=str, default="../weights", help="Path to model weights")
    args = parser.parse_args()
    
    validator = ParityValidator(args.model_path)
    results = validator.run_quick_test()
    
    return results

if __name__ == "__main__":
    main() 