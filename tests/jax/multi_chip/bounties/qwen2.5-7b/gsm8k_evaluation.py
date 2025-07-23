# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GSM8K evaluation for Qwen2.5-7B tensor parallel implementation.

This module evaluates mathematical reasoning capabilities on the GSM8K dataset
and compares performance across different mesh configurations to ensure
tensor parallelism maintains quality.

Following bounty requirements for GSM8K accuracy validation.
"""

import os
import re
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

import qwen_nnx
from qwen_nnx.generate import Generator
from mesh_configs import get_mesh_and_sharding, validate_mesh_configuration


class GSM8KEvaluator:
    """GSM8K evaluator for Qwen2.5-7B model."""
    
    def __init__(self, model_path: str, mesh_type: str = 'single', dtype=jnp.bfloat16):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model weights
            mesh_type: Mesh configuration ('single', '2x4', '1x8', etc.)
            dtype: Model dtype
        """
        self.model_path = model_path
        self.mesh_type = mesh_type
        self.dtype = dtype
        
        print(f"🔧 Initializing GSM8K evaluator...")
        print(f"    Model path: {model_path}")
        print(f"    Mesh type: {mesh_type}")
        print(f"    Dtype: {dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Setup mesh and model
        self.mesh, self.sharding_rules = get_mesh_and_sharding(mesh_type)
        validate_mesh_configuration(self.mesh, self.sharding_rules)
        
        # Load model
        with self.mesh:
            self.model = qwen_nnx.QwenModel.load_from_hf_pt_model(
                model_path,
                dtype=dtype,
                mesh=self.mesh,
                sharding_rules=self.sharding_rules,
            )
            
            # Initialize generator
            self.generator = Generator(self.model, max_seqlen=512)
        
        print("✅ GSM8K evaluator initialized successfully")
    
    def create_gsm8k_prompt(self, problem: str) -> str:
        """Create a prompt for GSM8K problem.
        
        Args:
            problem: The math problem statement
            
        Returns:
            Formatted prompt string
        """
        # Use a simple, clear prompt format
        prompt = f"Question: {problem}\nAnswer: Let me solve this step by step.\n\n"
        return prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="jax")["input_ids"]
        
        # Generate with the mesh context
        with self.mesh:
            # Use generator for response
            from flax import nnx
            rngs = nnx.Rngs(42)  # Fixed seed for reproducibility
            generated_ids = self.generator.generate(
                input_ids[0],  # Remove batch dim
                rngs,
                max_tokens=max_tokens,
                temp=0.1,  # Low temperature for deterministic results
                top_p=0.9
            )
        
        # Decode response
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract the generated part (remove input prompt)
        if prompt in response:
            response = response[len(prompt):].strip()
        
        return response
    
    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from generated text.
        
        Args:
            text: Generated response text
            
        Returns:
            Extracted numerical answer or None if not found
        """
        # Look for patterns that typically contain the final answer
        patterns = [
            r"The answer is\s*\$?(-?\d+(?:\.\d+)?)",  # "The answer is X"
            r"####\s*\$?(-?\d+(?:\.\d+)?)",           # "#### X" (GSM8K format)
            r"Therefore.*?\$?(-?\d+(?:\.\d+)?)",      # "Therefore X"
            r"So.*?\$?(-?\d+(?:\.\d+)?)",             # "So X"
            r"\$?(-?\d+(?:\.\d+)?)\s*\.?\s*$",        # Number at end of line
            r"=\s*\$?(-?\d+(?:\.\d+)?)",              # "= X"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the last match (most likely to be final answer)
                    answer_str = matches[-1]
                    answer = float(answer_str)
                    return answer
                except ValueError:
                    continue
        
        # If no clear pattern, try to find any number in the text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])  # Take last number
            except ValueError:
                pass
        
        return None
    
    def evaluate_problem(self, problem: str, expected_answer: float) -> Dict:
        """Evaluate a single GSM8K problem.
        
        Args:
            problem: Problem statement
            expected_answer: Expected correct answer
            
        Returns:
            Dictionary with evaluation results
        """
        # Create prompt
        prompt = self.create_gsm8k_prompt(problem)
        
        # Generate response
        start_time = time.time()
        try:
            response = self.generate_response(prompt)
            generation_time = time.time() - start_time
            
            # Extract answer
            predicted_answer = self.extract_answer(response)
            
            # Check correctness
            is_correct = False
            if predicted_answer is not None:
                # Allow small floating point differences
                is_correct = abs(predicted_answer - expected_answer) < 1e-6
            
            result = {
                'problem': problem,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'response': response,
                'generation_time': generation_time,
                'mesh_type': self.mesh_type
            }
            
        except Exception as e:
            result = {
                'problem': problem,
                'expected_answer': expected_answer,
                'predicted_answer': None,
                'is_correct': False,
                'response': f"Error: {str(e)}",
                'generation_time': time.time() - start_time,
                'mesh_type': self.mesh_type
            }
        
        return result
    
    def run_evaluation(self, problems: List[Dict], num_samples: Optional[int] = None) -> Dict:
        """Run GSM8K evaluation on a set of problems.
        
        Args:
            problems: List of problem dictionaries with 'question' and 'answer'
            num_samples: Number of problems to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary
        """
        if num_samples is not None:
            problems = problems[:num_samples]
        
        print(f"🧮 Starting GSM8K evaluation...")
        print(f"    Number of problems: {len(problems)}")
        print(f"    Mesh type: {self.mesh_type}")
        
        results = []
        correct_count = 0
        
        for i, problem_data in enumerate(problems):
            print(f"\n📝 Problem {i+1}/{len(problems)}")
            
            # Extract problem and answer
            problem = problem_data['question']
            # GSM8K answers are in format "Some text\n#### 42"
            answer_text = problem_data['answer']
            expected_answer = float(answer_text.split("####")[-1].strip())
            
            print(f"    Problem: {problem[:100]}...")
            print(f"    Expected answer: {expected_answer}")
            
            # Evaluate
            result = self.evaluate_problem(problem, expected_answer)
            results.append(result)
            
            # Log result
            if result['is_correct']:
                correct_count += 1
                print(f"    ✅ Correct! Predicted: {result['predicted_answer']}")
            else:
                print(f"    ❌ Wrong. Predicted: {result['predicted_answer']}")
            
            print(f"    Response: {result['response'][:200]}...")
            print(f"    Time: {result['generation_time']:.2f}s")
        
        # Calculate metrics
        accuracy = correct_count / len(problems) if problems else 0.0
        avg_time = sum(r['generation_time'] for r in results) / len(results) if results else 0.0
        
        evaluation_results = {
            'mesh_type': self.mesh_type,
            'num_problems': len(problems),
            'correct_count': correct_count,
            'accuracy': accuracy,
            'avg_generation_time': avg_time,
            'results': results
        }
        
        print(f"\n📊 GSM8K Evaluation Results ({self.mesh_type}):")
        print(f"    Problems evaluated: {len(problems)}")
        print(f"    Correct answers: {correct_count}")
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    Average generation time: {avg_time:.2f}s")
        
        return evaluation_results


def load_gsm8k_dataset(num_samples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K dataset problems.
    
    Args:
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of problem dictionaries
    """
    try:
        import datasets
        dataset = datasets.load_dataset("gsm8k", "main", split="test")
        problems = [{"question": item["question"], "answer": item["answer"]} 
                   for item in dataset]
        
        if num_samples is not None:
            problems = problems[:num_samples]
        
        print(f"📚 Loaded {len(problems)} GSM8K problems")
        return problems
        
    except ImportError:
        print("⚠️ datasets package not available, using sample problems")
        # Fallback to sample problems
        return get_sample_gsm8k_problems()


def get_sample_gsm8k_problems() -> List[Dict]:
    """Get sample GSM8K problems for testing.
    
    Returns:
        List of sample problem dictionaries
    """
    sample_problems = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?",
            "answer": "Janet's ducks lay 16 eggs per day.\nShe eats 3 for breakfast.\nShe bakes 4 into muffins.\nSo she has 16 - 3 - 4 = 9 eggs left.\nShe sells them for $2 each.\nSo she makes 9 * 2 = $18 every day.\n#### 18"
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take?",
            "answer": "A robe takes 2 bolts of blue fiber.\nIt takes half that much white fiber, so 2 / 2 = 1 bolt of white fiber.\nSo in total it takes 2 + 1 = 3 bolts of fiber.\n#### 3"
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "He bought the house for $80,000 and put in $50,000 in repairs for a total cost of 80,000 + 50,000 = $130,000.\nThe value increased by 150%, so the new value is 80,000 * 1.5 = $120,000 more than the original price.\nSo the house is now worth 80,000 + 120,000 = $200,000.\nHe made a profit of 200,000 - 130,000 = $70,000.\n#### 70000"
        }
    ]
    
    print(f"📚 Using {len(sample_problems)} sample GSM8K problems")
    return sample_problems


def compare_mesh_configurations(model_path: str, problems: List[Dict], 
                              mesh_configs: List[str]) -> Dict:
    """Compare GSM8K performance across different mesh configurations.
    
    Args:
        model_path: Path to model weights
        problems: List of GSM8K problems
        mesh_configs: List of mesh configuration names
        
    Returns:
        Comparison results dictionary
    """
    print(f"🔄 Comparing GSM8K performance across {len(mesh_configs)} configurations...")
    
    all_results = {}
    
    for mesh_config in mesh_configs:
        print(f"\n{'='*60}")
        print(f"Testing mesh configuration: {mesh_config}")
        
        try:
            evaluator = GSM8KEvaluator(model_path, mesh_config)
            results = evaluator.run_evaluation(problems)
            all_results[mesh_config] = results
            
        except Exception as e:
            print(f"❌ Failed to evaluate {mesh_config}: {e}")
            all_results[mesh_config] = {
                'mesh_type': mesh_config,
                'error': str(e),
                'accuracy': 0.0
            }
    
    # Create comparison summary
    print(f"\n{'='*60}")
    print("📊 GSM8K Performance Comparison:")
    print(f"{'Configuration':<12} {'Accuracy':<10} {'Problems':<10} {'Avg Time':<10}")
    print("-" * 45)
    
    for config, results in all_results.items():
        if 'accuracy' in results:
            accuracy = f"{results['accuracy']:.1%}"
            num_problems = results.get('num_problems', 0)
            avg_time = f"{results.get('avg_generation_time', 0):.2f}s"
        else:
            accuracy = "ERROR"
            num_problems = 0
            avg_time = "N/A"
        
        print(f"{config:<12} {accuracy:<10} {num_problems:<10} {avg_time:<10}")
    
    # Check if tensor parallel results match single device
    if 'single' in all_results:
        single_accuracy = all_results['single'].get('accuracy', 0.0)
        
        print(f"\n🔍 Tensor Parallel vs Single Device Comparison:")
        for config, results in all_results.items():
            if config != 'single' and 'accuracy' in results:
                config_accuracy = results['accuracy']
                diff = abs(config_accuracy - single_accuracy)
                status = "✅ PASS" if diff < 0.02 else "❌ FAIL"  # 2% tolerance
                print(f"    {config}: {config_accuracy:.1%} vs {single_accuracy:.1%} "
                      f"(diff: {diff:.1%}) {status}")
    
    return all_results


def main():
    """Main GSM8K evaluation script."""
    parser = argparse.ArgumentParser(description="GSM8K evaluation for Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, 
                       default="/root/dir716/tt-xla-qwen-dev/tests/jax/multi_chip/bounties/qwen25-7b/qwen25_7b_instruct_weights",
                       help="Path to model weights")
    parser.add_argument("--mesh_config", type=str, default="single",
                       choices=['single', '2x4', '1x8', '1x32', '8x4', 'replicated'],
                       help="Mesh configuration")
    parser.add_argument("--num_problems", type=int, default=10,
                       help="Number of problems to evaluate")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all available mesh configurations")
    parser.add_argument("--output_file", type=str, default=None,
                       help="File to save results (JSON format)")
    
    args = parser.parse_args()
    
    # Load problems
    problems = load_gsm8k_dataset(args.num_problems)
    
    if args.compare_all:
        # Test available configurations
        available_devices = len(jax.devices())
        mesh_configs = ['single', 'replicated']
        
        if available_devices >= 8:
            mesh_configs.extend(['1x8', '2x4'])
        if available_devices >= 32:
            mesh_configs.extend(['1x32', '8x4'])
        
        results = compare_mesh_configurations(args.model_path, problems, mesh_configs)
    else:
        # Test single configuration
        evaluator = GSM8KEvaluator(args.model_path, args.mesh_config)
        results = evaluator.run_evaluation(problems)
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {args.output_file}")
    
    return results


if __name__ == "__main__":
    results = main() 