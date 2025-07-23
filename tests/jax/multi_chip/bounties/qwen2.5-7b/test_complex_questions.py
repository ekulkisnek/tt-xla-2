#!/usr/bin/env python3
"""Test complex mathematical reasoning with proper text generation."""

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
import qwen_nnx
from qwen_nnx.generate import Generator

def test_complex_math():
    """Test the model with increasingly complex mathematical problems."""
    
    print("🧮 Loading Qwen2.5-7B for Complex Math Testing")
    print("=" * 60)
    
    # Load model and tokenizer
    model_path = '../qwen25-7b/qwen25_7b_instruct_weights'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = qwen_nnx.QwenModel.load_from_hf_pt_model(model_path, dtype=jnp.bfloat16)
    
    # Create generator for proper text generation
    generator = Generator(model, max_seqlen=512)
    rngs = nnx.Rngs(42)  # Fixed seed for reproducibility
    
    def generate_response(prompt, max_tokens=100):
        """Generate a complete response for the given prompt."""
        tokens = jnp.array(tokenizer.encode(prompt))
        generated_tokens = generator.generate(
            tokens,
            rngs,
            max_tokens=max_tokens,
            temp=0.1,  # Low temperature for deterministic results
            top_p=0.9
        )
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Test cases with different complexity levels and prompting strategies
    test_cases = [
        # Basic arithmetic
        {
            "prompt": "3+1=",
            "description": "Simple addition (baseline)",
            "expected": "4"
        },
        {
            "prompt": "What is 5+3?",
            "description": "Simple addition as question",
            "expected": "8"
        },
        {
            "prompt": "Calculate: 12 * 4",
            "description": "Simple multiplication",
            "expected": "48"
        },
        
        # Multi-step problems with different prompt formats
        {
            "prompt": "Question: If I have 10 apples and eat 3, how many apples do I have left?\nAnswer:",
            "description": "Simple word problem (Q&A format)",
            "expected": "7"
        },
        {
            "prompt": "Solve step by step: A store has 24 books. They sell 8 books and then receive 15 more books. How many books do they have now?",
            "description": "Multi-step word problem",
            "expected": "31"
        },
        
        # GSM8K-style problems with proper formatting
        {
            "prompt": """Question: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day?

Answer: Let me solve this step by step.

""",
            "description": "GSM8K: Janet's eggs (complex word problem)",
            "expected": "18"
        },
        
        {
            "prompt": """Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take?

Answer: Let me solve this step by step.

""",
            "description": "GSM8K: Fiber bolts (fraction calculation)",
            "expected": "3"
        },
        
        # Test with explicit instruction prompting
        {
            "prompt": """You are a helpful assistant that solves math problems step by step.

Question: Tom has 25 marbles. He gives 7 marbles to his sister and 5 marbles to his brother. How many marbles does Tom have left?

Answer: I need to subtract the marbles Tom gave away from his original amount.
""",
            "description": "Instructional prompt format",
            "expected": "13"
        },
        
        # More complex reasoning
        {
            "prompt": """Question: A school bought 150 pencils. They distributed 1/3 of them to grade 3, 1/5 of them to grade 4, and gave the rest to grade 5. How many pencils did grade 5 receive?

Answer: Let me calculate this step by step.

""",
            "description": "Fraction and remainder calculation",
            "expected": "70"
        }
    ]
    
    print(f"Testing {len(test_cases)} complex math problems...\n")
    
    results = []
    correct_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test {i}/{len(test_cases)}: {test_case['description']}")
        print(f"Prompt: {test_case['prompt'][:100]}{'...' if len(test_case['prompt']) > 100 else ''}")
        
        try:
            # Generate response
            response = generate_response(test_case['prompt'], max_tokens=150)
            
            # Extract just the generated part (remove input prompt)
            if test_case['prompt'] in response:
                generated = response[len(test_case['prompt']):].strip()
            else:
                generated = response.strip()
            
            print(f"Generated: {generated[:200]}{'...' if len(generated) > 200 else ''}")
            
            # Simple check for expected answer in response
            expected = str(test_case['expected'])
            is_correct = expected in generated
            
            if is_correct:
                correct_count += 1
                print(f"✅ CORRECT! Found expected answer: {expected}")
            else:
                print(f"❌ INCORRECT. Expected: {expected}")
            
            results.append({
                'test': test_case['description'],
                'prompt': test_case['prompt'],
                'expected': expected,
                'generated': generated,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'test': test_case['description'],
                'prompt': test_case['prompt'],
                'expected': test_case['expected'],
                'generated': f"ERROR: {str(e)}",
                'correct': False
            })
        
        print("-" * 80)
    
    # Summary
    accuracy = correct_count / len(test_cases) * 100
    print(f"\n📊 RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"Tests passed: {correct_count}/{len(test_cases)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Model: Qwen2.5-7B with tensor parallel infrastructure")
    
    # Show detailed results for failed cases
    if correct_count < len(test_cases):
        print(f"\n🔍 FAILED TESTS ANALYSIS:")
        for result in results:
            if not result['correct']:
                print(f"\n❌ {result['test']}")
                print(f"   Expected: {result['expected']}")
                print(f"   Generated: {result['generated'][:150]}...")
    
    return results, accuracy

if __name__ == "__main__":
    results, accuracy = test_complex_math()
    
    if accuracy >= 70:
        print(f"\n🎉 SUCCESS! Model achieves {accuracy:.1f}% accuracy on complex math problems!")
    else:
        print(f"\n🔧 NEEDS IMPROVEMENT: {accuracy:.1f}% accuracy. Working on prompt optimization...") 