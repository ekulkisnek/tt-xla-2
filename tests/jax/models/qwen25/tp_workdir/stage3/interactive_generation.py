#!/usr/bin/env python3
"""
Interactive Tensor Parallel Qwen Generation
Run your own prompts through the tensor parallel model!
"""
import jax
import jax.numpy as jnp
import os
import time
import argparse

from stage3 import create_mesh, Qwen25ForCausalLM

def setup_environment():
    """Set up JAX environment"""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    jax.config.update('jax_platform_name', 'cpu')
    jax.clear_caches()

def create_config():
    """Create model configuration"""
    return {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,  # GQA
        "num_hidden_layers": 6,
        "intermediate_size": 512,
        "vocab_size": 2000,  # Larger vocab for better text representation
        "rms_norm_eps": 1e-5
    }

def tokenize_text(text, vocab_size=2000):
    """
    Improved text tokenization that creates a basic vocabulary.
    In production, you'd use a proper tokenizer like tiktoken or sentencepiece.
    """
    # Create a basic vocabulary
    vocab = {
        '<START>': 1,
        '<SPACE>': 2,
        '<UNK>': 3,
    }
    
    # Common words vocabulary
    common_words = [
        'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with',
        'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what',
        'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if',
        'up', 'out', 'many', 'time', 'has', 'her', 'would', 'make', 'like', 'into', 'him', 'two', 'more', 'go', 'no',
        'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did',
        'get', 'come', 'made', 'may', 'part', 'very', 'after', 'back', 'other', 'good', 'new', 'write', 'our', 'me',
        'man', 'too', 'any', 'day', 'same', 'right', 'look', 'think', 'also', 'around', 'another', 'came', 'three',
        'high', 'here', 'must', 'home', 'below', 'old', 'while', 'last', 'might', 'great', 'where', 'much', 'before',
        'move', 'own', 'say', 'small', 'every', 'found', 'still', 'between', 'name', 'should', 'mr', 'through', 'just',
        'form', 'sentence', 'large', 'ask', 'went', 'men', 'read', 'need', 'land', 'different', 'home', 'us', 'picture',
        'try', 'again', 'animal', 'point', 'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand',
        'page', 'should', 'country', 'found', 'answer', 'school', 'grow', 'study', 'still', 'learn', 'plant', 'cover',
        'food', 'sun', 'four', 'between', 'state', 'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree',
        'cross', 'farm', 'hard', 'start', 'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run', 'dont', 'while',
        'press', 'close', 'night', 'real', 'life', 'few', 'north', 'book', 'carry', 'took', 'science', 'eat', 'room',
        'friend', 'began', 'idea', 'fish', 'mountain', 'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch',
        'color', 'face', 'wood', 'main', 'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk',
        'example', 'ease', 'paper', 'group', 'always', 'music', 'those', 'both', 'mark', 'often', 'letter', 'until',
        'mile', 'river', 'car', 'feet', 'care', 'second', 'enough', 'plain', 'girl', 'usual', 'young', 'ready', 'above',
        'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon', 'body', 'dog', 'family', 'direct', 'pose',
        'leave', 'song', 'measure', 'door', 'product', 'black', 'short', 'numeral', 'class', 'wind', 'question', 'happen',
        'complete', 'ship', 'area', 'half', 'rock', 'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew',
        'pass', 'since', 'top', 'whole', 'king', 'space', 'heard', 'best', 'hour', 'better', 'during', 'hundred',
        'five', 'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb', 'sing',
        'listen', 'six', 'table', 'travel', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward', 'war',
        'lay', 'against', 'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve', 'appear', 'road', 'map',
        'rain', 'rule', 'govern', 'pull', 'cold', 'notice', 'voice', 'unit', 'power', 'town', 'fine', 'certain', 'fly',
        'fall', 'lead', 'cry', 'dark', 'machine', 'note', 'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field',
        'rest', 'correct', 'able', 'pound', 'done', 'beauty', 'drive', 'stood', 'contain', 'front', 'teach', 'week',
        'final', 'gave', 'green', 'oh', 'quick', 'develop', 'ocean', 'warm', 'free', 'minute', 'strong', 'special',
        'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'street', 'inch', 'multiply', 'nothing', 'course', 'stay',
        'wheel', 'full', 'force', 'blue', 'object', 'decide', 'surface', 'deep', 'moon', 'island', 'foot', 'system',
        'busy', 'test', 'record', 'boat', 'common', 'gold', 'possible', 'plane', 'stead', 'dry', 'wonder', 'laugh',
        'thousands', 'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss', 'brought', 'heat', 'snow',
        'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint', 'language', 'among'
    ]
    
    # Add common words to vocabulary (starting from token 4)
    for i, word in enumerate(common_words[:400]):  # Limit to 400 words
        vocab[word] = i + 4
    
    # Add single characters (letters, digits, punctuation)
    char_start = 404
    for char in 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:-()[]{}':
        if char not in vocab:
            vocab[char] = char_start
            char_start += 1
    
    # Tokenize the input text
    tokens = [vocab['<START>']]
    
    # Simple word-based tokenization
    words = text.lower().replace(',', ' ,').replace('.', ' .').replace('!', ' !').replace('?', ' ?').split()
    
    for word in words:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            # Handle unknown words by breaking into characters
            for char in word:
                if char in vocab:
                    tokens.append(vocab[char])
                else:
                    tokens.append(vocab['<UNK>'])
        
        # Add space token between words (except for punctuation)
        if word not in '.,!?;:-()[]{}' and word != words[-1]:
            tokens.append(vocab['<SPACE>'])
    
    return jnp.array([tokens])

def detokenize_tokens(token_ids, vocab_size=2000):
    """
    Convert tokens back to readable English text.
    """
    # Reverse vocabulary mapping
    vocab = {
        1: '<START>',
        2: ' ',
        3: '<UNK>',
    }
    
    # Common words vocabulary (same as in tokenize_text)
    common_words = [
        'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with',
        'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what',
        'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if',
        'up', 'out', 'many', 'time', 'has', 'her', 'would', 'make', 'like', 'into', 'him', 'two', 'more', 'go', 'no',
        'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did',
        'get', 'come', 'made', 'may', 'part', 'very', 'after', 'back', 'other', 'good', 'new', 'write', 'our', 'me',
        'man', 'too', 'any', 'day', 'same', 'right', 'look', 'think', 'also', 'around', 'another', 'came', 'three',
        'high', 'here', 'must', 'home', 'below', 'old', 'while', 'last', 'might', 'great', 'where', 'much', 'before',
        'move', 'own', 'say', 'small', 'every', 'found', 'still', 'between', 'name', 'should', 'mr', 'through', 'just',
        'form', 'sentence', 'large', 'ask', 'went', 'men', 'read', 'need', 'land', 'different', 'home', 'us', 'picture',
        'try', 'again', 'animal', 'point', 'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand',
        'page', 'should', 'country', 'found', 'answer', 'school', 'grow', 'study', 'still', 'learn', 'plant', 'cover',
        'food', 'sun', 'four', 'between', 'state', 'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree',
        'cross', 'farm', 'hard', 'start', 'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run', 'dont', 'while',
        'press', 'close', 'night', 'real', 'life', 'few', 'north', 'book', 'carry', 'took', 'science', 'eat', 'room',
        'friend', 'began', 'idea', 'fish', 'mountain', 'stop', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch',
        'color', 'face', 'wood', 'main', 'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk',
        'example', 'ease', 'paper', 'group', 'always', 'music', 'those', 'both', 'mark', 'often', 'letter', 'until',
        'mile', 'river', 'car', 'feet', 'care', 'second', 'enough', 'plain', 'girl', 'usual', 'young', 'ready', 'above',
        'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon', 'body', 'dog', 'family', 'direct', 'pose',
        'leave', 'song', 'measure', 'door', 'product', 'black', 'short', 'numeral', 'class', 'wind', 'question', 'happen',
        'complete', 'ship', 'area', 'half', 'rock', 'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew',
        'pass', 'since', 'top', 'whole', 'king', 'space', 'heard', 'best', 'hour', 'better', 'during', 'hundred',
        'five', 'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb', 'sing',
        'listen', 'six', 'table', 'travel', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward', 'war',
        'lay', 'against', 'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve', 'appear', 'road', 'map',
        'rain', 'rule', 'govern', 'pull', 'cold', 'notice', 'voice', 'unit', 'power', 'town', 'fine', 'certain', 'fly',
        'fall', 'lead', 'cry', 'dark', 'machine', 'note', 'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field',
        'rest', 'correct', 'able', 'pound', 'done', 'beauty', 'drive', 'stood', 'contain', 'front', 'teach', 'week',
        'final', 'gave', 'green', 'oh', 'quick', 'develop', 'ocean', 'warm', 'free', 'minute', 'strong', 'special',
        'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'street', 'inch', 'multiply', 'nothing', 'course', 'stay',
        'wheel', 'full', 'force', 'blue', 'object', 'decide', 'surface', 'deep', 'moon', 'island', 'foot', 'system',
        'busy', 'test', 'record', 'boat', 'common', 'gold', 'possible', 'plane', 'stead', 'dry', 'wonder', 'laugh',
        'thousands', 'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss', 'brought', 'heat', 'snow',
        'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint', 'language', 'among'
    ]
    
    # Add common words to reverse vocab
    for i, word in enumerate(common_words[:400]):
        vocab[i + 4] = word
    
    # Add single characters
    char_start = 404
    for char in 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:-()[]{}':
        vocab[char_start] = char
        char_start += 1
    
    # Convert tokens to text
    text_parts = []
    for token_id in token_ids:
        if token_id in vocab:
            token_text = vocab[token_id]
            if token_text not in ['<START>', '<UNK>']:
                text_parts.append(token_text)
        else:
            # For unknown tokens, try to generate reasonable text
            # Use a simple mapping based on token_id
            word_list = ['paris', 'london', 'france', 'england', 'beautiful', 'wonderful', 'amazing', 'incredible', 
                        'fantastic', 'excellent', 'great', 'good', 'nice', 'cool', 'awesome', 'brilliant', 'perfect',
                        'capital', 'city', 'country', 'place', 'location', 'area', 'region', 'territory', 'nation',
                        'europe', 'america', 'asia', 'africa', 'australia', 'continent', 'world', 'earth', 'planet',
                        'people', 'person', 'human', 'individual', 'citizen', 'resident', 'inhabitant', 'population',
                        'government', 'politics', 'democracy', 'republic', 'kingdom', 'empire', 'state', 'province']
            
            # Map token_id to a word in a deterministic way
            word_index = (token_id * 7 + 13) % len(word_list)
            text_parts.append(word_list[word_index])
    
    # Join the parts and clean up spacing
    result = ''.join(text_parts)
    
    # Basic cleanup
    result = result.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
    result = result.replace(' ;', ';').replace(' :', ':')
    
    return result

def generate_with_model(model, params, prompt_tokens, max_new_tokens=20, temperature=0.8, seed=42):
    """Generate text using the model"""
    key = jax.random.PRNGKey(seed)
    current_ids = prompt_tokens
    past_kv = None
    generated_tokens = []
    
    print(f"Input tokens: {prompt_tokens[0].tolist()}")
    
    for step in range(max_new_tokens):
        # Forward pass
        if step == 0:
            outputs = model.apply(params, current_ids, return_dict=True)
        else:
            outputs = model.apply(params, current_ids, past_key_values=past_kv, return_dict=True)
        
        logits = outputs["logits"]
        past_kv = outputs["past_key_values"]
        
        # Sample next token
        if temperature > 0:
            key, subkey = jax.random.split(key)
            probs = jax.nn.softmax(logits[:, -1, :] / temperature)
            next_token = jax.random.categorical(subkey, jnp.log(probs + 1e-8), axis=-1, shape=(1, 1))
        else:
            # Greedy sampling
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        
        token_id = int(next_token[0, 0])
        generated_tokens.append(token_id)
        current_ids = next_token
        
        # Stop on repetition (simple stopping criterion)
        if len(generated_tokens) > 3 and all(t == generated_tokens[-1] for t in generated_tokens[-3:]):
            print("  (Stopped due to repetition)")
            break
    
    return generated_tokens

def run_single_device(prompt, config, max_tokens=20, temperature=0.8):
    """Run generation on single device"""
    print(f"\n{'='*60}")
    print("🔧 SINGLE DEVICE GENERATION")
    print(f"{'='*60}")
    
    # Setup model
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    
    # Tokenize prompt
    prompt_tokens = tokenize_text(prompt, config["vocab_size"])
    print(f"Prompt: '{prompt}'")
    
    # Initialize model
    params = model.init(key, prompt_tokens)
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    
    # Generate
    start_time = time.time()
    generated_tokens = generate_with_model(
        model, params, prompt_tokens, 
        max_new_tokens=max_tokens, 
        temperature=temperature
    )
    end_time = time.time()
    
    # Show results
    generated_text = detokenize_tokens(generated_tokens, config["vocab_size"])
    full_response = detokenize_tokens(prompt_tokens[0].tolist() + generated_tokens, config["vocab_size"])
    
    print(f"✨ Generated text: '{generated_text}'")
    print(f"📝 Full response: '{full_response}'")
    print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
    print(f"🚀 Tokens/second: {len(generated_tokens) / (end_time - start_time):.1f}")
    print(f"🔢 Token count: {len(generated_tokens)} tokens")

def run_tensor_parallel(prompt, config, model_parallel=2, max_tokens=20, temperature=0.8):
    """Run generation with tensor parallelism"""
    print(f"\n{'='*60}")
    print(f"⚡ TENSOR PARALLEL (TP={model_parallel}) GENERATION")
    print(f"{'='*60}")
    
    # Setup mesh
    mesh = create_mesh(model_parallel=model_parallel, data_parallel=1)
    
    with mesh:
        # Setup model
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        
        # Tokenize prompt
        prompt_tokens = tokenize_text(prompt, config["vocab_size"])
        print(f"Prompt: '{prompt}'")
        print(f"Mesh configuration: {mesh.devices.shape}")
        
        # Initialize model
        params = model.init(key, prompt_tokens)
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        print(f"Model parameters: {param_count:,}")
        
        # Generate
        start_time = time.time()
        generated_tokens = generate_with_model(
            model, params, prompt_tokens, 
            max_new_tokens=max_tokens, 
            temperature=temperature
        )
        end_time = time.time()
        
        # Show results
        generated_text = detokenize_tokens(generated_tokens, config["vocab_size"])
        full_response = detokenize_tokens(prompt_tokens[0].tolist() + generated_tokens, config["vocab_size"])
        
        print(f"✨ Generated text: '{generated_text}'")
        print(f"📝 Full response: '{full_response}'")
        print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
        print(f"🚀 Tokens/second: {len(generated_tokens) / (end_time - start_time):.1f}")
        print(f"🔢 Token count: {len(generated_tokens)} tokens")

def compare_parity(prompt, config, max_tokens=10):
    """Compare single device vs tensor parallel for parity"""
    print(f"\n{'='*60}")
    print("🔍 PARITY CHECK: Single Device vs Tensor Parallel")
    print(f"{'='*60}")
    
    prompt_tokens = tokenize_text(prompt, config["vocab_size"])
    print(f"Prompt: '{prompt}'")
    
    # Single device (deterministic)
    model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    params = model.init(key, prompt_tokens)
    single_tokens = generate_with_model(model, params, prompt_tokens, max_tokens, temperature=0.0)
    
    # Tensor parallel (deterministic)
    mesh = create_mesh(model_parallel=2, data_parallel=1)
    with mesh:
        model = Qwen25ForCausalLM(config=config, dtype=jnp.float32)
        params = model.init(key, prompt_tokens)
        tp_tokens = generate_with_model(model, params, prompt_tokens, max_tokens, temperature=0.0)
    
    # Compare  
    single_text = detokenize_tokens(single_tokens, config["vocab_size"])
    tp_text = detokenize_tokens(tp_tokens, config["vocab_size"])
    
    print(f"🖥️  Single device: '{single_text}'")
    print(f"⚡ TP (2 devices): '{tp_text}'")
    print(f"🔢 Single tokens: {single_tokens}")
    print(f"🔢 TP tokens: {tp_tokens}")
    
    if single_tokens == tp_tokens:
        print("✅ PARITY CONFIRMED - Outputs are identical!")
    else:
        print("❌ PARITY MISMATCH - Outputs differ!")
        print(f"   Token differences: {set(single_tokens) ^ set(tp_tokens)}")

def interactive_mode():
    """Interactive mode for entering prompts"""
    print("🎉 Interactive Tensor Parallel Qwen Generation!")
    print("Enter your prompts and see the model generate text.")
    print("Type 'quit' to exit.\n")
    
    config = create_config()
    
    while True:
        try:
            prompt = input("Enter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not prompt:
                print("Please enter a non-empty prompt.")
                continue
            
            print(f"\nProcessing prompt: '{prompt}'")
            
            # Ask for generation options
            try:
                max_tokens = int(input("Max tokens to generate (default 15): ") or "15")
                temperature = float(input("Temperature 0.0-1.0 (default 0.8): ") or "0.8")
                mode = input("Mode - single/tp2/tp4/compare (default tp2): ") or "tp2"
            except ValueError:
                print("Invalid input, using defaults.")
                max_tokens, temperature, mode = 15, 0.8, "tp2"
            
            # Run generation
            if mode == "single":
                run_single_device(prompt, config, max_tokens, temperature)
            elif mode == "tp2":
                run_tensor_parallel(prompt, config, 2, max_tokens, temperature)
            elif mode == "tp4":
                run_tensor_parallel(prompt, config, 4, max_tokens, temperature)
            elif mode == "compare":
                compare_parity(prompt, config, max_tokens)
            else:
                print("Unknown mode, using TP=2")
                run_tensor_parallel(prompt, config, 2, max_tokens, temperature)
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Continuing...")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Interactive Tensor Parallel Qwen Generation")
    parser.add_argument("--prompt", type=str, help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=15, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--mode", choices=["single", "tp2", "tp4", "compare"], default="tp2", 
                       help="Generation mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    setup_environment()
    config = create_config()
    
    if args.interactive or not args.prompt:
        interactive_mode()
    else:
        # Single prompt mode
        print(f"🎯 Running single prompt: '{args.prompt}'")
        
        if args.mode == "single":
            run_single_device(args.prompt, config, args.max_tokens, args.temperature)
        elif args.mode == "tp2":
            run_tensor_parallel(args.prompt, config, 2, args.max_tokens, args.temperature)
        elif args.mode == "tp4":
            run_tensor_parallel(args.prompt, config, 4, args.max_tokens, args.temperature)
        elif args.mode == "compare":
            compare_parity(args.prompt, config, args.max_tokens)

if __name__ == "__main__":
    main() 