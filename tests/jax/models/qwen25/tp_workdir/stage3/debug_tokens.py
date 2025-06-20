#!/usr/bin/env python3
"""Debug what tokens are being generated"""

import sys
sys.path.append('../../')
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("../../weights/")

# Check what token ID corresponds to the generated characters
text = "⽊"
token_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Text '{text}' -> Token IDs: {token_ids}")

# Check what token ID for "Hello"
hello_ids = tokenizer.encode("Hello", add_special_tokens=False)
print(f"'Hello' -> Token IDs: {hello_ids}")

# Check some specific token IDs around those ranges
for i in range(9705, 9710):
    token = tokenizer.decode([i], skip_special_tokens=True)
    print(f"Token ID {i}: '{token}'")

# Check decode of the full sequence we're seeing
full_ids = [9707, 4511, 4511, 4511, 4511]  # Hello + 4 repeating tokens
full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
print(f"Full sequence {full_ids}: '{full_text}'") 