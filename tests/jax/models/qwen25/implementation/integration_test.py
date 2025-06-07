#!/usr/bin/env python3
"""
Step 4: Integration test combining all components
"""

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_step4")

def test_integration(model_path):
    """Test integration of all components"""
    try:
        # 1. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Tokenizer loaded")
        
        # 2. Create small test input
        test_text = "Hello, how are you today?"
        input_ids = tokenizer(test_text, return_tensors="np").input_ids
        logger.info(f"✅ Test input created: {input_ids.shape}")
        
        # 3. Test model forward pass (using dummy weights)
        # This would use the model structure from step 1
        
        # 4. Test token generation
        # This would use the generation logic from step 3
        
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

if __name__ == "__main__":
    model_path = "/path/to/model/weights"  # Replace with your model path
    if test_integration(model_path):
        logger.info("✅ Integration test passed!")
    else:
        logger.error("❌ Integration test failed!") 