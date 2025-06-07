#!/usr/bin/env python3
"""
Step 2: Weight loading functionality
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen25_step2")

def get_param_path(name):
    """Map a PyTorch parameter name to its Flax path."""
    # Start with basic mappings
    direct_mapping = {
        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
        "model.norm.weight": ("norm", "scale"),
        "lm_head.weight": ("lm_head", "kernel"),
    }
    
    if name in direct_mapping:
        return direct_mapping[name]
    
    # Add basic pattern matching
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
    layer_norm_match = re.match(layer_norm_pattern, name)
    if layer_norm_match:
        layer_idx = int(layer_norm_match.group(1))
        norm_type = layer_norm_match.group(2)
        layer_name = f"layers_{layer_idx}"
        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
        return (layer_name, norm_name, "scale")
    
    logger.warning(f"Unknown parameter pattern: {name}")
    return None

def test_weight_loading(model_path):
    """Test weight loading with a single safetensors file"""
    try:
        # Find first safetensors file
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        if not safetensors_files:
            logger.error("No safetensors files found")
            return False
        
        file_path = os.path.join(model_path, safetensors_files[0])
        logger.info(f"Testing weight loading with {file_path}")
        
        # Try loading a few parameters
        with safe_open(file_path, framework="numpy") as f:
            for key in list(f.keys())[:5]:  # Test first 5 parameters
                param = f.get_tensor(key)
                param_path = get_param_path(key)
                if param_path:
                    logger.info(f"Successfully mapped {key} to {param_path}")
                else:
                    logger.warning(f"Could not map {key}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing weight loading: {e}")
        return False

if __name__ == "__main__":
    model_path = "/path/to/model/weights"  # Replace with your model path
    if test_weight_loading(model_path):
        logger.info("✅ Weight loading test passed!")
    else:
        logger.error("❌ Weight loading test failed!") 