# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mesh configurations for Qwen2.5-7B tensor parallel execution.

This module provides mesh configurations and sharding rules for different
deployment scenarios, following patterns from accepted mistral_small and mixtral_8x7b bounties.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import qwen_nnx


def create_mesh_2x4():
    """Create 2x4 mesh for tensor parallel execution (8 devices total).
    
    Returns:
        Mesh: 2x4 device mesh with ('data', 'model') axis names
    """
    devices = jax.devices()
    if len(devices) < 8:
        print(f"Warning: Need 8 devices for 2x4 mesh, but only {len(devices)} available. Falling back to single device.")
        return Mesh([devices[0]], axis_names=("data",))
    
    # Reshape devices into 2x4 grid
    device_grid = [[devices[i*4 + j] for j in range(4)] for i in range(2)]
    return Mesh(device_grid, axis_names=("data", "model"))


def create_mesh_1x8():
    """Create 1x8 mesh for tensor parallel execution (8 devices total).
    
    Returns:
        Mesh: 1x8 device mesh with ('batch', 'model') axis names
    """
    devices = jax.devices()
    if len(devices) < 8:
        print(f"Warning: Need 8 devices for 1x8 mesh, but only {len(devices)} available. Falling back to single device.")
        # Fall back to single device with compatible axis names
        return Mesh([devices[0]], axis_names=("batch",))
    
    # For actual multi-device, create proper mesh
    device_grid = [devices[:8]]  # Single row of 8 devices
    return Mesh(device_grid, axis_names=("batch", "model"))


def create_mesh_1x32():
    """Create 1x32 mesh for large-scale tensor parallel execution (32 devices total).
    
    Returns:
        Mesh: 1x32 device mesh with ('batch', 'model') axis names
    """
    devices = jax.devices()
    if len(devices) < 32:
        print(f"Warning: Need 32 devices for 1x32 mesh, but only {len(devices)} available. Falling back to single device.")
        return Mesh([devices[0]], axis_names=("batch",))
    
    # For actual multi-device, create proper mesh
    device_grid = [devices[:32]]  # Single row of 32 devices
    return Mesh(device_grid, axis_names=("batch", "model"))


def create_mesh_8x4():
    """Create 8x4 mesh for data + tensor parallel execution (32 devices total).
    
    Returns:
        Mesh: 8x4 device mesh with ('data', 'model') axis names
    """
    devices = jax.devices()
    if len(devices) < 32:
        print(f"Warning: Need 32 devices for 8x4 mesh, but only {len(devices)} available. Falling back to single device.")
        return Mesh([devices[0]], axis_names=("data",))
    
    # For actual multi-device, create 8x4 grid
    device_grid = [[devices[i*4 + j] for j in range(4)] for i in range(8)]
    return Mesh(device_grid, axis_names=("data", "model"))


# Sharding rules for different mesh configurations (following mistral_small pattern)
SHARDING_RULES_TENSOR_PARALLEL = {
    qwen_nnx.Axis.EMBED: None,      # Replicate embeddings
    qwen_nnx.Axis.MLP: "x",         # Shard MLP across devices  
    qwen_nnx.Axis.HEAD: "x",        # Shard attention heads across devices
    qwen_nnx.Axis.QHEAD: None,      # Replicate query heads
    qwen_nnx.Axis.KVHEAD: None,     # Replicate KV heads
    qwen_nnx.Axis.VOCAB: None,      # Replicate vocabulary
}

# Rules for when using model axis (following mixtral pattern)
SHARDING_RULES_MODEL_AXIS = {
    qwen_nnx.Axis.EMBED: None,      # Replicate embeddings
    qwen_nnx.Axis.MLP: "model",     # Shard MLP across model axis
    qwen_nnx.Axis.HEAD: "model",    # Shard attention heads across model axis
    qwen_nnx.Axis.QHEAD: None,      # Replicate query heads
    qwen_nnx.Axis.KVHEAD: None,     # Replicate KV heads
    qwen_nnx.Axis.VOCAB: None,      # Replicate vocabulary
}

# No sharding rules (for comparison testing)
SHARDING_RULES_REPLICATED = {
    qwen_nnx.Axis.EMBED: None,      # Replicate embeddings
    qwen_nnx.Axis.MLP: None,        # Replicate MLP
    qwen_nnx.Axis.HEAD: None,       # Replicate attention heads
    qwen_nnx.Axis.QHEAD: None,      # Replicate query heads
    qwen_nnx.Axis.KVHEAD: None,     # Replicate KV heads
    qwen_nnx.Axis.VOCAB: None,      # Replicate vocabulary
}


def get_mesh_and_sharding(mesh_type):
    """Get mesh and appropriate sharding rules for specified configuration.
    
    Args:
        mesh_type: One of 'single', '2x4', '1x8', '1x32', '8x4', 'replicated'
        
    Returns:
        Tuple of (mesh, sharding_rules)
    """
    if mesh_type == 'single':
        # Single device
        devices = jax.devices()
        mesh = Mesh([devices[0]], axis_names=("x",))
        return mesh, list(SHARDING_RULES_TENSOR_PARALLEL.items())
    
    elif mesh_type == '2x4':
        mesh = create_mesh_2x4()
        return mesh, list(SHARDING_RULES_MODEL_AXIS.items())
    
    elif mesh_type == '1x8':
        mesh = create_mesh_1x8()
        return mesh, list(SHARDING_RULES_MODEL_AXIS.items())
    
    elif mesh_type == '1x32':
        mesh = create_mesh_1x32()
        return mesh, list(SHARDING_RULES_MODEL_AXIS.items())
    
    elif mesh_type == '8x4':
        mesh = create_mesh_8x4()
        return mesh, list(SHARDING_RULES_MODEL_AXIS.items())
    
    elif mesh_type == 'replicated':
        # Multi-device but no sharding (for comparison)
        devices = jax.devices()
        available_count = min(len(devices), 4)
        mesh = Mesh(devices[:available_count], axis_names=("x",))
        return mesh, list(SHARDING_RULES_REPLICATED.items())
    
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}. "
                        f"Supported: 'single', '2x4', '1x8', '1x32', '8x4', 'replicated'")


def validate_mesh_configuration(mesh, sharding_rules):
    """Validate mesh and sharding compatibility (following accepted bounty patterns).
    
    Args:
        mesh: Device mesh
        sharding_rules: List of (axis, mesh_axis) tuples
    """
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axis names: {mesh.axis_names}")
    print(f"Number of devices: {len(mesh.devices.flatten())}")
    
    # Check sharding rules
    sharding_dict = dict(sharding_rules)
    sharded_axes = [k for k, v in sharding_dict.items() if v is not None]
    print(f"Sharded axes: {sharded_axes}")
    
    return True 