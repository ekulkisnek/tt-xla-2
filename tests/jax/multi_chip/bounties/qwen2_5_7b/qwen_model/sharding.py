# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sharding configuration for Qwen2.5-7B tensor parallel execution.

This module provides mesh configurations and sharding rules for different
deployment scenarios including 2x4, 1x8, 1x32, and 8x4 mesh shapes.
"""

import logging
from typing import Dict, List, Tuple, Optional
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from enum import Enum

logger = logging.getLogger(__name__)


class Axis(str, Enum):
    """Logical axis names for tensor parallelism."""
    EMBED = "embed"
    MLP = "mlp" 
    HEAD = "head"
    QHEAD = "qhead"
    KVHEAD = "kvhead"
    VOCAB = "vocab"

    def __str__(self) -> str:
        return self.value


class MeshConfigurationError(Exception):
    """Error in device mesh setup."""
    pass


class ShardingError(Exception):
    """Error in tensor sharding configuration."""
    pass


def create_mesh_2x4() -> Mesh:
    """Create 2x4 mesh for tensor parallel execution.
    
    Returns:
        Mesh: 2x4 device mesh with ('data', 'model') axis names
        
    Raises:
        MeshConfigurationError: If insufficient devices available
    """
    devices = jax.devices()
    if len(devices) < 8:
        raise MeshConfigurationError(f"Need at least 8 devices for 2x4 mesh, got {len(devices)}")
    
    mesh_devices = jnp.array(devices[:8]).reshape(2, 4)
    logger.info(f"Created 2x4 mesh with devices: {[d.id for d in devices[:8]]}")
    return Mesh(mesh_devices, axis_names=("data", "model"))


def create_mesh_1x8() -> Mesh:
    """Create 1x8 mesh for tensor parallel execution.
    
    Returns:
        Mesh: 1x8 device mesh with ('batch', 'model') axis names
        
    Raises:
        MeshConfigurationError: If insufficient devices available
    """
    devices = jax.devices()
    if len(devices) < 8:
        raise MeshConfigurationError(f"Need at least 8 devices for 1x8 mesh, got {len(devices)}")
    
    mesh_devices = jnp.array(devices[:8]).reshape(1, 8)
    logger.info(f"Created 1x8 mesh with devices: {[d.id for d in devices[:8]]}")
    return Mesh(mesh_devices, axis_names=("batch", "model"))


def create_mesh_1x32() -> Mesh:
    """Create 1x32 mesh for large-scale tensor parallel execution.
    
    Returns:
        Mesh: 1x32 device mesh with ('batch', 'model') axis names
        
    Raises:
        MeshConfigurationError: If insufficient devices available
    """
    devices = jax.devices()
    if len(devices) < 32:
        raise MeshConfigurationError(f"Need at least 32 devices for 1x32 mesh, got {len(devices)}")
    
    mesh_devices = jnp.array(devices[:32]).reshape(1, 32)
    logger.info(f"Created 1x32 mesh with devices: {[d.id for d in devices[:32]]}")
    return Mesh(mesh_devices, axis_names=("batch", "model"))


def create_mesh_8x4() -> Mesh:
    """Create 8x4 mesh for data + tensor parallel execution.
    
    Returns:
        Mesh: 8x4 device mesh with ('data', 'model') axis names
        
    Raises:
        MeshConfigurationError: If insufficient devices available
    """
    devices = jax.devices()
    if len(devices) < 32:
        raise MeshConfigurationError(f"Need at least 32 devices for 8x4 mesh, got {len(devices)}")
    
    mesh_devices = jnp.array(devices[:32]).reshape(8, 4)
    logger.info(f"Created 8x4 mesh with devices: {[d.id for d in devices[:32]]}")
    return Mesh(mesh_devices, axis_names=("data", "model"))


# Sharding rules for different mesh configurations
SHARDING_RULES_2X4 = {
    Axis.EMBED: None,      # Replicate embeddings
    Axis.MLP: "model",     # Shard MLP across model axis
    Axis.HEAD: "model",    # Shard attention heads across model axis
    Axis.QHEAD: None,      # Replicate query heads
    Axis.KVHEAD: None,     # Replicate KV heads
    Axis.VOCAB: None,      # Replicate vocabulary
}

SHARDING_RULES_1X8 = {
    Axis.EMBED: None,      # Replicate embeddings
    Axis.MLP: "model",     # Shard MLP across model axis
    Axis.HEAD: "model",    # Shard attention heads across model axis
    Axis.QHEAD: None,      # Replicate query heads
    Axis.KVHEAD: None,     # Replicate KV heads
    Axis.VOCAB: None,      # Replicate vocabulary
}

SHARDING_RULES_1X32 = {
    Axis.EMBED: None,      # Replicate embeddings
    Axis.MLP: "model",     # Shard MLP across model axis
    Axis.HEAD: "model",    # Shard attention heads across model axis
    Axis.QHEAD: None,      # Replicate query heads
    Axis.KVHEAD: None,     # Replicate KV heads
    Axis.VOCAB: None,      # Replicate vocabulary
}

SHARDING_RULES_8X4 = {
    Axis.EMBED: None,      # Replicate embeddings
    Axis.MLP: "model",     # Shard MLP across model axis
    Axis.HEAD: "model",    # Shard attention heads across model axis
    Axis.QHEAD: None,      # Replicate query heads
    Axis.KVHEAD: None,     # Replicate KV heads
    Axis.VOCAB: None,      # Replicate vocabulary
}


def get_optimal_sharding_rules(mesh_shape: Tuple[int, ...]) -> Dict[Axis, Optional[str]]:
    """Get optimal sharding rules based on mesh configuration.
    
    Args:
        mesh_shape: Shape of the device mesh
        
    Returns:
        Dictionary mapping axis names to mesh axis names
        
    Raises:
        ShardingError: If mesh shape is not supported
    """
    if mesh_shape == (2, 4):
        return SHARDING_RULES_2X4
    elif mesh_shape == (1, 8):
        return SHARDING_RULES_1X8
    elif mesh_shape == (1, 32):
        return SHARDING_RULES_1X32
    elif mesh_shape == (8, 4):
        return SHARDING_RULES_8X4
    else:
        raise ShardingError(f"Unsupported mesh shape: {mesh_shape}")


def validate_mesh_configuration(mesh: Mesh, sharding_rules: Dict[Axis, Optional[str]]) -> None:
    """Validate mesh and sharding compatibility.
    
    Args:
        mesh: Device mesh
        sharding_rules: Mapping of logical to mesh axis names
        
    Raises:
        MeshConfigurationError: If mesh and sharding are incompatible
    """
    # Check if sharding rules reference valid mesh axes
    mesh_axis_names = set(mesh.axis_names)
    used_axes = {v for v in sharding_rules.values() if v is not None}
    
    invalid_axes = used_axes - mesh_axis_names
    if invalid_axes:
        raise MeshConfigurationError(
            f"Sharding rules reference invalid mesh axes: {invalid_axes}. "
            f"Available axes: {mesh_axis_names}"
        )
    
    # Check if tensor parallel degree matches mesh size
    model_axis_size = mesh.shape[mesh.axis_names.index("model")] if "model" in mesh.axis_names else 1
    sharding_factor = len([v for v in sharding_rules.values() if v == "model"])
    
    if sharding_factor > 0 and model_axis_size < 2:
        raise MeshConfigurationError(
            f"Tensor parallel sharding requested but model axis size is {model_axis_size}"
        )
    
    logger.info(f"Validated mesh {mesh.shape} with sharding rules: {sharding_rules}")


def create_mesh(mesh_type: str) -> Tuple[Mesh, Dict[Axis, Optional[str]]]:
    """Create mesh and sharding rules for specified configuration.
    
    Args:
        mesh_type: One of '2x4', '1x8', '1x32', '8x4'
        
    Returns:
        Tuple of (mesh, sharding_rules)
        
    Raises:
        ValueError: If mesh_type is not supported
    """
    mesh_creators = {
        "2x4": (create_mesh_2x4, SHARDING_RULES_2X4),
        "1x8": (create_mesh_1x8, SHARDING_RULES_1X8),
        "1x32": (create_mesh_1x32, SHARDING_RULES_1X32),
        "8x4": (create_mesh_8x4, SHARDING_RULES_8X4),
    }
    
    if mesh_type not in mesh_creators:
        raise ValueError(f"Unsupported mesh type: {mesh_type}. Supported: {list(mesh_creators.keys())}")
    
    create_fn, sharding_rules = mesh_creators[mesh_type]
    mesh = create_fn()
    validate_mesh_configuration(mesh, sharding_rules)
    
    return mesh, sharding_rules


def setup_logging(level: int = logging.INFO) -> None:
    """Setup structured logging for sharding operations.
    
    Args:
        level: Logging level
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level) 