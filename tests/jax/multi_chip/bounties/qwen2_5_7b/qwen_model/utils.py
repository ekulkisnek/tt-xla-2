# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
from jax import Array
from jax.lax import with_sharding_constraint
from flax.typing import Path
from jax import ShapeDtypeStruct
import jax

def timer(name: str):
    start = None
    """Context manager for timing."""
    def enter():
        nonlocal start
        start = time.monotonic()
        return None

    def exit():
        end = time.monotonic()
        print(f"{name}: {(end - start) * 1000:0.3f} ms")

    return enter, exit

def keystr_simple(path: Path, separator: str = ".") -> str:
    def get_key(p):
        if hasattr(p, 'key'):
            return p.key
        elif hasattr(p, 'name'):
            return p.name
        elif hasattr(p, 'idx'):
            return str(p.idx)
        else:
            return str(p)
    return separator.join(str(get_key(p)) for p in path)

def update_sharding(abs_array: ShapeDtypeStruct, sharding) -> ShapeDtypeStruct:
    """Update sharding on a ShapeDtypeStruct.

    Equivalent to `a.update(sharding=sharding)` on newer Jax versions.
    """
    return ShapeDtypeStruct(
        shape=abs_array.shape,
        dtype=abs_array.dtype,
        sharding=sharding,
        weak_type=abs_array.weak_type,
    ) 