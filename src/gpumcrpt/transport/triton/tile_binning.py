from __future__ import annotations

import triton
import triton.language as tl


# Autotuning configurations for RTX A4000 (Ampere architecture)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'NUM_WARPS': 2}, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['n'],  # Tune based on problem size
)
@triton.jit
def scatter_by_tile_kernel(
    tile_ptr, lin_ptr, val_ptr,
    cursor_ptr,  # int32 per tile, initialized = bin_offsets
    out_lin_ptr, out_val_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,   # Autotuned warp count
):
    """
    Optimized tile binning kernel using Triton 3.5.1 features:
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    - Implicit boundary checking
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n

    # Load input data with cache hints
    tile = tl.load(tile_ptr + offs, mask=m, other=-1, cache_modifier=".cg").to(tl.int32)
    lin = tl.load(lin_ptr + offs, mask=m, other=-1, cache_modifier=".cg").to(tl.int32)
    val = tl.load(val_ptr + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)

    good = tile >= 0
    pos = tl.atomic_add(cursor_ptr + tile, 1, mask=good)
    
    # Store output data with cache hints
    tl.store(out_lin_ptr + pos, lin, mask=good, cache_modifier=".cg")
    tl.store(out_val_ptr + pos, val, mask=good, cache_modifier=".cg")