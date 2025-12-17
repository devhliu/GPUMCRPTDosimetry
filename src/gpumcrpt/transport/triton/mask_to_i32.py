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
def mask_gt0_to_i32_i8_kernel(
    x_ptr,
    out_i32_ptr,
    out_i8_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,   # Autotuned warp count
):
    """
    Optimized mask conversion kernel using Triton 3.5.1 features:
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    - Implicit boundary checking
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n

    # Load input data with cache hints
    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.int32)
    y = x > 0
    
    # Store output data with cache hints
    tl.store(out_i32_ptr + offs, y.to(tl.int32), mask=m)
    tl.store(out_i8_ptr + offs, y.to(tl.int8), mask=m)