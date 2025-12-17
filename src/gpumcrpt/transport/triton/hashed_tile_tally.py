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
def hist_hashed_bins_kernel(
    lin_ptr,
    bin_counts_ptr,
    n: tl.constexpr,
    tile_shift: tl.constexpr,
    bin_mask: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,   # Autotuned warp count
):
    """
    Optimized histogram kernel using Triton 3.5.1 features:
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    
    # Load input data with cache hints
    lin = tl.load(lin_ptr + offs, mask=m, other=-1, cache_modifier=".cg").to(tl.int32)
    good = lin >= 0
    tile = lin >> tile_shift
    b = tile & bin_mask
    
    # Atomic add with cache hint
    tl.atomic_add(bin_counts_ptr + b, 1, mask=good, cache_modifier=".cg")


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
def scatter_hashed_bins_kernel(
    lin_ptr, val_ptr,
    bin_cursor_ptr,
    out_lin_ptr, out_val_ptr,
    n: tl.constexpr,
    tile_shift: tl.constexpr,
    bin_mask: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,   # Autotuned warp count
):
    """
    Optimized scatter kernel using Triton 3.5.1 features:
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n

    # Load input data with cache hints
    lin = tl.load(lin_ptr + offs, mask=m, other=-1, cache_modifier=".cg").to(tl.int32)
    val = tl.load(val_ptr + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)

    good = lin >= 0
    tile = lin >> tile_shift
    b = tile & bin_mask

    # Atomic add and store with cache hints
    pos = tl.atomic_add(bin_cursor_ptr + b, 1, mask=good, cache_modifier=".cg")
    tl.store(out_lin_ptr + pos, lin, mask=good, cache_modifier=".cg")
    tl.store(out_val_ptr + pos, val, mask=good, cache_modifier=".cg")


@triton.jit
def reduce_bins_hash_active_kernel_r1(
    active_bins_ptr,      # int32[A]
    bin_offsets_ptr,      # int32[n_bins] exclusive offsets
    bin_counts_ptr,       # int32[n_bins]
    in_lin_ptr, in_val_ptr,
    edep_ptr,
    A: tl.constexpr,

    H: tl.constexpr,          # power-of-two; ALSO equals LOAD_BLOCK
    PROBES: tl.constexpr,     # bounded probes
):
    """
    R1 reducer: assumes H == LOAD_BLOCK.
    One program per active bin.

    Implementation idea:
    - Maintain slot keys/acc arrays size H.
    - Process events in chunks of H so memory loads are coalesced.
    - For each element, try to insert at slot s = lin&(H-1) with probing.

    This avoids O(H×B) broadcast compares.
    """
    pid = tl.program_id(0)
    if pid >= A:
        return

    b = tl.load(active_bins_ptr + pid).to(tl.int32)
    start = tl.load(bin_offsets_ptr + b).to(tl.int32)
    count = tl.load(bin_counts_ptr + b).to(tl.int32)
    end = start + count

    # local hash table
    keys = tl.full((H,), -1, tl.int32)
    acc = tl.zeros((H,), tl.float32)

    i = start
    while i < end:
        offs = i + tl.arange(0, H)
        m = offs < end

        lin = tl.load(in_lin_ptr + offs, mask=m, other=-1).to(tl.int32)
        val = tl.load(in_val_ptr + offs, mask=m, other=0.0).to(tl.float32)

        good = lin >= 0
        lin = tl.where(good, lin, -1)
        val = tl.where(good, val, 0.0)

        s = lin & (H - 1)
        done = lin == -1

        # bounded linear probe
        for _ in range(PROBES):
            k = tl.load(keys + s, mask=True, other=-1)
            empty = k == -1
            same = k == lin
            can = (~done) & (empty | same)

            # claim empty slots
            claim = can & empty
            keys = tl.where(claim, lin, keys)

            # add into acc[s] using a "scatter-add into vector" trick:
            # We update by adding val where index matches s.
            idx = tl.arange(0, H)
            acc = acc + tl.where(idx == s, tl.where(can, val, 0.0), 0.0)

            done = done | can
            s = (s + 1) & (H - 1)

        i += H

    # flush (H atomics per active bin worst-case; much less contention than per-step atomics)
    idx = tl.arange(0, H)
    k = tl.load(keys + idx, mask=True, other=-1).to(tl.int32)
    v = tl.load(acc + idx, mask=True, other=0.0).to(tl.float32)
    good = k >= 0
    tl.atomic_add(edep_ptr + k, v, mask=good)