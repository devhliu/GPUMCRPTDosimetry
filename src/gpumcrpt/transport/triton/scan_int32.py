from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def exclusive_scan_int32_block_kernel(
    x_ptr, out_ptr, block_sums_ptr,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Block-wise exclusive scan.
    out[pid*BLOCK + i] = sum_{k<i} x[pid*BLOCK + k]
    block_sums[pid] = sum_{k in block} x[k]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n

    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.int32)

    # inclusive scan via Hillis–Steele (BLOCK power-of-two)
    s = x
    step = 1
    while step < BLOCK:
        prev = tl.cat([tl.zeros((step,), tl.int32), s[:-step]], axis=0)
        s = s + prev
        step *= 2

    excl = s - x
    tl.store(out_ptr + offs, excl, mask=m)

    # block sum: inclusive last element (or 0 if block empty)
    # use the last element of s, masked
    last = tl.load(s + (BLOCK - 1), mask=True, other=0)  # scalar
    # If the block extends past n, last element may not be valid (x padded with 0); still ok.
    tl.store(block_sums_ptr + pid, last.to(tl.int32), mask=True)


@triton.jit
def add_block_offsets_kernel(
    out_ptr, block_offsets_ptr,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n

    base = tl.load(block_offsets_ptr + pid, mask=True, other=0).to(tl.int32)
    out = tl.load(out_ptr + offs, mask=m, other=0).to(tl.int32)
    tl.store(out_ptr + offs, out + base, mask=m)