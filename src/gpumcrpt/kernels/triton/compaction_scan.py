from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def status_to_i32_mask_kernel(status_ptr, mask_i32_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    """
    mask_i32[i] = 1 if status[i] != 0 else 0
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    s = tl.load(status_ptr + i, mask=m, other=0).to(tl.int8)
    alive = s != 0
    tl.store(mask_i32_ptr + i, alive.to(tl.int32), mask=m)


@triton.jit
def scatter_compact_index_kernel(mask_i32_ptr, prefix_ptr, out_indices_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    """
    Given:
      mask_i32[i] in {0,1}
      prefix[i] = exclusive scan of mask_i32
    Write:
      out_indices[prefix[i]] = i if mask_i32[i]==1
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    keep = tl.load(mask_i32_ptr + i, mask=m, other=0).to(tl.int32)
    p = tl.load(prefix_ptr + i, mask=m, other=0).to(tl.int32)
    tl.store(out_indices_ptr + p, i, mask=m & (keep == 1))