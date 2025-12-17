from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def edep_tile_accum_kernel(
    lin_ptr, val_ptr,
    out_lin_ptr, out_val_ptr,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Reduce (lin, val) pairs within each program where lin is identical.
    This is a *local* reduction; global reduction still requires either sorting
    or a second pass. Intended as a building block.

    Assumption for effectiveness:
    - input (lin_ptr) is already grouped by lin (e.g., after sorting by tile/voxel)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n

    lin = tl.load(lin_ptr + offs, mask=m, other=-1).to(tl.int32)
    val = tl.load(val_ptr + offs, mask=m, other=0.0).to(tl.float32)

    # Identify segment starts (lin changes)
    lin_prev = tl.cat([tl.full([1], -2, tl.int32), lin[:-1]], axis=0)
    is_start = lin != lin_prev
    # Keep only one representative per segment for output
    # Compute segment id via cumsum
    seg_id = tl.cumsum(is_start.to(tl.int32), axis=0) - 1

    # For each segment, sum values (naive O(BLOCK^2) not acceptable)
    # Triton does not provide segmented reduction primitive directly.
    # Therefore, this kernel is a placeholder and not used by default.
    # Phase 7.x performance path will instead use:
    #   - sort by lin, then torch.segment_reduce or custom CUB-like reduce
    # until a fast segmented reduction kernel is implemented.

    tl.store(out_lin_ptr + offs, lin, mask=m)
    tl.store(out_val_ptr + offs, val, mask=m)