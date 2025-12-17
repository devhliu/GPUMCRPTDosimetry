from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def reduce_max_int32_kernel(
    x_ptr, out_ptr,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.int32)
    mx = tl.max(x, axis=0)
    tl.store(out_ptr + pid, mx, mask=True)