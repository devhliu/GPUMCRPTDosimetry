from __future__ import annotations

import torch
import triton

from gpumcrpt.transport.triton.scan_int32 import (
    exclusive_scan_int32_block_kernel,
    add_block_offsets_kernel,
)


@torch.no_grad()
def exclusive_scan_int32(x: torch.Tensor, *, block: int = 1024) -> torch.Tensor:
    """
    Exclusive scan for int32 tensor x on GPU.
    Returns y where y[i] = sum_{k<i} x[k].

    Constraints:
    - x must be int32 on CUDA
    - block must be power of two
    """
    assert x.is_cuda and x.dtype == torch.int32
    n = x.numel()
    y = torch.empty_like(x)

    n_blocks = (n + block - 1) // block
    block_sums = torch.empty((n_blocks,), device=x.device, dtype=torch.int32)

    grid = (n_blocks,)
    exclusive_scan_int32_block_kernel[grid](
        x, y, block_sums,
        n=n,
        BLOCK=block,
        num_warps=4,
    )

    if n_blocks > 1:
        # Recursively scan block_sums (n_blocks is typically small: e.g., 128 for 131072 bins with block=1024)
        if n_blocks <= block:
            block_offsets = torch.empty_like(block_sums)
            tmp = torch.empty((1,), device=x.device, dtype=torch.int32)
            exclusive_scan_int32_block_kernel[(1,)](
                block_sums, block_offsets, tmp,
                n=n_blocks,
                BLOCK=block,
                num_warps=4,
            )
        else:
            block_offsets = exclusive_scan_int32(block_sums, block=block)

        add_block_offsets_kernel[grid](
            y, block_offsets,
            n=n,
            BLOCK=block,
            num_warps=4,
        )

    return y