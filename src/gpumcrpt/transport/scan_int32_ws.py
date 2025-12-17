from __future__ import annotations

from dataclasses import dataclass
import torch

from gpumcrpt.transport.triton.scan_int32 import (
    exclusive_scan_int32_block_kernel,
    add_block_offsets_kernel,
)


@dataclass
class Int32ScanWorkspace:
    """
    Workspace for exclusive scan of a fixed-length int32 array `n` with fixed power-of-two `block`.
    """
    n: int
    block: int
    out: torch.Tensor            # int32 [n]
    block_sums: torch.Tensor     # int32 [n_blocks]
    block_offsets: torch.Tensor  # int32 [n_blocks]
    tmp: torch.Tensor            # int32 [1]

    @staticmethod
    def allocate(device: torch.device | str, n: int, block: int = 1024) -> "Int32ScanWorkspace":
        device = torch.device(device)
        n_blocks = (n + block - 1) // block
        return Int32ScanWorkspace(
            n=int(n),
            block=int(block),
            out=torch.empty((n,), device=device, dtype=torch.int32),
            block_sums=torch.empty((n_blocks,), device=device, dtype=torch.int32),
            block_offsets=torch.empty((n_blocks,), device=device, dtype=torch.int32),
            tmp=torch.empty((1,), device=device, dtype=torch.int32),
        )


@torch.no_grad()
def exclusive_scan_int32_into(
    x: torch.Tensor,
    ws: Int32ScanWorkspace,
    *,
    out: torch.Tensor | None = None,
    num_warps: int = 4,
) -> torch.Tensor:
    """
    Exclusive scan into a preallocated output buffer.
    Returns the output tensor.

    x: int32 [n] on CUDA
    out: optional int32 [n]; if not provided, uses ws.out
    """
    assert x.is_cuda and x.dtype == torch.int32
    assert x.numel() == ws.n

    y = ws.out if out is None else out
    assert y.is_cuda and y.dtype == torch.int32 and y.numel() == ws.n

    n = ws.n
    block = ws.block
    n_blocks = ws.block_sums.numel()

    grid = (n_blocks,)
    exclusive_scan_int32_block_kernel[grid](
        x, y, ws.block_sums,
        n=n,
        BLOCK=block,
        num_warps=num_warps,
    )

    if n_blocks > 1:
        if n_blocks <= block:
            exclusive_scan_int32_block_kernel[(1,)](
                ws.block_sums, ws.block_offsets, ws.tmp,
                n=n_blocks,
                BLOCK=block,
                num_warps=num_warps,
            )
        else:
            # Rare for our expected n_bins, but keep correct:
            sub_ws = Int32ScanWorkspace.allocate(x.device, n_blocks, block=block)
            exclusive_scan_int32_into(ws.block_sums, sub_ws, out=ws.block_offsets, num_warps=num_warps)

        add_block_offsets_kernel[grid](
            y, ws.block_offsets,
            n=n,
            BLOCK=block,
            num_warps=num_warps,
        )

    return y