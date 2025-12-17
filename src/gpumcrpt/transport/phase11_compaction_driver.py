from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import triton

from gpumcrpt.kernels.triton.compaction_scan import status_to_i32_mask_kernel, scatter_compact_index_kernel


@dataclass
class CompactionWorkspace:
    mask_i32: torch.Tensor
    prefix_i32: torch.Tensor
    out_indices: torch.Tensor


def allocate_compaction_ws(device: torch.device, max_n: int) -> CompactionWorkspace:
    return CompactionWorkspace(
        mask_i32=torch.empty((max_n,), device=device, dtype=torch.int32),
        prefix_i32=torch.empty((max_n,), device=device, dtype=torch.int32),
        out_indices=torch.empty((max_n,), device=device, dtype=torch.int32),
    )


@torch.no_grad()
def build_active_indices_from_status(
    status: torch.Tensor,
    n: int,
    ws: CompactionWorkspace,
    *,
    exclusive_scan_int32,      # hook to your existing scan (CUB or Triton)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      active_indices (int32 view of length n_alive)
      n_alive (scalar int32 tensor on GPU if your scan returns it; otherwise return as python int)
    """
    grid = (triton.cdiv(n, 256),)
    status_to_i32_mask_kernel[grid](status, ws.mask_i32, n=n, BLOCK=256, num_warps=4)

    # prefix_i32 = exclusive scan(mask_i32)
    # This must be your existing GPU scan. It should also optionally return total sum.
    total_alive = exclusive_scan_int32(ws.mask_i32, ws.prefix_i32, n=n)

    scatter_compact_index_kernel[grid](ws.mask_i32, ws.prefix_i32, ws.out_indices, n=n, BLOCK=256, num_warps=4)

    # active indices are first total_alive entries
    return ws.out_indices, total_alive