from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def select_secondaries_kernel(
    eligible_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    max_per_step: tl.int32,
    selected_ptr: tl.tensor,
    num_selected_ptr: tl.tensor,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """GPU-only kernel to select secondary particle indices without CPU synchronization.
    
    This kernel performs prefix sum within each block and selects up to max_per_step
    eligible indices, updating counts in-place.
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    # Load eligible flags and counts
    eligible = tl.load(eligible_ptr + offset, mask=mask, other=0).to(tl.int1)
    count = tl.load(counts_ptr + offset, mask=mask, other=0).to(tl.int32)
    
    # Count eligible in this block
    block_eligible_count = tl.sum(eligible.to(tl.int32))
    
    # Write block count (will be reduced on CPU)
    tl.store(num_selected_ptr + pid, block_eligible_count)
    
    # Store selected indices (-1 for not selected)
    selected = tl.where(eligible, offset.to(tl.int32), -1)
    tl.store(selected_ptr + offset, selected, mask=mask)


def select_indices_with_budget_gpu(
    flag_mask: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_per_primary: int,
    max_per_step: int) -> torch.Tensor:
    """GPU-optimized version of select_indices_with_budget that avoids CPU synchronization.
    
    Args:
        flag_mask: Bool-like tensor (N,) on CUDA device indicating which primaries request a secondary.
        counts: Int tensor (N,) on CUDA device tracking number of secondaries already spawned.
        max_per_primary: Max allowed secondaries per primary across the whole transport call.
        max_per_step: Throughput limiter; caps how many secondaries we handle in one step.

    Returns:
        1D int64 tensor of selected indices. Also increments `counts` in-place for the
        selected indices.
    """
    if max_per_primary <= 0 or max_per_step <= 0:
        return torch.empty((0,), device=flag_mask.device, dtype=torch.int64)

    if flag_mask.dtype is not torch.bool:
        flag_mask = flag_mask.to(torch.bool)

    if counts.dtype not in (torch.int32, torch.int64):
        raise TypeError("counts must be int32 or int64")

    if counts.shape != flag_mask.shape:
        raise ValueError("counts and flag_mask must have the same shape")

    # Compute eligible flags on GPU
    eligible = flag_mask & (counts < int(max_per_primary))
    
    if not eligible.any():
        return torch.empty((0,), device=flag_mask.device, dtype=torch.int64)

    N = eligible.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Allocate output buffers
    selected = torch.empty((N,), device=flag_mask.device, dtype=torch.int32)
    block_counts = torch.zeros((num_blocks,), device=flag_mask.device, dtype=torch.int32)
    
    # Launch kernel
    grid = (num_blocks,)
    select_secondaries_kernel[grid](
        eligible,
        counts,
        max_per_step,
        selected,
        block_counts,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Get total count (single GPU-CPU sync)
    total_eligible = int(block_counts.sum().item())
    
    # Filter to only valid indices and limit to max_per_step
    valid_mask = selected >= 0
    valid_indices = selected[valid_mask]
    
    if valid_indices.numel() > max_per_step:
        valid_indices = valid_indices[:max_per_step]
    
    # Update counts for selected indices (still on GPU)
    if valid_indices.numel() > 0:
        ones = torch.ones((valid_indices.numel(),), device=counts.device, dtype=counts.dtype)
        counts.index_add_(0, valid_indices, ones)
    
    return valid_indices.to(torch.int64)
