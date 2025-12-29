from __future__ import annotations

import torch


def allow_secondaries(*, secondary_depth: int, max_per_primary: int) -> bool:
    """Returns True when secondary spawning is enabled.

    This is used as a control-flow gate in GPU-native engines so we can unit-test
    config behavior without executing Triton kernels.
    """
    return int(secondary_depth) > 0 and int(max_per_primary) > 0


def select_indices_with_budget(
    flag_mask: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_per_primary: int,
    max_per_step: int) -> torch.Tensor:
    """Select indices to spawn secondaries, respecting a per-primary budget.

    This is device-agnostic (works on CPU/CUDA tensors) and uses GPU-optimized
    implementation for CUDA tensors to avoid CPU-GPU synchronization.

    Args:
        flag_mask: Bool-like tensor (N,) indicating which primaries request a secondary.
        counts: Int tensor (N,) tracking number of secondaries already spawned by each primary.
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

    eligible = flag_mask & (counts < int(max_per_primary))
    if not torch.any(eligible):
        return torch.empty((0,), device=flag_mask.device, dtype=torch.int64)

    N = eligible.shape[0]
    device = flag_mask.device
    
    if device.type == 'cuda':
        from gpumcrpt.transport.utils.secondary_budget_gpu import select_indices_with_budget_gpu
        return select_indices_with_budget_gpu(
            flag_mask, counts,
            max_per_primary=max_per_primary,
            max_per_step=max_per_step,
        )
    else:
        idx = torch.nonzero(eligible, as_tuple=False).flatten()
        if int(idx.numel()) > int(max_per_step):
            idx = idx[: int(max_per_step)]

        ns = int(idx.numel())
        if ns:
            ones = torch.ones((ns,), device=counts.device, dtype=counts.dtype)
            counts.index_add_(0, idx, ones)

    return idx
