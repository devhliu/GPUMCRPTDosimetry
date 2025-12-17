from __future__ import annotations

from dataclasses import dataclass
import torch
import triton

from gpumcrpt.transport.triton.hashed_tile_tally import (
    hist_hashed_bins_kernel,
    scatter_hashed_bins_kernel,
)
from gpumcrpt.transport.triton.hashed_tile_tally_r1 import reduce_bins_hash_active_kernel_r1
from gpumcrpt.transport.triton.compaction import prefix_sum_compact
from gpumcrpt.transport.scan_int32 import exclusive_scan_int32


@dataclass
class HashedTileTallyR1Config:
    enabled: bool = True
    auto: bool = True
    tile_shift: int = 9

    n_bins_pow2: int = 131072
    min_events: int = 200_000

    block: int = 256
    hash_H: int = 256
    hash_probes: int = 8
    num_warps: int = 4


@torch.no_grad()
def hashed_tile_accumulate_r1_no_torch_nonzero_cumsum(
    edep_flat: torch.Tensor,
    lin_i32: torch.Tensor,
    val_f32: torch.Tensor,
    cfg: HashedTileTallyR1Config,
    ws,
) -> None:
    if lin_i32.numel() == 0:
        return
    good = lin_i32 >= 0
    if not good.any():
        return

    lin = lin_i32[good].to(torch.int32)
    val = val_f32[good].to(torch.float32)
    N = int(lin.numel())

    use_hashed = cfg.enabled and ((not cfg.auto) or (N >= int(cfg.min_events)))
    if not use_hashed:
        edep_flat.index_add_(0, lin.to(torch.int64), val)
        return

    n_bins = int(cfg.n_bins_pow2)
    bin_mask = n_bins - 1
    B = int(cfg.block)
    H = int(cfg.hash_H)
    if H != B:
        raise ValueError("R1 requires hash_H == block (e.g. 256)")

    w = ws.get(edep_flat.device, n_bins=n_bins, capacity=N, active_cap=n_bins)

    # 1) histogram
    w["bin_counts"].zero_()
    grid = (triton.cdiv(N, B),)
    hist_hashed_bins_kernel[grid](
        lin,
        w["bin_counts"],
        n=N,
        tile_shift=int(cfg.tile_shift),
        bin_mask=bin_mask,
        BLOCK=B,
        num_warps=int(cfg.num_warps),
    )

    # 2) active bins via compaction
    active_mask = w["bin_counts"] > 0
    active_bins_i64, A = prefix_sum_compact(active_mask)
    A = int(A.item())  # sync; see note below

    # NOTE: To avoid sync, rely on active_bins_i64.numel() instead if compact returns trimmed tensor.
    # If your prefix_sum_compact returns fixed-size buffer + count, we can make it return trimmed view.
    active_bins = active_bins_i64[:A].to(torch.int32)
    if A == 0:
        return

    # 3) offsets via exclusive scan kernel (no torch.cumsum)
    w["bin_offsets"].copy_(exclusive_scan_int32(w["bin_counts"].to(torch.int32), block=1024))
    w["bin_cursor"].copy_(w["bin_offsets"])

    # 4) scatter
    scatter_hashed_bins_kernel[grid](
        lin, val,
        w["bin_cursor"],
        w["out_lin"], w["out_val"],
        n=N,
        tile_shift=int(cfg.tile_shift),
        bin_mask=bin_mask,
        BLOCK=B,
        num_warps=int(cfg.num_warps),
    )

    # 5) reduce active
    reduce_bins_hash_active_kernel_r1[(A,)](
        active_bins,
        w["bin_offsets"],
        w["bin_counts"],
        w["out_lin"], w["out_val"],
        edep_flat,
        w["fail"][:A],
        w["miss0"][:A],
        A=A,
        H=H,
        PROBES=int(cfg.hash_probes),
        num_warps=int(cfg.num_warps),
    )