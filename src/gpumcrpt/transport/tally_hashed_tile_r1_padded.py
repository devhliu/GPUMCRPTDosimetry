from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import triton

from gpumcrpt.transport.triton.hashed_tile_tally import (
    hist_hashed_bins_kernel,
    scatter_hashed_bins_kernel,
)
from gpumcrpt.transport.triton.active_bins_padded import (
    build_active_bins_padded_kernel,
    reduce_bins_hash_active_padded_kernel_r1,
)
from gpumcrpt.transport.scan_int32_ws import Int32ScanWorkspace, exclusive_scan_int32_into
from gpumcrpt.transport.triton.mask_to_i32 import mask_gt0_to_i32_i8_kernel


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

    scan_block: int = 1024
    scan_num_warps: int = 4


class HashedTileTallyWorkspace:
    def __init__(self):
        self._cache: Dict[Tuple[torch.device, int, int, int], Dict[str, object]] = {}

    def get(self, device: torch.device | str, n_bins: int, capacity: int, scan_block: int) -> Dict[str, object]:
        device = torch.device(device)
        key = (device, int(n_bins), int(capacity), int(scan_block))
        if key in self._cache:
            return self._cache[key]

        ws: Dict[str, object] = {
            "bin_counts": torch.zeros((n_bins,), device=device, dtype=torch.int32),
            "bin_offsets": torch.empty((n_bins,), device=device, dtype=torch.int32),
            "bin_cursor": torch.empty((n_bins,), device=device, dtype=torch.int32),

            "out_lin": torch.empty((capacity,), device=device, dtype=torch.int32),
            "out_val": torch.empty((capacity,), device=device, dtype=torch.float32),

            "active_bins": torch.empty((n_bins,), device=device, dtype=torch.int32),

            # Phase 8.4: keep both mask formats in workspace
            "active_mask_i32": torch.empty((n_bins,), device=device, dtype=torch.int32),
            "active_mask_i8": torch.empty((n_bins,), device=device, dtype=torch.int8),

            "scan_counts": Int32ScanWorkspace.allocate(device, n_bins, block=scan_block),
            "scan_mask": Int32ScanWorkspace.allocate(device, n_bins, block=scan_block),
        }
        self._cache[key] = ws
        return ws


@torch.no_grad()
def hashed_tile_accumulate_r1_padded(
    edep_flat: torch.Tensor,
    lin_i32: torch.Tensor,
    val_f32: torch.Tensor,
    cfg: HashedTileTallyR1Config,
    ws: HashedTileTallyWorkspace,
    *,
    capacity: int | None = None,
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

    H = int(cfg.hash_H)
    B = int(cfg.block)
    if H != B:
        raise ValueError("R1 requires hash_H == block")
    if N > (int(capacity) if capacity is not None else N):
        raise ValueError("N exceeds workspace capacity; pass bucket capacity")

    cap = int(capacity) if capacity is not None else N
    w = ws.get(edep_flat.device, n_bins=n_bins, capacity=cap, scan_block=int(cfg.scan_block))

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

    # 2) bin offsets scan
    exclusive_scan_int32_into(
        w["bin_counts"],
        w["scan_counts"],
        out=w["bin_offsets"],
        num_warps=int(cfg.scan_num_warps),
    )
    w["bin_cursor"].copy_(w["bin_offsets"])

    # 3) scatter
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

    # 4) build active mask + prefix + padded active bins list (no casts/allocs)
    grid_bins = (triton.cdiv(n_bins, 256),)
    mask_gt0_to_i32_i8_kernel[grid_bins](
        w["bin_counts"],
        w["active_mask_i32"],
        w["active_mask_i8"],
        n=n_bins,
        BLOCK=256,
        num_warps=4,
    )

    active_prefix = exclusive_scan_int32_into(
        w["active_mask_i32"],
        w["scan_mask"],
        out=w["scan_mask"].out,
        num_warps=int(cfg.scan_num_warps),
    )

    w["active_bins"].fill_(-1)
    build_active_bins_padded_kernel[grid_bins](
        w["active_mask_i8"], active_prefix,
        w["active_bins"],
        n=n_bins,
        BLOCK=256,
        num_warps=4,
    )

    # 5) reduce fixed grid
    reduce_bins_hash_active_padded_kernel_r1[(n_bins,)](
        w["active_bins"],
        w["bin_offsets"], w["bin_counts"],
        w["out_lin"], w["out_val"],
        edep_flat,
        n_bins=n_bins,
        H=H,
        PROBES=int(cfg.hash_probes),
        num_warps=int(cfg.num_warps),
    )