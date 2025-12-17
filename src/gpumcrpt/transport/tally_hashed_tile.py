from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import triton

from gpumcrpt.transport.triton.hashed_tile_tally import (
    hist_hashed_bins_kernel,
    scatter_hashed_bins_kernel,
    reduce_bins_hash_active_kernel_r1,
)


@dataclass
class HashedTileTallyConfig:
    enabled: bool = False
    auto: bool = True
    tile_shift: int = 9

    n_bins_pow2: int = 131072
    min_events: int = 200_000

    hash_H: int = 256        # R1: equal to load_block
    hash_probes: int = 8
    block: int = 256
    num_warps: int = 4


class HashedTileTallyWorkspace:
    def __init__(self):
        self._cache: Dict[Tuple[torch.device, int, int], Dict[str, torch.Tensor]] = {}

    def get(self, device: torch.device | str, n_bins: int, capacity: int) -> Dict[str, torch.Tensor]:
        device = torch.device(device)
        key = (device, int(n_bins), int(capacity))
        if key in self._cache:
            return self._cache[key]

        ws = {
            "bin_counts": torch.zeros((n_bins,), device=device, dtype=torch.int32),
            "bin_offsets": torch.empty((n_bins,), device=device, dtype=torch.int32),
            "bin_cursor": torch.empty((n_bins,), device=device, dtype=torch.int32),
            "out_lin": torch.empty((capacity,), device=device, dtype=torch.int32),
            "out_val": torch.empty((capacity,), device=device, dtype=torch.float32),
        }
        self._cache[key] = ws
        return ws


@torch.no_grad()
def hashed_tile_accumulate(edep_flat, lin_i32, val_f32, cfg: HashedTileTallyConfig, ws: HashedTileTallyWorkspace | None = None):
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
    if (n_bins & (n_bins - 1)) != 0:
        raise ValueError("n_bins_pow2 must be power of 2")
    bin_mask = n_bins - 1

    H = int(cfg.hash_H)
    if (H & (H - 1)) != 0:
        raise ValueError("hash_H must be power of 2 for R1")
    # R1 requires H == block == 256 usually; we enforce H <= block and use H for chunking
    if H != int(cfg.block):
        raise ValueError("R1 requires hash_H == block (set both to 256 or both to 512)")

    if ws is None:
        ws = HashedTileTallyWorkspace()
    w = ws.get(edep_flat.device, n_bins=n_bins, capacity=N)

    w["bin_counts"].zero_()
    grid = (triton.cdiv(N, int(cfg.block)),)
    hist_hashed_bins_kernel[grid](
        lin,
        w["bin_counts"],
        n=N,
        tile_shift=int(cfg.tile_shift),
        bin_mask=bin_mask,
        BLOCK=int(cfg.block),
        num_warps=int(cfg.num_warps),
    )

    active_bins = torch.nonzero(w["bin_counts"], as_tuple=False).flatten().to(torch.int32)
    A = int(active_bins.numel())
    if A == 0:
        return

    c = torch.cumsum(w["bin_counts"], dim=0)
    w["bin_offsets"][0] = 0
    w["bin_offsets"][1:] = c[:-1]
    w["bin_cursor"].copy_(w["bin_offsets"])

    scatter_hashed_bins_kernel[grid](
        lin, val,
        w["bin_cursor"],
        w["out_lin"], w["out_val"],
        n=N,
        tile_shift=int(cfg.tile_shift),
        bin_mask=bin_mask,
        BLOCK=int(cfg.block),
        num_warps=int(cfg.num_warps),
    )

    grid2 = (A,)
    reduce_bins_hash_active_kernel_r1[grid2](
        active_bins,
        w["bin_offsets"],
        w["bin_counts"],
        w["out_lin"], w["out_val"],
        edep_flat,
        A=A,
        H=H,
        PROBES=int(cfg.hash_probes),
        num_warps=int(cfg.num_warps),
    )