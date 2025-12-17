from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import triton

from gpumcrpt.transport.triton.hashed_tile_tally import (
    hist_hashed_bins_kernel,
    scatter_hashed_bins_kernel,
)
from gpumcrpt.transport.triton.hashed_tile_tally_r1 import reduce_bins_hash_active_kernel_r1


@dataclass
class HashedTileTallyR1Config:
    enabled: bool = True
    auto: bool = True
    tile_shift: int = 9

    n_bins_pow2: int = 131072
    min_events: int = 200_000

    block: int = 256          # must equal hash_H
    hash_H: int = 256
    hash_probes: int = 8
    num_warps: int = 4

    diagnostics: bool = False
    fallback_on_fail: bool = True


class HashedTileTallyWorkspace:
    def __init__(self):
        self._cache: Dict[Tuple[torch.device, int, int], Dict[str, torch.Tensor]] = {}

    def get(self, device: torch.device | str, n_bins: int, capacity: int, active_cap: int) -> Dict[str, torch.Tensor]:
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

            # diagnostics buffers sized to max active bins (worst-case n_bins, but we store per A each call)
            "miss0": torch.empty((n_bins,), device=device, dtype=torch.int32),
            "fail": torch.empty((n_bins,), device=device, dtype=torch.int32),
        }
        self._cache[key] = ws
        return ws


@torch.no_grad()
def hashed_tile_accumulate_r1(
    edep_flat: torch.Tensor,
    lin_i32: torch.Tensor,
    val_f32: torch.Tensor,
    cfg: HashedTileTallyR1Config,
    ws: HashedTileTallyWorkspace,
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
    if (n_bins & (n_bins - 1)) != 0:
        raise ValueError("n_bins_pow2 must be power of two")
    bin_mask = n_bins - 1

    H = int(cfg.hash_H)
    B = int(cfg.block)
    if H != B:
        raise ValueError("R1 requires hash_H == block (e.g., 256)")
    if (H & (H - 1)) != 0:
        raise ValueError("hash_H must be power of two")

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

    # 2) active bins
    active_bins = torch.nonzero(w["bin_counts"], as_tuple=False).flatten().to(torch.int32)
    A = int(active_bins.numel())
    if A == 0:
        return

    # 3) offsets
    c = torch.cumsum(w["bin_counts"], dim=0)
    w["bin_offsets"][0] = 0
    w["bin_offsets"][1:] = c[:-1]
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

    # 5) reduce active bins (R1)
    if cfg.diagnostics:
        miss0 = w["miss0"][:A]
        fail = w["fail"][:A]
    else:
        # dummy minimal tensors (avoids extra alloc); kernel writes but we ignore
        miss0 = w["miss0"][:A]
        fail = w["fail"][:A]

    grid2 = (A,)
    reduce_bins_hash_active_kernel_r1[grid2](
        active_bins,
        w["bin_offsets"],
        w["bin_counts"],
        w["out_lin"], w["out_val"],
        edep_flat,
        fail,
        miss0,
        A=A,
        H=H,
        PROBES=int(cfg.hash_probes),
        num_warps=int(cfg.num_warps),
    )

    # Optional safety fallback: if any fail occurred, fall back to index_add for correctness
    # (fail should be zero if probes are enough, but this guards against pathological collisions)
    if cfg.fallback_on_fail:
        if int(fail.max().item()) > 0:
            edep_flat.index_add_(0, lin.to(torch.int64), val)