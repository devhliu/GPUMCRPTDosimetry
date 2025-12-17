from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class SortedVoxelTallyConfig:
    enabled: bool = False
    tile_shift: int = 9
    min_events: int = 200_000


@torch.no_grad()
def sorted_voxel_accumulate(
    edep_flat: torch.Tensor,
    lin_i32: torch.Tensor,
    val_f32: torch.Tensor,
    cfg: SortedVoxelTallyConfig,
) -> None:
    good = lin_i32 >= 0
    if not good.any():
        return

    lin = lin_i32[good].to(torch.int64)
    val = val_f32[good].to(torch.float32)

    # fast path: no sorting
    if (not cfg.enabled) or (lin.numel() < cfg.min_events):
        edep_flat.index_add_(0, lin, val)
        return

    # key for optional tile grouping then voxel grouping
    if cfg.tile_shift > 0:
        tile = (lin >> cfg.tile_shift).to(torch.int64)
        key = (tile << 32) | (lin & 0xFFFFFFFF)
    else:
        key = lin

    order = torch.argsort(key)
    lin_s = lin[order]
    val_s = val[order]

    start = torch.ones((lin_s.numel(),), device=lin_s.device, dtype=torch.bool)
    start[1:] = lin_s[1:] != lin_s[:-1]
    seg = torch.cumsum(start.to(torch.int64), dim=0) - 1
    nseg = int(seg[-1].item()) + 1

    seg_sum = torch.zeros((nseg,), device=val_s.device, dtype=val_s.dtype)
    seg_sum.index_add_(0, seg, val_s)

    seg_lin = lin_s[start]
    edep_flat.index_add_(0, seg_lin, seg_sum)