from __future__ import annotations

from dataclasses import dataclass
import torch
import triton

from gpumcrpt.kernels.triton.compaction_lazy_scan import (
    make_alive_mask_i32_kernel,
    scatter_indices_from_prefix_kernel,
)
from gpumcrpt.kernels.triton.compaction_pack_guarded import (
    pack_particlebank_guarded_kernel,
    pack_vacancybank_guarded_kernel,
)

# Triton scan workspace (NOT torch.cumsum)
from gpumcrpt.transport.scan_int32_ws import Int32ScanWorkspace, exclusive_scan_int32_into


@dataclass
class LazyCompactionWS:
    mask_i32: torch.Tensor
    prefix_i32: torch.Tensor
    idx_i32: torch.Tensor
    scan_ws: Int32ScanWorkspace
    total_i32: torch.Tensor  # int32[1]


def alloc_lazy_compaction_ws(device: torch.device, cap: int, scan_block: int = 1024) -> LazyCompactionWS:
    return LazyCompactionWS(
        mask_i32=torch.empty((cap,), device=device, dtype=torch.int32),
        prefix_i32=torch.empty((cap,), device=device, dtype=torch.int32),
        idx_i32=torch.empty((cap,), device=device, dtype=torch.int32),
        scan_ws=Int32ScanWorkspace.allocate(device, cap, block=scan_block),
        total_i32=torch.zeros((1,), device=device, dtype=torch.int32),
    )


@torch.no_grad()
def lazy_compact_particlebank_pingpong(
    *,
    src, dst,
    n_dirty: int,                      # python int from single sync at step start
    global_counters: torch.Tensor,      # int32[>=3]
    counter_idx: int,                  # 0 photon, 1 electron
    ws: LazyCompactionWS,
    block: int = 256,
    num_warps: int = 4,
) -> None:
    """
    Lazy compaction using Triton scan workspace:
      1) mask = (status==1) for [0:n_dirty]
      2) prefix = exclusive_scan(mask)
      3) total = prefix[n-1] + mask[n-1] (GPU scalar)
      4) idx_list[prefix[i]] = i for keep lanes
      5) pack dst[0:total) using idx_list, guarded by total read on GPU
      6) global_counters[counter_idx] = total (GPU, no sync)
    """
    if n_dirty <= 0:
        global_counters[counter_idx].zero_()
        return

    n = int(n_dirty)
    grid = (triton.cdiv(n, block),)

    make_alive_mask_i32_kernel[grid](src.status, ws.mask_i32, n=n, BLOCK=block, num_warps=num_warps)

    # EXCLUSIVE SCAN (Triton workspace)
    exclusive_scan_int32_into(ws.mask_i32[:n], ws.scan_ws, out=ws.prefix_i32[:n], num_warps=num_warps)

    # total alive on GPU scalar
    # total = prefix[n-1] + mask[n-1]
    ws.total_i32[0] = ws.prefix_i32[n - 1] + ws.mask_i32[n - 1]

    # build compacted index list
    scatter_indices_from_prefix_kernel[grid](ws.mask_i32, ws.prefix_i32, ws.idx_i32, n=n, BLOCK=block, num_warps=num_warps)

    # pack dst (oversubscribed by n_dirty, guarded by total_i32 inside kernel)
    pack_particlebank_guarded_kernel[grid](
        src.x, src.y, src.z,
        src.dx, src.dy, src.dz,
        src.E, src.w, src.ebin,
        src.rng_key0, src.rng_key1, src.rng_ctr0, src.rng_ctr1, src.rng_ctr2, src.rng_ctr3,
        dst.x, dst.y, dst.z,
        dst.dx, dst.dy, dst.dz,
        dst.E, dst.w, dst.ebin,
        dst.rng_key0, dst.rng_key1, dst.rng_ctr0, dst.rng_ctr1, dst.rng_ctr2, dst.rng_ctr3,
        dst.status,
        ws.idx_i32,
        ws.total_i32,
        n_dirty=n,
        BLOCK=block,
        num_warps=num_warps,
    )

    # update counter on GPU
    global_counters[counter_idx] = ws.total_i32[0]


@torch.no_grad()
def lazy_compact_vacancybank_pingpong(
    *,
    src, dst,
    n_dirty: int,
    global_counters: torch.Tensor,
    counter_idx: int,                  # 2 for vacancies
    ws: LazyCompactionWS,
    block: int = 256,
    num_warps: int = 4,
) -> None:
    if n_dirty <= 0:
        global_counters[counter_idx].zero_()
        return

    n = int(n_dirty)
    grid = (triton.cdiv(n, block),)

    make_alive_mask_i32_kernel[grid](src.status, ws.mask_i32, n=n, BLOCK=block, num_warps=num_warps)
    exclusive_scan_int32_into(ws.mask_i32[:n], ws.scan_ws, out=ws.prefix_i32[:n], num_warps=num_warps)
    ws.total_i32[0] = ws.prefix_i32[n - 1] + ws.mask_i32[n - 1]
    scatter_indices_from_prefix_kernel[grid](ws.mask_i32, ws.prefix_i32, ws.idx_i32, n=n, BLOCK=block, num_warps=num_warps)

    pack_vacancybank_guarded_kernel[grid](
        src.x, src.y, src.z,
        src.atom_Z, src.shell_idx, src.w,
        src.rng_key0, src.rng_key1, src.rng_ctr0, src.rng_ctr1, src.rng_ctr2, src.rng_ctr3,
        dst.x, dst.y, dst.z,
        dst.atom_Z, dst.shell_idx, dst.w,
        dst.rng_key0, dst.rng_key1, dst.rng_ctr0, dst.rng_ctr1, dst.rng_ctr2, dst.rng_ctr3,
        dst.status,
        ws.idx_i32,
        ws.total_i32,
        n_dirty=n,
        BLOCK=block,
        num_warps=num_warps,
    )

    global_counters[counter_idx] = ws.total_i32[0]