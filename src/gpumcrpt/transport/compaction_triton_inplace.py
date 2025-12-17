from __future__ import annotations

from dataclasses import dataclass
import torch
import triton

from gpumcrpt.kernels.triton.compact_inplace_soa import (
    status_to_mask_i32_kernel,
    scatter_pack_particlebank_soa_kernel,
    scatter_pack_vacancybank_soa_kernel,
    last_prefix_plus_last_mask_kernel,
)

from gpumcrpt.transport.scan_int32_ws import Int32ScanWorkspace, exclusive_scan_int32_into


@dataclass
class CompactionWS:
    mask_i32: torch.Tensor
    prefix_i32: torch.Tensor
    scan_ws: Int32ScanWorkspace
    total_i32: torch.Tensor  # int32[1] on GPU


def alloc_compaction_ws(device: torch.device, cap: int, scan_block: int = 1024) -> CompactionWS:
    return CompactionWS(
        mask_i32=torch.empty((cap,), device=device, dtype=torch.int32),
        prefix_i32=torch.empty((cap,), device=device, dtype=torch.int32),
        scan_ws=Int32ScanWorkspace.allocate(device, cap, block=scan_block),
        total_i32=torch.zeros((1,), device=device, dtype=torch.int32),
    )


@torch.no_grad()
def compact_particle_bank_pingpong_count_sync(
    *,
    src, dst,
    count_i32: torch.Tensor,     # self.global_counters[idx] view (1-element tensor preferred)
    ws: CompactionWS,
    scan_num_warps: int = 4,
) -> int:
    """
    Compaction style: inplace_pack (ping-pong buffers).
    Scan only current count N with ONE CPU sync to fetch N.
    Returns new_count as python int (from ONE CPU sync at end).
    """
    # --- SINGLE CPU SYNC (your preference) ---
    n = int(count_i32.item())
    if n == 0:
        return 0

    grid = (triton.cdiv(n, 256),)
    status_to_mask_i32_kernel[grid](src.status, ws.mask_i32, n=n, BLOCK=256, num_warps=4)

    # prefix scan (GPU)
    exclusive_scan_int32_into(ws.mask_i32[:n], ws.scan_ws, out=ws.prefix_i32[:n], num_warps=scan_num_warps)

    # compute total alive on GPU scalar: prefix[n-1] + mask[n-1]
    last_prefix_plus_last_mask_kernel[(1,)](ws.mask_i32, ws.prefix_i32, ws.total_i32, n=n)

    # scatter pack all fields
    scatter_pack_particlebank_soa_kernel[grid](
        src.x, src.y, src.z,
        src.dx, src.dy, src.dz,
        src.E, src.w, src.ebin,
        src.rng_key0, src.rng_key1, src.rng_ctr0, src.rng_ctr1, src.rng_ctr2, src.rng_ctr3,
        src.status,
        dst.x, dst.y, dst.z,
        dst.dx, dst.dy, dst.dz,
        dst.E, dst.w, dst.ebin,
        dst.rng_key0, dst.rng_key1, dst.rng_ctr0, dst.rng_ctr1, dst.rng_ctr2, dst.rng_ctr3,
        dst.status,
        ws.mask_i32, ws.prefix_i32,
        n=n,
        BLOCK=256,
        num_warps=4,
    )

    # --- SINGLE CPU SYNC (same one, if you want to count only once overall you can delay reading until end-of-step) ---
    new_n = int(ws.total_i32.item())
    count_i32.fill_(new_n)
    return new_n


@torch.no_grad()
def compact_vacancy_bank_pingpong_count_sync(
    *,
    src, dst,
    count_i32: torch.Tensor,
    ws: CompactionWS,
    scan_num_warps: int = 4,
) -> int:
    n = int(count_i32.item())
    if n == 0:
        return 0

    grid = (triton.cdiv(n, 256),)
    status_to_mask_i32_kernel[grid](src.status, ws.mask_i32, n=n, BLOCK=256, num_warps=4)

    exclusive_scan_int32_into(ws.mask_i32[:n], ws.scan_ws, out=ws.prefix_i32[:n], num_warps=scan_num_warps)
    last_prefix_plus_last_mask_kernel[(1,)](ws.mask_i32, ws.prefix_i32, ws.total_i32, n=n)

    scatter_pack_vacancybank_soa_kernel[grid](
        src.x, src.y, src.z,
        src.atom_Z, src.shell_idx, src.w,
        src.rng_key0, src.rng_key1, src.rng_ctr0, src.rng_ctr1, src.rng_ctr2, src.rng_ctr3,
        src.status,
        dst.x, dst.y, dst.z,
        dst.atom_Z, dst.shell_idx, dst.w,
        dst.rng_key0, dst.rng_key1, dst.rng_ctr0, dst.rng_ctr1, dst.rng_ctr2, dst.rng_ctr3,
        dst.status,
        ws.mask_i32, ws.prefix_i32,
        n=n,
        BLOCK=256,
        num_warps=4,
    )

    new_n = int(ws.total_i32.item())
    count_i32.fill_(new_n)
    return new_n