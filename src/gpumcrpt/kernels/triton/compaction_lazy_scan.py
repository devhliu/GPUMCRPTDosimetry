from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def make_alive_mask_i32_kernel(
    status_ptr,          # int8
    mask_ptr,            # int32
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    mask[i] = 1 if status[i] == 1 else 0
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    s = tl.load(status_ptr + i, mask=m, other=0).to(tl.int8)
    keep = s == 1
    tl.store(mask_ptr + i, keep.to(tl.int32), mask=m)


@triton.jit
def scatter_indices_from_prefix_kernel(
    mask_ptr,            # int32
    prefix_ptr,          # int32 exclusive prefix
    out_idx_ptr,         # int32 [n_alive]
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Writes compacted indices:
      if mask[i]==1: out_idx[prefix[i]] = i
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    keep = tl.load(mask_ptr + i, mask=m, other=0).to(tl.int32)
    p = tl.load(prefix_ptr + i, mask=m, other=0).to(tl.int32)
    tl.store(out_idx_ptr + p, i, mask=m & (keep == 1))


@triton.jit
def pack_particlebank_from_indices_kernel(
    # src bank
    sx_ptr, sy_ptr, sz_ptr,
    sdx_ptr, sdy_ptr, sdz_ptr,
    sE_ptr, sw_ptr, sebin_ptr,
    srk0_ptr, srk1_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,
    sstatus_ptr,

    # dst bank
    dx_ptr, dy_ptr, dz_ptr,
    ddx_ptr, ddy_ptr, ddz_ptr,
    dE_ptr, dw_ptr, debin_ptr,
    drk0_ptr, drk1_ptr, drc0_ptr, drc1_ptr, drc2_ptr, drc3_ptr,
    dstatus_ptr,

    # idx list
    idx_ptr,             # int32 [n_alive]
    n_alive: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < n_alive

    i = tl.load(idx_ptr + j, mask=m, other=0).to(tl.int32)

    x = tl.load(sx_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(sy_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(sz_ptr + i, mask=m, other=0.0).to(tl.float32)
    ux = tl.load(sdx_ptr + i, mask=m, other=0.0).to(tl.float32)
    uy = tl.load(sdy_ptr + i, mask=m, other=0.0).to(tl.float32)
    uz = tl.load(sdz_ptr + i, mask=m, other=0.0).to(tl.float32)
    E = tl.load(sE_ptr + i, mask=m, other=0.0).to(tl.float32)
    w = tl.load(sw_ptr + i, mask=m, other=0.0).to(tl.float32)
    ebin = tl.load(sebin_ptr + i, mask=m, other=0).to(tl.int32)
    k0 = tl.load(srk0_ptr + i, mask=m, other=0).to(tl.int32)
    k1 = tl.load(srk1_ptr + i, mask=m, other=0).to(tl.int32)
    c0 = tl.load(src0_ptr + i, mask=m, other=0).to(tl.int32)
    c1 = tl.load(src1_ptr + i, mask=m, other=0).to(tl.int32)
    c2 = tl.load(src2_ptr + i, mask=m, other=0).to(tl.int32)
    c3 = tl.load(src3_ptr + i, mask=m, other=0).to(tl.int32)

    tl.store(dx_ptr + j, x, mask=m)
    tl.store(dy_ptr + j, y, mask=m)
    tl.store(dz_ptr + j, z, mask=m)
    tl.store(ddx_ptr + j, ux, mask=m)
    tl.store(ddy_ptr + j, uy, mask=m)
    tl.store(ddz_ptr + j, uz, mask=m)
    tl.store(dE_ptr + j, E, mask=m)
    tl.store(dw_ptr + j, w, mask=m)
    tl.store(debin_ptr + j, ebin, mask=m)
    tl.store(drk0_ptr + j, k0, mask=m)
    tl.store(drk1_ptr + j, k1, mask=m)
    tl.store(drc0_ptr + j, c0, mask=m)
    tl.store(drc1_ptr + j, c1, mask=m)
    tl.store(drc2_ptr + j, c2, mask=m)
    tl.store(drc3_ptr + j, c3, mask=m)
    tl.store(dstatus_ptr + j, 1, mask=m)


@triton.jit
def pack_vacancybank_from_indices_kernel(
    sx_ptr, sy_ptr, sz_ptr,
    sZ_ptr, sshell_ptr, sw_ptr,
    srk0_ptr, srk1_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,
    sstatus_ptr,

    dx_ptr, dy_ptr, dz_ptr,
    dZ_ptr, dshell_ptr, dw_ptr,
    drk0_ptr, drk1_ptr, drc0_ptr, drc1_ptr, drc2_ptr, drc3_ptr,
    dstatus_ptr,

    idx_ptr,
    n_alive: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < n_alive
    i = tl.load(idx_ptr + j, mask=m, other=0).to(tl.int32)

    x = tl.load(sx_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(sy_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(sz_ptr + i, mask=m, other=0.0).to(tl.float32)
    Z = tl.load(sZ_ptr + i, mask=m, other=0).to(tl.int32)
    shell = tl.load(sshell_ptr + i, mask=m, other=0).to(tl.int32)
    w = tl.load(sw_ptr + i, mask=m, other=0.0).to(tl.float32)
    k0 = tl.load(srk0_ptr + i, mask=m, other=0).to(tl.int32)
    k1 = tl.load(srk1_ptr + i, mask=m, other=0).to(tl.int32)
    c0 = tl.load(src0_ptr + i, mask=m, other=0).to(tl.int32)
    c1 = tl.load(src1_ptr + i, mask=m, other=0).to(tl.int32)
    c2 = tl.load(src2_ptr + i, mask=m, other=0).to(tl.int32)
    c3 = tl.load(src3_ptr + i, mask=m, other=0).to(tl.int32)

    tl.store(dx_ptr + j, x, mask=m)
    tl.store(dy_ptr + j, y, mask=m)
    tl.store(dz_ptr + j, z, mask=m)
    tl.store(dZ_ptr + j, Z, mask=m)
    tl.store(dshell_ptr + j, shell, mask=m)
    tl.store(dw_ptr + j, w, mask=m)
    tl.store(drk0_ptr + j, k0, mask=m)
    tl.store(drk1_ptr + j, k1, mask=m)
    tl.store(drc0_ptr + j, c0, mask=m)
    tl.store(drc1_ptr + j, c1, mask=m)
    tl.store(drc2_ptr + j, c2, mask=m)
    tl.store(drc3_ptr + j, c3, mask=m)
    tl.store(dstatus_ptr + j, 1, mask=m)