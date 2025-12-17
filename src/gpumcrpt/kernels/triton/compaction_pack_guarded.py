from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def pack_particlebank_guarded_kernel(
    # src
    sx_ptr, sy_ptr, sz_ptr,
    sdx_ptr, sdy_ptr, sdz_ptr,
    sE_ptr, sw_ptr, sebin_ptr,
    srk0_ptr, srk1_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,

    # dst
    dx_ptr, dy_ptr, dz_ptr,
    ddx_ptr, ddy_ptr, ddz_ptr,
    dE_ptr, dw_ptr, debin_ptr,
    drk0_ptr, drk1_ptr, drc0_ptr, drc1_ptr, drc2_ptr, drc3_ptr,
    dstatus_ptr,

    idx_ptr,             # int32[cap]
    total_ptr,           # int32[1] total alive
    n_dirty: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < n_dirty

    total = tl.load(total_ptr + 0).to(tl.int32)
    do = m & (j < total)

    i = tl.load(idx_ptr + j, mask=do, other=0).to(tl.int32)

    x = tl.load(sx_ptr + i, mask=do, other=0.0).to(tl.float32)
    y = tl.load(sy_ptr + i, mask=do, other=0.0).to(tl.float32)
    z = tl.load(sz_ptr + i, mask=do, other=0.0).to(tl.float32)
    ux = tl.load(sdx_ptr + i, mask=do, other=0.0).to(tl.float32)
    uy = tl.load(sdy_ptr + i, mask=do, other=0.0).to(tl.float32)
    uz = tl.load(sdz_ptr + i, mask=do, other=0.0).to(tl.float32)
    E = tl.load(sE_ptr + i, mask=do, other=0.0).to(tl.float32)
    w = tl.load(sw_ptr + i, mask=do, other=0.0).to(tl.float32)
    ebin = tl.load(sebin_ptr + i, mask=do, other=0).to(tl.int32)
    k0 = tl.load(srk0_ptr + i, mask=do, other=0).to(tl.int32)
    k1 = tl.load(srk1_ptr + i, mask=do, other=0).to(tl.int32)
    c0 = tl.load(src0_ptr + i, mask=do, other=0).to(tl.int32)
    c1 = tl.load(src1_ptr + i, mask=do, other=0).to(tl.int32)
    c2 = tl.load(src2_ptr + i, mask=do, other=0).to(tl.int32)
    c3 = tl.load(src3_ptr + i, mask=do, other=0).to(tl.int32)

    tl.store(dx_ptr + j, x, mask=do)
    tl.store(dy_ptr + j, y, mask=do)
    tl.store(dz_ptr + j, z, mask=do)
    tl.store(ddx_ptr + j, ux, mask=do)
    tl.store(ddy_ptr + j, uy, mask=do)
    tl.store(ddz_ptr + j, uz, mask=do)
    tl.store(dE_ptr + j, E, mask=do)
    tl.store(dw_ptr + j, w, mask=do)
    tl.store(debin_ptr + j, ebin, mask=do)
    tl.store(drk0_ptr + j, k0, mask=do)
    tl.store(drk1_ptr + j, k1, mask=do)
    tl.store(drc0_ptr + j, c0, mask=do)
    tl.store(drc1_ptr + j, c1, mask=do)
    tl.store(drc2_ptr + j, c2, mask=do)
    tl.store(drc3_ptr + j, c3, mask=do)
    tl.store(dstatus_ptr + j, 1, mask=do)


@triton.jit
def pack_vacancybank_guarded_kernel(
    sx_ptr, sy_ptr, sz_ptr,
    sZ_ptr, sshell_ptr, sw_ptr,
    srk0_ptr, srk1_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,

    dx_ptr, dy_ptr, dz_ptr,
    dZ_ptr, dshell_ptr, dw_ptr,
    drk0_ptr, drk1_ptr, drc0_ptr, drc1_ptr, drc2_ptr, drc3_ptr,
    dstatus_ptr,

    idx_ptr,
    total_ptr,
    n_dirty: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    m = j < n_dirty

    total = tl.load(total_ptr + 0).to(tl.int32)
    do = m & (j < total)

    i = tl.load(idx_ptr + j, mask=do, other=0).to(tl.int32)

    x = tl.load(sx_ptr + i, mask=do, other=0.0).to(tl.float32)
    y = tl.load(sy_ptr + i, mask=do, other=0.0).to(tl.float32)
    z = tl.load(sz_ptr + i, mask=do, other=0.0).to(tl.float32)
    Z = tl.load(sZ_ptr + i, mask=do, other=0).to(tl.int32)
    shell = tl.load(sshell_ptr + i, mask=do, other=0).to(tl.int32)
    w = tl.load(sw_ptr + i, mask=do, other=0.0).to(tl.float32)
    k0 = tl.load(srk0_ptr + i, mask=do, other=0).to(tl.int32)
    k1 = tl.load(srk1_ptr + i, mask=do, other=0).to(tl.int32)
    c0 = tl.load(src0_ptr + i, mask=do, other=0).to(tl.int32)
    c1 = tl.load(src1_ptr + i, mask=do, other=0).to(tl.int32)
    c2 = tl.load(src2_ptr + i, mask=do, other=0).to(tl.int32)
    c3 = tl.load(src3_ptr + i, mask=do, other=0).to(tl.int32)

    tl.store(dx_ptr + j, x, mask=do)
    tl.store(dy_ptr + j, y, mask=do)
    tl.store(dz_ptr + j, z, mask=do)
    tl.store(dZ_ptr + j, Z, mask=do)
    tl.store(dshell_ptr + j, shell, mask=do)
    tl.store(dw_ptr + j, w, mask=do)
    tl.store(drk0_ptr + j, k0, mask=do)
    tl.store(drk1_ptr + j, k1, mask=do)
    tl.store(drc0_ptr + j, c0, mask=do)
    tl.store(drc1_ptr + j, c1, mask=do)
    tl.store(drc2_ptr + j, c2, mask=do)
    tl.store(drc3_ptr + j, c3, mask=do)
    tl.store(dstatus_ptr + j, 1, mask=do)