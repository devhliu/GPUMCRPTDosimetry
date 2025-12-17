from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def status_to_mask_i32_kernel(
    status_ptr,
    mask_i32_ptr,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    s = tl.load(status_ptr + i, mask=m, other=0).to(tl.int8)
    keep = s == 1
    tl.store(mask_i32_ptr + i, keep.to(tl.int32), mask=m)


@triton.jit
def last_prefix_plus_last_mask_kernel(
    mask_i32_ptr,
    prefix_i32_ptr,
    out_total_ptr,      # int32[1]
    n: tl.constexpr,
):
    """
    total = prefix[n-1] + mask[n-1]
    Writes to out_total_ptr[0].
    """
    # single program
    if tl.program_id(0) == 0:
        last = n - 1
        last_mask = tl.load(mask_i32_ptr + last).to(tl.int32)
        last_pref = tl.load(prefix_i32_ptr + last).to(tl.int32)
        tl.store(out_total_ptr + 0, last_pref + last_mask)


@triton.jit
def scatter_pack_particlebank_soa_kernel(
    # src
    sx_ptr, sy_ptr, sz_ptr,
    sdx_ptr, sdy_ptr, sdz_ptr,
    sE_ptr, sw_ptr, sebin_ptr,
    srk0_ptr, srk1_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,
    sstatus_ptr,

    # dst
    dx_ptr, dy_ptr, dz_ptr,
    ddx_ptr, ddy_ptr, ddz_ptr,
    dE_ptr, dw_ptr, debin_ptr,
    drk0_ptr, drk1_ptr, drc0_ptr, drc1_ptr, drc2_ptr, drc3_ptr,
    dstatus_ptr,

    # mask/prefix
    mask_i32_ptr,
    prefix_i32_ptr,

    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n

    keep = tl.load(mask_i32_ptr + i, mask=m, other=0).to(tl.int32)
    pos = tl.load(prefix_i32_ptr + i, mask=m, other=0).to(tl.int32)
    do = m & (keep == 1)

    x = tl.load(sx_ptr + i, mask=do, other=0.0).to(tl.float32)
    y = tl.load(sy_ptr + i, mask=do, other=0.0).to(tl.float32)
    z = tl.load(sz_ptr + i, mask=do, other=0.0).to(tl.float32)

    dx = tl.load(sdx_ptr + i, mask=do, other=0.0).to(tl.float32)
    dy = tl.load(sdy_ptr + i, mask=do, other=0.0).to(tl.float32)
    dz = tl.load(sdz_ptr + i, mask=do, other=0.0).to(tl.float32)

    E = tl.load(sE_ptr + i, mask=do, other=0.0).to(tl.float32)
    w = tl.load(sw_ptr + i, mask=do, other=0.0).to(tl.float32)
    ebin = tl.load(sebin_ptr + i, mask=do, other=0).to(tl.int32)

    k0 = tl.load(srk0_ptr + i, mask=do, other=0).to(tl.int32)
    k1 = tl.load(srk1_ptr + i, mask=do, other=0).to(tl.int32)
    c0 = tl.load(src0_ptr + i, mask=do, other=0).to(tl.int32)
    c1 = tl.load(src1_ptr + i, mask=do, other=0).to(tl.int32)
    c2 = tl.load(src2_ptr + i, mask=do, other=0).to(tl.int32)
    c3 = tl.load(src3_ptr + i, mask=do, other=0).to(tl.int32)

    tl.store(dx_ptr + pos, x, mask=do)
    tl.store(dy_ptr + pos, y, mask=do)
    tl.store(dz_ptr + pos, z, mask=do)

    tl.store(ddx_ptr + pos, dx, mask=do)
    tl.store(ddy_ptr + pos, dy, mask=do)
    tl.store(ddz_ptr + pos, dz, mask=do)

    tl.store(dE_ptr + pos, E, mask=do)
    tl.store(dw_ptr + pos, w, mask=do)
    tl.store(debin_ptr + pos, ebin, mask=do)

    tl.store(drk0_ptr + pos, k0, mask=do)
    tl.store(drk1_ptr + pos, k1, mask=do)
    tl.store(drc0_ptr + pos, c0, mask=do)
    tl.store(drc1_ptr + pos, c1, mask=do)
    tl.store(drc2_ptr + pos, c2, mask=do)
    tl.store(drc3_ptr + pos, c3, mask=do)

    tl.store(dstatus_ptr + pos, 1, mask=do)


@triton.jit
def scatter_pack_vacancybank_soa_kernel(
    # src
    sx_ptr, sy_ptr, sz_ptr,
    sZ_ptr, sshell_ptr, sw_ptr,
    srk0_ptr, srk1_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr,
    sstatus_ptr,

    # dst
    dx_ptr, dy_ptr, dz_ptr,
    dZ_ptr, dshell_ptr, dw_ptr,
    drk0_ptr, drk1_ptr, drc0_ptr, drc1_ptr, drc2_ptr, drc3_ptr,
    dstatus_ptr,

    # mask/prefix
    mask_i32_ptr,
    prefix_i32_ptr,

    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n

    keep = tl.load(mask_i32_ptr + i, mask=m, other=0).to(tl.int32)
    pos = tl.load(prefix_i32_ptr + i, mask=m, other=0).to(tl.int32)
    do = m & (keep == 1)

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

    tl.store(dx_ptr + pos, x, mask=do)
    tl.store(dy_ptr + pos, y, mask=do)
    tl.store(dz_ptr + pos, z, mask=do)

    tl.store(dZ_ptr + pos, Z, mask=do)
    tl.store(dshell_ptr + pos, shell, mask=do)
    tl.store(dw_ptr + pos, w, mask=do)

    tl.store(drk0_ptr + pos, k0, mask=do)
    tl.store(drk1_ptr + pos, k1, mask=do)
    tl.store(drc0_ptr + pos, c0, mask=do)
    tl.store(drc1_ptr + pos, c1, mask=do)
    tl.store(drc2_ptr + pos, c2, mask=do)
    tl.store(drc3_ptr + pos, c3, mask=do)

    tl.store(dstatus_ptr + pos, 1, mask=do)