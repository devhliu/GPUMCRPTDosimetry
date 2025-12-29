from __future__ import annotations

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 2}),
    ],
    key=['Ns'],
    warmup=10,
    rep=20,
)
@triton.jit
def append_photons_bank_soa_kernel(
    src_x_ptr, src_y_ptr, src_z_ptr,
    src_dx_ptr, src_dy_ptr, src_dz_ptr,
    src_E_ptr, src_w_ptr,
    src_id_ptr,
    src_has_ptr,

    dst_x_ptr, dst_y_ptr, dst_z_ptr,
    dst_dx_ptr, dst_dy_ptr, dst_dz_ptr,
    dst_E_ptr, dst_w_ptr, dst_ebin_ptr,
    dst_id_ptr,
    dst_status_ptr,

    global_counters_ptr,
    PHOTON_COUNT_IDX: tl.constexpr,

    NB: tl.constexpr,
    common_log_E_min: tl.constexpr,
    common_log_step_inv: tl.constexpr,

    Ns: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < Ns

    has = tl.load(src_has_ptr + i, mask=m, other=0).to(tl.int1)

    x = tl.load(src_x_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(src_y_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(src_z_ptr + i, mask=m, other=0.0).to(tl.float32)
    dx = tl.load(src_dx_ptr + i, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(src_dy_ptr + i, mask=m, other=0.0).to(tl.float32)
    dz = tl.load(src_dz_ptr + i, mask=m, other=0.0).to(tl.float32)
    E = tl.load(src_E_ptr + i, mask=m, other=0.0).to(tl.float32)
    w = tl.load(src_w_ptr + i, mask=m, other=0.0).to(tl.float32)
    photon_id = tl.load(src_id_ptr + i, mask=m, other=0).to(tl.int64)

    dst_idx = tl.atomic_add(global_counters_ptr + PHOTON_COUNT_IDX, 1, mask=m & has).to(tl.int32)

    tl.store(dst_x_ptr + dst_idx, x, mask=m & has)
    tl.store(dst_y_ptr + dst_idx, y, mask=m & has)
    tl.store(dst_z_ptr + dst_idx, z, mask=m & has)
    tl.store(dst_dx_ptr + dst_idx, dx, mask=m & has)
    tl.store(dst_dy_ptr + dst_idx, dy, mask=m & has)
    tl.store(dst_dz_ptr + dst_idx, dz, mask=m & has)
    tl.store(dst_E_ptr + dst_idx, E, mask=m & has)
    tl.store(dst_w_ptr + dst_idx, w, mask=m & has)

    Epos = tl.maximum(E, 1e-30)
    lnE = tl.log(Epos)
    t = (lnE - common_log_E_min) * common_log_step_inv
    ebin = t.to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(NB - 1, ebin))
    tl.store(dst_ebin_ptr + dst_idx, ebin, mask=m & has)

    tl.store(dst_id_ptr + dst_idx, photon_id, mask=m & has)

    tl.store(dst_status_ptr + dst_idx, 1, mask=m & has)
