from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def append_vacancies_full_bank_soa_kernel(
    # src staging (Ns)
    src_x_ptr, src_y_ptr, src_z_ptr,
    src_atom_Z_ptr,                 # int32
    src_shell_idx_ptr,              # int32
    src_w_ptr,                      # float32
    src_rng_key0_ptr, src_rng_key1_ptr,
    src_rng_ctr0_ptr, src_rng_ctr1_ptr, src_rng_ctr2_ptr, src_rng_ctr3_ptr,
    src_has_ptr,                    # int8

    # dst vacancy bank
    dst_x_ptr, dst_y_ptr, dst_z_ptr,
    dst_atom_Z_ptr,
    dst_shell_idx_ptr,
    dst_w_ptr,
    dst_rng_key0_ptr, dst_rng_key1_ptr,
    dst_rng_ctr0_ptr, dst_rng_ctr1_ptr, dst_rng_ctr2_ptr, dst_rng_ctr3_ptr,
    dst_status_ptr,                 # int8

    global_counters_ptr,            # int32*
    VAC_COUNT_IDX: tl.constexpr,    # 2

    Ns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < Ns

    has = tl.load(src_has_ptr + i, mask=m, other=0).to(tl.int1)

    x = tl.load(src_x_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(src_y_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(src_z_ptr + i, mask=m, other=0.0).to(tl.float32)
    Z = tl.load(src_atom_Z_ptr + i, mask=m, other=0).to(tl.int32)
    shell = tl.load(src_shell_idx_ptr + i, mask=m, other=0).to(tl.int32)

    w = tl.load(src_w_ptr + i, mask=m, other=0.0).to(tl.float32)
    k0 = tl.load(src_rng_key0_ptr + i, mask=m, other=0).to(tl.int32)
    k1 = tl.load(src_rng_key1_ptr + i, mask=m, other=0).to(tl.int32)
    c0 = tl.load(src_rng_ctr0_ptr + i, mask=m, other=0).to(tl.int32)
    c1 = tl.load(src_rng_ctr1_ptr + i, mask=m, other=0).to(tl.int32)
    c2 = tl.load(src_rng_ctr2_ptr + i, mask=m, other=0).to(tl.int32)
    c3 = tl.load(src_rng_ctr3_ptr + i, mask=m, other=0).to(tl.int32)

    dst_idx = tl.atomic_add(global_counters_ptr + VAC_COUNT_IDX, 1, mask=m & has).to(tl.int32)

    tl.store(dst_x_ptr + dst_idx, x, mask=m & has)
    tl.store(dst_y_ptr + dst_idx, y, mask=m & has)
    tl.store(dst_z_ptr + dst_idx, z, mask=m & has)
    tl.store(dst_atom_Z_ptr + dst_idx, Z, mask=m & has)
    tl.store(dst_shell_idx_ptr + dst_idx, shell, mask=m & has)

    tl.store(dst_w_ptr + dst_idx, w, mask=m & has)
    tl.store(dst_rng_key0_ptr + dst_idx, k0, mask=m & has)
    tl.store(dst_rng_key1_ptr + dst_idx, k1, mask=m & has)
    tl.store(dst_rng_ctr0_ptr + dst_idx, c0, mask=m & has)
    tl.store(dst_rng_ctr1_ptr + dst_idx, c1, mask=m & has)
    tl.store(dst_rng_ctr2_ptr + dst_idx, c2, mask=m & has)
    tl.store(dst_rng_ctr3_ptr + dst_idx, c3, mask=m & has)

    tl.store(dst_status_ptr + dst_idx, 1, mask=m & has)