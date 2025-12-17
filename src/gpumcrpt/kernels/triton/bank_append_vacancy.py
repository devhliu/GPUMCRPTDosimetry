from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def bank_append_vacancies_kernel(
    # src vacancy staging (length Ns)
    src_pos_cm_ptr, src_mat_ptr, src_shell_ptr, src_w_ptr,
    src_rng_key0_ptr, src_rng_key1_ptr,
    src_rng_ctr0_ptr, src_rng_ctr1_ptr, src_rng_ctr2_ptr, src_rng_ctr3_ptr,
    src_has_ptr,                # int8 [Ns]

    # dst vacancy bank
    bank_pos_cm_ptr, bank_mat_ptr, bank_shell_ptr, bank_w_ptr,
    bank_rng_key0_ptr, bank_rng_key1_ptr,
    bank_rng_ctr0_ptr, bank_rng_ctr1_ptr, bank_rng_ctr2_ptr, bank_rng_ctr3_ptr,
    bank_alive_ptr,

    vac_count_ptr,              # int32[1]

    Ns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < Ns

    has = tl.load(src_has_ptr + i, mask=m, other=0).to(tl.int1)

    z = tl.load(src_pos_cm_ptr + i * 3 + 0, mask=m, other=0.0).to(tl.float32)
    y = tl.load(src_pos_cm_ptr + i * 3 + 1, mask=m, other=0.0).to(tl.float32)
    x = tl.load(src_pos_cm_ptr + i * 3 + 2, mask=m, other=0.0).to(tl.float32)

    mat = tl.load(src_mat_ptr + i, mask=m, other=0).to(tl.int32)
    shell = tl.load(src_shell_ptr + i, mask=m, other=0).to(tl.int8)
    w = tl.load(src_w_ptr + i, mask=m, other=0.0).to(tl.float32)

    k0 = tl.load(src_rng_key0_ptr + i, mask=m, other=0).to(tl.int32)
    k1 = tl.load(src_rng_key1_ptr + i, mask=m, other=0).to(tl.int32)
    c0 = tl.load(src_rng_ctr0_ptr + i, mask=m, other=0).to(tl.int32)
    c1 = tl.load(src_rng_ctr1_ptr + i, mask=m, other=0).to(tl.int32)
    c2 = tl.load(src_rng_ctr2_ptr + i, mask=m, other=0).to(tl.int32)
    c3 = tl.load(src_rng_ctr3_ptr + i, mask=m, other=0).to(tl.int32)

    idx = tl.atomic_add(vac_count_ptr, 1, mask=m & has).to(tl.int32)

    tl.store(bank_pos_cm_ptr + idx * 3 + 0, z, mask=m & has)
    tl.store(bank_pos_cm_ptr + idx * 3 + 1, y, mask=m & has)
    tl.store(bank_pos_cm_ptr + idx * 3 + 2, x, mask=m & has)

    tl.store(bank_mat_ptr + idx, mat, mask=m & has)
    tl.store(bank_shell_ptr + idx, shell, mask=m & has)
    tl.store(bank_w_ptr + idx, w, mask=m & has)

    tl.store(bank_rng_key0_ptr + idx, k0, mask=m & has)
    tl.store(bank_rng_key1_ptr + idx, k1, mask=m & has)
    tl.store(bank_rng_ctr0_ptr + idx, c0, mask=m & has)
    tl.store(bank_rng_ctr1_ptr + idx, c1, mask=m & has)
    tl.store(bank_rng_ctr2_ptr + idx, c2, mask=m & has)
    tl.store(bank_rng_ctr3_ptr + idx, c3, mask=m & has)

    tl.store(bank_alive_ptr + idx, 1, mask=m & has)