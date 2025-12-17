from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def append_vacancy_aux_bank_soa_kernel(
    # src (length Ns)
    src_w_ptr,
    src_rng_key0_ptr, src_rng_key1_ptr,
    src_rng_ctr0_ptr, src_rng_ctr1_ptr, src_rng_ctr2_ptr, src_rng_ctr3_ptr,
    src_has_ptr,

    # dst vacancy bank aux (aligned to same indices as append_vacancies_bank_soa_kernel!)
    # IMPORTANT: must use the SAME atomic counter and obtain SAME dst_idx mapping.
    # Therefore this kernel is intended to be fused with append_vacancies_bank_soa_kernel in production.
    # For Phase 10, we provide a fused kernel below instead.

    dst_w_ptr,
    dst_rng_key0_ptr, dst_rng_key1_ptr,
    dst_rng_ctr0_ptr, dst_rng_ctr1_ptr, dst_rng_ctr2_ptr, dst_rng_ctr3_ptr,

    # global counters
    global_counters_ptr,
    VAC_COUNT_IDX: tl.constexpr,

    Ns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    raise tl.static_assert(False, "Use the fused vacancy append kernel: append_vacancies_full_bank_soa_kernel")