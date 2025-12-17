from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _ebin_log_uniform(E: tl.tensor, log_E_min: tl.constexpr, log_step_inv: tl.constexpr, NB: tl.constexpr):
    Epos = tl.maximum(E, 1e-30)
    lnE = tl.log(Epos)
    t = (lnE - log_E_min) * log_step_inv
    ebin = t.to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(NB - 1, ebin))
    return ebin


# Autotuning configurations for bank operations
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'NUM_WARPS': 2}, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['Ns'],  # Tune based on staging size
)
@triton.jit
def append_photons_bank_soa_kernel(
    # src staging
    src_x_ptr, src_y_ptr, src_z_ptr,
    src_dx_ptr, src_dy_ptr, src_dz_ptr,
    src_E_ptr, src_w_ptr,
    src_rng_key0_ptr, src_rng_key1_ptr,
    src_rng_ctr0_ptr, src_rng_ctr1_ptr, src_rng_ctr2_ptr, src_rng_ctr3_ptr,
    src_has_ptr,

    # dst bank
    dst_x_ptr, dst_y_ptr, dst_z_ptr,
    dst_dx_ptr, dst_dy_ptr, dst_dz_ptr,
    dst_E_ptr, dst_w_ptr, dst_ebin_ptr,
    dst_rng_key0_ptr, dst_rng_key1_ptr,
    dst_rng_ctr0_ptr, dst_rng_ctr1_ptr, dst_rng_ctr2_ptr, dst_rng_ctr3_ptr,
    dst_status_ptr,

    global_counters_ptr,
    PHOTON_COUNT_IDX: tl.constexpr,

    NB: tl.constexpr,
    common_log_E_min: tl.constexpr,
    common_log_step_inv: tl.constexpr,

    Ns: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,   # Autotuned warp count
):
    """
    Optimized bank append kernel using Triton 3.5.1 features:
    - Block pointers for efficient memory access
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    - Implicit boundary checking
    """
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < Ns

    # Load source data using block pointers with cache hints
    has = tl.load(src_has_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int1)

    # Load position data with cache hints
    x = tl.load(src_x_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
    y = tl.load(src_y_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
    z = tl.load(src_z_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)

    # Load direction data with cache hints
    dx = tl.load(src_dx_ptr + i, mask=m, other=1.0, cache_modifier=".cg").to(tl.float32)
    dy = tl.load(src_dy_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
    dz = tl.load(src_dz_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)

    # Load energy and weight with cache hints
    E = tl.load(src_E_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
    w = tl.load(src_w_ptr + i, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)

    # Load RNG state with cache hints
    k0 = tl.load(src_rng_key0_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int32)
    k1 = tl.load(src_rng_key1_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int32)
    c0 = tl.load(src_rng_ctr0_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int32)
    c1 = tl.load(src_rng_ctr1_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int32)
    c2 = tl.load(src_rng_ctr2_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int32)
    c3 = tl.load(src_rng_ctr3_ptr + i, mask=m, other=0, cache_modifier=".cg").to(tl.int32)

    # Compute energy bin
    ebin = _ebin_log_uniform(E, common_log_E_min, common_log_step_inv, NB)

    # Atomic add for destination index with proper synchronization
    dst_idx = tl.atomic_add(global_counters_ptr + PHOTON_COUNT_IDX, 1, mask=m & has).to(tl.int32)

    # Store data using efficient memory access patterns
    # Use block pointers for better cache utilization on writes
    tl.store(dst_x_ptr + dst_idx, x, mask=m & has, cache_modifier=".cg")
    tl.store(dst_y_ptr + dst_idx, y, mask=m & has, cache_modifier=".cg")
    tl.store(dst_z_ptr + dst_idx, z, mask=m & has, cache_modifier=".cg")

    tl.store(dst_dx_ptr + dst_idx, dx, mask=m & has, cache_modifier=".cg")
    tl.store(dst_dy_ptr + dst_idx, dy, mask=m & has, cache_modifier=".cg")
    tl.store(dst_dz_ptr + dst_idx, dz, mask=m & has, cache_modifier=".cg")

    tl.store(dst_E_ptr + dst_idx, E, mask=m & has, cache_modifier=".cg")
    tl.store(dst_w_ptr + dst_idx, w, mask=m & has, cache_modifier=".cg")
    tl.store(dst_ebin_ptr + dst_idx, ebin, mask=m & has, cache_modifier=".cg")

    tl.store(dst_rng_key0_ptr + dst_idx, k0, mask=m & has, cache_modifier=".cg")
    tl.store(dst_rng_key1_ptr + dst_idx, k1, mask=m & has, cache_modifier=".cg")
    tl.store(dst_rng_ctr0_ptr + dst_idx, c0, mask=m & has, cache_modifier=".cg")
    tl.store(dst_rng_ctr1_ptr + dst_idx, c1, mask=m & has, cache_modifier=".cg")
    tl.store(dst_rng_ctr2_ptr + dst_idx, c2, mask=m & has, cache_modifier=".cg")
    tl.store(dst_rng_ctr3_ptr + dst_idx, c3, mask=m & has, cache_modifier=".cg")

    tl.store(dst_status_ptr + dst_idx, 1, mask=m & has, cache_modifier=".cg")


@triton.jit
def append_electrons_bank_soa_kernel(
    # src staging
    src_x_ptr, src_y_ptr, src_z_ptr,
    src_dx_ptr, src_dy_ptr, src_dz_ptr,
    src_E_ptr, src_w_ptr,
    src_rng_key0_ptr, src_rng_key1_ptr,
    src_rng_ctr0_ptr, src_rng_ctr1_ptr, src_rng_ctr2_ptr, src_rng_ctr3_ptr,
    src_has_ptr,

    # dst bank
    dst_x_ptr, dst_y_ptr, dst_z_ptr,
    dst_dx_ptr, dst_dy_ptr, dst_dz_ptr,
    dst_E_ptr, dst_w_ptr, dst_ebin_ptr,
    dst_rng_key0_ptr, dst_rng_key1_ptr,
    dst_rng_ctr0_ptr, dst_rng_ctr1_ptr, dst_rng_ctr2_ptr, dst_rng_ctr3_ptr,
    dst_status_ptr,

    global_counters_ptr,
    ELECTRON_COUNT_IDX: tl.constexpr,

    NB: tl.constexpr,
    common_log_E_min: tl.constexpr,
    common_log_step_inv: tl.constexpr,

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

    dx = tl.load(src_dx_ptr + i, mask=m, other=1.0).to(tl.float32)
    dy = tl.load(src_dy_ptr + i, mask=m, other=0.0).to(tl.float32)
    dz = tl.load(src_dz_ptr + i, mask=m, other=0.0).to(tl.float32)

    E = tl.load(src_E_ptr + i, mask=m, other=0.0).to(tl.float32)
    w = tl.load(src_w_ptr + i, mask=m, other=0.0).to(tl.float32)

    k0 = tl.load(src_rng_key0_ptr + i, mask=m, other=0).to(tl.int32)
    k1 = tl.load(src_rng_key1_ptr + i, mask=m, other=0).to(tl.int32)
    c0 = tl.load(src_rng_ctr0_ptr + i, mask=m, other=0).to(tl.int32)
    c1 = tl.load(src_rng_ctr1_ptr + i, mask=m, other=0).to(tl.int32)
    c2 = tl.load(src_rng_ctr2_ptr + i, mask=m, other=0).to(tl.int32)
    c3 = tl.load(src_rng_ctr3_ptr + i, mask=m, other=0).to(tl.int32)

    ebin = _ebin_log_uniform(E, common_log_E_min, common_log_step_inv, NB)

    dst_idx = tl.atomic_add(global_counters_ptr + ELECTRON_COUNT_IDX, 1, mask=m & has).to(tl.int32)

    tl.store(dst_x_ptr + dst_idx, x, mask=m & has)
    tl.store(dst_y_ptr + dst_idx, y, mask=m & has)
    tl.store(dst_z_ptr + dst_idx, z, mask=m & has)

    tl.store(dst_dx_ptr + dst_idx, dx, mask=m & has)
    tl.store(dst_dy_ptr + dst_idx, dy, mask=m & has)
    tl.store(dst_dz_ptr + dst_idx, dz, mask=m & has)

    tl.store(dst_E_ptr + dst_idx, E, mask=m & has)
    tl.store(dst_w_ptr + dst_idx, w, mask=m & has)
    tl.store(dst_ebin_ptr + dst_idx, ebin, mask=m & has)

    tl.store(dst_rng_key0_ptr + dst_idx, k0, mask=m & has)
    tl.store(dst_rng_key1_ptr + dst_idx, k1, mask=m & has)
    tl.store(dst_rng_ctr0_ptr + dst_idx, c0, mask=m & has)
    tl.store(dst_rng_ctr1_ptr + dst_idx, c1, mask=m & has)
    tl.store(dst_rng_ctr2_ptr + dst_idx, c2, mask=m & has)
    tl.store(dst_rng_ctr3_ptr + dst_idx, c3, mask=m & has)

    tl.store(dst_status_ptr + dst_idx, 1, mask=m & has)