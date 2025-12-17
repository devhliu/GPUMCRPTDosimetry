from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.kernels.triton.rng_philox import rng_u01_philox


@triton.jit
def atomic_relaxation_kernel(
    # vacancy inputs
    vac_pos_cm_ptr, vac_mat_ptr, vac_shell_ptr, vac_w_ptr,
    # RNG (Philox): key + ctr per particle (SoA)
    vac_rng_key0_ptr, vac_rng_key1_ptr,
    vac_rng_ctr0_ptr, vac_rng_ctr1_ptr, vac_rng_ctr2_ptr, vac_rng_ctr3_ptr,

    # tables [M,S]
    fluor_yield_ptr, E_xray_MeV_ptr, E_auger_MeV_ptr,

    # photon outputs
    out_ph_pos_cm_ptr, out_ph_dir_ptr, out_ph_E_MeV_ptr, out_ph_w_ptr,
    out_ph_rng_key0_ptr, out_ph_rng_key1_ptr,
    out_ph_rng_ctr0_ptr, out_ph_rng_ctr1_ptr, out_ph_rng_ctr2_ptr, out_ph_rng_ctr3_ptr,
    out_has_ph_ptr,

    # electron outputs (Auger)
    out_e_pos_cm_ptr, out_e_dir_ptr, out_e_E_MeV_ptr, out_e_w_ptr,
    out_e_rng_key0_ptr, out_e_rng_key1_ptr,
    out_e_rng_ctr0_ptr, out_e_rng_ctr1_ptr, out_e_rng_ctr2_ptr, out_e_rng_ctr3_ptr,
    out_has_e_ptr,

    # local deposit
    edep_flat_ptr,

    # constants
    Nv: tl.constexpr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    photon_cut_MeV: tl.constexpr,
    e_cut_MeV: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Phase 9.2: use counter-based Philox RNG (no LCG placeholders).
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < Nv

    z = tl.load(vac_pos_cm_ptr + i * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(vac_pos_cm_ptr + i * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(vac_pos_cm_ptr + i * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    mat = tl.load(vac_mat_ptr + i, mask=mask, other=0).to(tl.int32)
    shell = tl.load(vac_shell_ptr + i, mask=mask, other=0).to(tl.int32)
    w = tl.load(vac_w_ptr + i, mask=mask, other=0.0).to(tl.float32)

    key0 = tl.load(vac_rng_key0_ptr + i, mask=mask, other=0).to(tl.uint32)
    key1 = tl.load(vac_rng_key1_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr0 = tl.load(vac_rng_ctr0_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr1 = tl.load(vac_rng_ctr1_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr2 = tl.load(vac_rng_ctr2_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr3 = tl.load(vac_rng_ctr3_ptr + i, mask=mask, other=0).to(tl.uint32)

    u0, u1, u2, _, ctr0, ctr1, ctr2, ctr3 = rng_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3)

    mat = tl.maximum(0, tl.minimum(mat, M - 1))
    shell = tl.maximum(0, tl.minimum(shell, S - 1))
    base = mat * S + shell

    fy = tl.load(fluor_yield_ptr + base, mask=mask, other=0.0).to(tl.float32)
    Ex = tl.load(E_xray_MeV_ptr + base, mask=mask, other=0.0).to(tl.float32)
    Ea = tl.load(E_auger_MeV_ptr + base, mask=mask, other=0.0).to(tl.float32)

    emit_x = u0 < fy

    # isotropic direction using u1,u2
    cost = 2.0 * u1 - 1.0
    sint = tl.sqrt(tl.maximum(0.0, 1.0 - cost * cost))
    phi = 6.28318530718 * u2
    ux = sint * tl.cos(phi)
    uy = sint * tl.sin(phi)
    uz = cost

    # voxel index for local deposit
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    has_ph = mask & inside & emit_x & (Ex >= photon_cut_MeV)
    dep_x = mask & (emit_x & (~has_ph))

    has_e = mask & inside & (~emit_x) & (Ea >= e_cut_MeV)
    dep_e = mask & ((~emit_x) & (~has_e))

    depE = tl.where(dep_x, Ex * w, 0.0) + tl.where(dep_e, Ea * w, 0.0)
    tl.atomic_add(edep_flat_ptr + lin, depE, mask=inside & (depE > 0.0))

    # write RNG state for both outputs (same updated state)
    # photon
    tl.store(out_ph_rng_key0_ptr + i, key0, mask=mask)
    tl.store(out_ph_rng_key1_ptr + i, key1, mask=mask)
    tl.store(out_ph_rng_ctr0_ptr + i, ctr0, mask=mask)
    tl.store(out_ph_rng_ctr1_ptr + i, ctr1, mask=mask)
    tl.store(out_ph_rng_ctr2_ptr + i, ctr2, mask=mask)
    tl.store(out_ph_rng_ctr3_ptr + i, ctr3, mask=mask)

    # electron
    tl.store(out_e_rng_key0_ptr + i, key0, mask=mask)
    tl.store(out_e_rng_key1_ptr + i, key1, mask=mask)
    tl.store(out_e_rng_ctr0_ptr + i, ctr0, mask=mask)
    tl.store(out_e_rng_ctr1_ptr + i, ctr1, mask=mask)
    tl.store(out_e_rng_ctr2_ptr + i, ctr2, mask=mask)
    tl.store(out_e_rng_ctr3_ptr + i, ctr3, mask=mask)

    # photon outputs
    tl.store(out_ph_pos_cm_ptr + i * 3 + 0, z, mask=mask)
    tl.store(out_ph_pos_cm_ptr + i * 3 + 1, y, mask=mask)
    tl.store(out_ph_pos_cm_ptr + i * 3 + 2, x, mask=mask)
    tl.store(out_ph_dir_ptr + i * 3 + 0, uz, mask=mask)
    tl.store(out_ph_dir_ptr + i * 3 + 1, uy, mask=mask)
    tl.store(out_ph_dir_ptr + i * 3 + 2, ux, mask=mask)
    tl.store(out_ph_E_MeV_ptr + i, Ex, mask=mask)
    tl.store(out_ph_w_ptr + i, w, mask=mask)
    tl.store(out_has_ph_ptr + i, has_ph.to(tl.int8), mask=mask)

    # electron outputs
    tl.store(out_e_pos_cm_ptr + i * 3 + 0, z, mask=mask)
    tl.store(out_e_pos_cm_ptr + i * 3 + 1, y, mask=mask)
    tl.store(out_e_pos_cm_ptr + i * 3 + 2, x, mask=mask)
    tl.store(out_e_dir_ptr + i * 3 + 0, uz, mask=mask)
    tl.store(out_e_dir_ptr + i * 3 + 1, uy, mask=mask)
    tl.store(out_e_dir_ptr + i * 3 + 2, ux, mask=mask)
    tl.store(out_e_E_MeV_ptr + i, Ea, mask=mask)
    tl.store(out_e_w_ptr + i, w, mask=mask)
    tl.store(out_has_e_ptr + i, has_e.to(tl.int8), mask=mask)