from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.kernels.triton.rng_philox import rng_u01_philox


@triton.jit
def atomic_relaxation_soa_kernel(
    # vacancy inputs (active staging)
    vac_x_ptr, vac_y_ptr, vac_z_ptr,
    vac_Z_ptr,              # int32 atomic number
    vac_shell_ptr,          # int32 shell index
    vac_w_ptr,              # float32 weight
    vac_rng_key0_ptr, vac_rng_key1_ptr,
    vac_rng_ctr0_ptr, vac_rng_ctr1_ptr, vac_rng_ctr2_ptr, vac_rng_ctr3_ptr,

    # tables keyed by atomic Z: [Zmax+1, S]
    fluor_yield_ptr,
    E_xray_MeV_ptr,
    E_auger_MeV_ptr,
    Zmax: tl.constexpr,
    S: tl.constexpr,

    # photon staging outputs
    out_ph_x_ptr, out_ph_y_ptr, out_ph_z_ptr,
    out_ph_dx_ptr, out_ph_dy_ptr, out_ph_dz_ptr,
    out_ph_E_ptr, out_ph_w_ptr,
    out_ph_rng_key0_ptr, out_ph_rng_key1_ptr,
    out_ph_rng_ctr0_ptr, out_ph_rng_ctr1_ptr, out_ph_rng_ctr2_ptr, out_ph_rng_ctr3_ptr,
    out_ph_has_ptr,

    # electron staging outputs
    out_e_x_ptr, out_e_y_ptr, out_e_z_ptr,
    out_e_dx_ptr, out_e_dy_ptr, out_e_dz_ptr,
    out_e_E_ptr, out_e_w_ptr,
    out_e_rng_key0_ptr, out_e_rng_key1_ptr,
    out_e_rng_ctr0_ptr, out_e_rng_ctr1_ptr, out_e_rng_ctr2_ptr, out_e_rng_ctr3_ptr,
    out_e_has_ptr,

    # local deposit
    edep_flat_ptr,
    Zdim: tl.constexpr, Ydim: tl.constexpr, Xdim: tl.constexpr,
    voxel_x_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_z_cm: tl.constexpr,

    photon_cut_MeV: tl.constexpr,
    e_cut_MeV: tl.constexpr,

    Nv: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < Nv

    x = tl.load(vac_x_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(vac_y_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(vac_z_ptr + i, mask=m, other=0.0).to(tl.float32)
    Zatom = tl.load(vac_Z_ptr + i, mask=m, other=1).to(tl.int32)
    shell = tl.load(vac_shell_ptr + i, mask=m, other=0).to(tl.int32)
    w = tl.load(vac_w_ptr + i, mask=m, other=0.0).to(tl.float32)

    Zatom = tl.maximum(1, tl.minimum(Zmax, Zatom))
    shell = tl.maximum(0, tl.minimum(S - 1, shell))
    base = Zatom * S + shell

    key0 = tl.load(vac_rng_key0_ptr + i, mask=m, other=0).to(tl.int32)
    key1 = tl.load(vac_rng_key1_ptr + i, mask=m, other=0).to(tl.int32)
    ctr0 = tl.load(vac_rng_ctr0_ptr + i, mask=m, other=0).to(tl.int32)
    ctr1 = tl.load(vac_rng_ctr1_ptr + i, mask=m, other=0).to(tl.int32)
    ctr2 = tl.load(vac_rng_ctr2_ptr + i, mask=m, other=0).to(tl.int32)
    ctr3 = tl.load(vac_rng_ctr3_ptr + i, mask=m, other=0).to(tl.int32)

    u0, u1, u2, _, ctr0u, ctr1u, ctr2u, ctr3u = rng_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3)

    fy = tl.load(fluor_yield_ptr + base, mask=m, other=0.0).to(tl.float32)
    Ex = tl.load(E_xray_MeV_ptr + base, mask=m, other=0.0).to(tl.float32)
    Ea = tl.load(E_auger_MeV_ptr + base, mask=m, other=0.0).to(tl.float32)

    emit_x = u0 < fy

    cost = 2.0 * u1 - 1.0
    sint = tl.sqrt(tl.maximum(0.0, 1.0 - cost * cost))
    phi = 6.28318530718 * u2
    dx = sint * tl.cos(phi)
    dy = sint * tl.sin(phi)
    dz = cost

    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    inside = (ix >= 0) & (ix < Xdim) & (iy >= 0) & (iy < Ydim) & (iz >= 0) & (iz < Zdim)
    lin = iz * (Ydim * Xdim) + iy * Xdim + ix

    has_ph = m & inside & emit_x & (Ex >= photon_cut_MeV)
    has_e = m & inside & (~emit_x) & (Ea >= e_cut_MeV)

    dep_x = m & emit_x & (~has_ph)
    dep_e = m & (~emit_x) & (~has_e)
    depE = tl.where(dep_x, Ex * w, 0.0) + tl.where(dep_e, Ea * w, 0.0)
    tl.atomic_add(edep_flat_ptr + lin, depE, mask=inside & (depE > 0.0))

    # photon staging
    tl.store(out_ph_x_ptr + i, x, mask=m)
    tl.store(out_ph_y_ptr + i, y, mask=m)
    tl.store(out_ph_z_ptr + i, z, mask=m)
    tl.store(out_ph_dx_ptr + i, dx, mask=m)
    tl.store(out_ph_dy_ptr + i, dy, mask=m)
    tl.store(out_ph_dz_ptr + i, dz, mask=m)
    tl.store(out_ph_E_ptr + i, Ex, mask=m)
    tl.store(out_ph_w_ptr + i, w, mask=m)
    tl.store(out_ph_rng_key0_ptr + i, key0, mask=m)
    tl.store(out_ph_rng_key1_ptr + i, key1, mask=m)
    tl.store(out_ph_rng_ctr0_ptr + i, ctr0u.to(tl.int32), mask=m)
    tl.store(out_ph_rng_ctr1_ptr + i, ctr1u.to(tl.int32), mask=m)
    tl.store(out_ph_rng_ctr2_ptr + i, ctr2u.to(tl.int32), mask=m)
    tl.store(out_ph_rng_ctr3_ptr + i, ctr3u.to(tl.int32), mask=m)
    tl.store(out_ph_has_ptr + i, has_ph.to(tl.int8), mask=m)

    # electron staging
    tl.store(out_e_x_ptr + i, x, mask=m)
    tl.store(out_e_y_ptr + i, y, mask=m)
    tl.store(out_e_z_ptr + i, z, mask=m)
    tl.store(out_e_dx_ptr + i, dx, mask=m)
    tl.store(out_e_dy_ptr + i, dy, mask=m)
    tl.store(out_e_dz_ptr + i, dz, mask=m)
    tl.store(out_e_E_ptr + i, Ea, mask=m)
    tl.store(out_e_w_ptr + i, w, mask=m)
    tl.store(out_e_rng_key0_ptr + i, key0, mask=m)
    tl.store(out_e_rng_key1_ptr + i, key1, mask=m)
    tl.store(out_e_rng_ctr0_ptr + i, ctr0u.to(tl.int32), mask=m)
    tl.store(out_e_rng_ctr1_ptr + i, ctr1u.to(tl.int32), mask=m)
    tl.store(out_e_rng_ctr2_ptr + i, ctr2u.to(tl.int32), mask=m)
    tl.store(out_e_rng_ctr3_ptr + i, ctr3u.to(tl.int32), mask=m)
    tl.store(out_e_has_ptr + i, has_e.to(tl.int8), mask=m)