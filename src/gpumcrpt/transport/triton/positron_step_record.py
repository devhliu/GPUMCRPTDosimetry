from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton.rng import rand_uniform_u01


@triton.jit
def positron_condensed_step_record_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    material_id_ptr, rho_ptr, ref_rho_ptr,
    S_restricted_ptr, range_csda_ptr,
    P_brem_per_cm_ptr, P_delta_per_cm_ptr,
    # record outputs
    rec_lin_ptr, rec_val_ptr,
    out_pos_ptr, out_dir_ptr, out_E_ptr, out_w_ptr, out_rng_ptr, out_ebin_ptr,
    out_alive_ptr, out_emit_brem_ptr, out_emit_delta_ptr, out_stop_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr, ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    f_vox: tl.constexpr, f_range: tl.constexpr, max_dE_frac: tl.constexpr,
    e_cut_MeV: tl.constexpr,
):
    """
    Record-mode condensed-history step for positrons:
      - record continuous deposition (lin, dE*w)
      - do NOT atomic add in-kernel
      - produce stop flag when E drops below cutoff (annihilation-at-rest handled outside)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    w = tl.load(w_ptr + offs, mask=True, other=0.0)
    rng = tl.load(rng_ptr + offs, mask=True, other=123456789)
    ebin = tl.load(ebin_ptr + offs, mask=True, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)
    uz = tl.load(dir_ptr + offs * 3 + 0, mask=True, other=1.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=True, other=0.0)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=True, other=0.0)

    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    mat = tl.load(material_id_ptr + lin, mask=inside, other=0).to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))
    rho = tl.load(rho_ptr + lin, mask=inside, other=1e-3).to(tl.float32)
    rho_ref = tl.load(ref_rho_ptr + mat, mask=inside, other=1.0).to(tl.float32)
    rho_scale = rho / tl.maximum(rho_ref, 1e-6)

    off = mat * ECOUNT + ebin
    S = tl.load(S_restricted_ptr + off, mask=inside, other=0.0) * rho_scale
    R = tl.load(range_csda_ptr + off, mask=inside, other=1e-3)

    vox_mean = (voxel_z_cm + voxel_y_cm + voxel_x_cm) * (1.0 / 3.0)
    ds = tl.minimum(f_vox * vox_mean, f_range * tl.maximum(R, 1e-6))
    ds = tl.maximum(ds, 1e-5)

    dE = tl.minimum(S * ds, max_dE_frac * E)
    dE = tl.minimum(dE, E)

    rec_lin = tl.where(inside, lin, -1)
    rec_val = tl.where(inside, dE * w, 0.0)
    tl.store(rec_lin_ptr + offs, rec_lin.to(tl.int32), mask=True)
    tl.store(rec_val_ptr + offs, rec_val.to(tl.float32), mask=True)

    E2 = E - dE
    z2 = z + ds * uz
    y2 = y + ds * uy
    x2 = x + ds * ux

    stop = inside & (E2 < e_cut_MeV)
    alive = inside & (E2 >= e_cut_MeV)

    lam_b = tl.load(P_brem_per_cm_ptr + off, mask=inside, other=0.0) * rho_scale
    lam_d = tl.load(P_delta_per_cm_ptr + off, mask=inside, other=0.0) * rho_scale
    Pb = 1.0 - tl.exp(-lam_b * ds)
    Pd = 1.0 - tl.exp(-lam_d * ds)

    u1, rng = rand_uniform_u01(rng)
    u2, rng = rand_uniform_u01(rng)
    emit_brem = alive & (u1 < Pb)
    emit_delta = alive & (u2 < Pd)

    tl.store(out_pos_ptr + offs * 3 + 0, z2, mask=True)
    tl.store(out_pos_ptr + offs * 3 + 1, y2, mask=True)
    tl.store(out_pos_ptr + offs * 3 + 2, x2, mask=True)
    tl.store(out_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_E_ptr + offs, E2, mask=True)
    tl.store(out_w_ptr + offs, w, mask=True)
    tl.store(out_rng_ptr + offs, rng, mask=True)
    tl.store(out_ebin_ptr + offs, ebin, mask=True)

    tl.store(out_alive_ptr + offs, alive.to(tl.int8), mask=True)
    tl.store(out_emit_brem_ptr + offs, emit_brem.to(tl.int8), mask=True)
    tl.store(out_emit_delta_ptr + offs, emit_delta.to(tl.int8), mask=True)
    tl.store(out_stop_ptr + offs, stop.to(tl.int8), mask=True)