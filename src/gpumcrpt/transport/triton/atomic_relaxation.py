from __future__ import annotations

import triton
import triton.language as tl
from gpumcrpt.transport.triton.rng import rand_uniform_u01


@triton.jit
def atomic_relaxation_kernel(
    vac_pos_ptr, vac_mat_ptr, vac_shell_ptr, vac_w_ptr, vac_rng_ptr,
    # relaxation tables:
    fluor_yield_ptr,   # float32 [M,S]
    E_xray_ptr,        # float32 [M,S] MeV
    E_auger_ptr,       # float32 [M,S] MeV
    M: tl.constexpr,
    S: tl.constexpr,
    # cutoffs:
    photon_cut_MeV: tl.constexpr,
    e_cut_MeV: tl.constexpr,
    # outputs: photons and electrons (one slot each; use has_* masks)
    out_ph_pos_ptr, out_ph_dir_ptr, out_ph_E_ptr, out_ph_w_ptr, out_ph_rng_ptr, out_has_ph_ptr,
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr, out_e_rng_ptr, out_has_e_ptr,
    # local deposit if below cutoff:
    edep_ptr,
    material_id_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    z = tl.load(vac_pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(vac_pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(vac_pos_ptr + offs * 3 + 2, mask=True, other=0.0)

    mat = tl.load(vac_mat_ptr + offs, mask=True, other=0).to(tl.int32)
    shell = tl.load(vac_shell_ptr + offs, mask=True, other=0).to(tl.int32)
    w = tl.load(vac_w_ptr + offs, mask=True, other=0.0).to(tl.float32)
    rng = tl.load(vac_rng_ptr + offs, mask=True, other=1234567).to(tl.int32)

    mat = tl.maximum(0, tl.minimum(mat, M - 1))
    shell = tl.maximum(0, tl.minimum(shell, S - 1))
    base = mat * S + shell

    fy = tl.load(fluor_yield_ptr + base, mask=True, other=0.0).to(tl.float32)
    Ex = tl.load(E_xray_ptr + base, mask=True, other=0.0).to(tl.float32)
    Ea = tl.load(E_auger_ptr + base, mask=True, other=0.0).to(tl.float32)

    u, rng = rand_uniform_u01(rng)
    emit_x = u < fy

    # isotropic direction for X-ray / Auger electron
    u1, rng = rand_uniform_u01(rng)
    u2, rng = rand_uniform_u01(rng)
    cost = 2.0 * u1 - 1.0
    sint = tl.sqrt(tl.maximum(0.0, 1.0 - cost * cost))
    phi = 2.0 * tl.pi * u2
    ux = sint * tl.cos(phi)
    uy = sint * tl.sin(phi)
    uz = cost

    # Determine voxel for local deposit if needed
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    # X-ray branch
    has_ph = emit_x & (Ex >= photon_cut_MeV) & inside
    dep_x = emit_x & ((Ex < photon_cut_MeV) | (~inside))

    # Auger branch
    has_e = (~emit_x) & (Ea >= e_cut_MeV) & inside
    dep_e = (~emit_x) & ((Ea < e_cut_MeV) | (~inside))

    # deposit locally if below cutoff or out of bounds
    depE = tl.where(dep_x, Ex * w, 0.0) + tl.where(dep_e, Ea * w, 0.0)
    tl.atomic_add(edep_ptr + lin, depE, mask=inside & (depE > 0.0))

    # output photon
    tl.store(out_ph_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_ph_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_ph_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_ph_E_ptr + offs, Ex, mask=True)
    tl.store(out_ph_w_ptr + offs, w, mask=True)
    tl.store(out_ph_rng_ptr + offs, rng, mask=True)
    tl.store(out_has_ph_ptr + offs, has_ph.to(tl.int8), mask=True)

    # output electron (Auger)
    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_e_E_ptr + offs, Ea, mask=True)
    tl.store(out_e_w_ptr + offs, w, mask=True)
    tl.store(out_e_rng_ptr + offs, rng, mask=True)
    tl.store(out_has_e_ptr + offs, has_e.to(tl.int8), mask=True)