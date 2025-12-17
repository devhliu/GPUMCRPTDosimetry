from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.kernels.triton.rng_philox import rng_u01_philox


@triton.jit
def photon_photoelectric_pe_soa_kernel(
    # inputs (packed staging SoA, length Npe)
    in_x_ptr, in_y_ptr, in_z_ptr,
    in_dx_ptr, in_dy_ptr, in_dz_ptr,
    in_E_ptr, in_w_ptr,
    in_ebin_ptr,  # not used in PE itself but kept for interface consistency

    # Philox RNG per photon (SoA)
    in_rng_key0_ptr, in_rng_key1_ptr,
    in_rng_ctr0_ptr, in_rng_ctr1_ptr, in_rng_ctr2_ptr, in_rng_ctr3_ptr,

    # geometry/material
    material_id_ptr,              # int32 [Z*Y*X]
    material_atom_Z_ptr,          # int32 [M]

    # PE shell tables by material: [M,S]
    shell_cdf_ptr,                # fp32 [M,S]
    E_bind_MeV_ptr,               # fp32 [M,S]

    # outputs: photoelectron staging (SoA)
    out_e_x_ptr, out_e_y_ptr, out_e_z_ptr,
    out_e_dx_ptr, out_e_dy_ptr, out_e_dz_ptr,
    out_e_E_ptr, out_e_w_ptr,
    out_e_rng_key0_ptr, out_e_rng_key1_ptr,
    out_e_rng_ctr0_ptr, out_e_rng_ctr1_ptr, out_e_rng_ctr2_ptr, out_e_rng_ctr3_ptr,
    out_e_has_ptr,                # int8

    # outputs: vacancy staging (SoA) (no RNG here; engine passes e RNG into vacancy append)
    out_v_x_ptr, out_v_y_ptr, out_v_z_ptr,
    out_v_atom_Z_ptr,             # int32
    out_v_shell_idx_ptr,          # int32
    out_v_has_ptr,                # int8

    edep_flat_ptr,

    # constants
    Npe: tl.constexpr,
    Zdim: tl.constexpr, Ydim: tl.constexpr, Xdim: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    voxel_x_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_z_cm: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < Npe

    x = tl.load(in_x_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(in_y_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(in_z_ptr + i, mask=m, other=0.0).to(tl.float32)

    dx = tl.load(in_dx_ptr + i, mask=m, other=1.0).to(tl.float32)
    dy = tl.load(in_dy_ptr + i, mask=m, other=0.0).to(tl.float32)
    dz = tl.load(in_dz_ptr + i, mask=m, other=0.0).to(tl.float32)

    E = tl.load(in_E_ptr + i, mask=m, other=0.0).to(tl.float32)
    w = tl.load(in_w_ptr + i, mask=m, other=0.0).to(tl.float32)

    key0 = tl.load(in_rng_key0_ptr + i, mask=m, other=0).to(tl.int32)
    key1 = tl.load(in_rng_key1_ptr + i, mask=m, other=0).to(tl.int32)
    ctr0 = tl.load(in_rng_ctr0_ptr + i, mask=m, other=0).to(tl.int32)
    ctr1 = tl.load(in_rng_ctr1_ptr + i, mask=m, other=0).to(tl.int32)
    ctr2 = tl.load(in_rng_ctr2_ptr + i, mask=m, other=0).to(tl.int32)
    ctr3 = tl.load(in_rng_ctr3_ptr + i, mask=m, other=0).to(tl.int32)

    u0, _, _, _, ctr0u, ctr1u, ctr2u, ctr3u = rng_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3)

    # voxel lookup
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    inside = (ix >= 0) & (ix < Xdim) & (iy >= 0) & (iy < Ydim) & (iz >= 0) & (iz < Zdim)
    lin = iz * (Ydim * Xdim) + iy * Xdim + ix

    mat = tl.load(material_id_ptr + lin, mask=m & inside, other=0).to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))
    base = mat * S

    atom_Z = tl.load(material_atom_Z_ptr + mat, mask=m, other=0).to(tl.int32)

    shell = tl.zeros([BLOCK], dtype=tl.int32)
    c0 = tl.load(shell_cdf_ptr + base + 0, mask=m, other=1.0).to(tl.float32)
    shell += (u0 > c0).to(tl.int32)
    if tl.constexpr(S) > 1:
        c1 = tl.load(shell_cdf_ptr + base + 1, mask=m, other=1.0).to(tl.float32)
        shell += (u0 > c1).to(tl.int32)
    if tl.constexpr(S) > 2:
        c2 = tl.load(shell_cdf_ptr + base + 2, mask=m, other=1.0).to(tl.float32)
        shell += (u0 > c2).to(tl.int32)
    if tl.constexpr(S) > 3:
        c3 = tl.load(shell_cdf_ptr + base + 3, mask=m, other=1.0).to(tl.float32)
        shell += (u0 > c3).to(tl.int32)
    shell = tl.minimum(shell, S - 1)

    Ebind = tl.load(E_bind_MeV_ptr + base + shell, mask=m, other=0.0).to(tl.float32)
    Ee = tl.maximum(E - Ebind, 0.0)

    has_e = m & inside
    has_v = m & inside & (Ebind > 0.0)

    # electron staging
    tl.store(out_e_x_ptr + i, x, mask=m)
    tl.store(out_e_y_ptr + i, y, mask=m)
    tl.store(out_e_z_ptr + i, z, mask=m)
    tl.store(out_e_dx_ptr + i, dx, mask=m)
    tl.store(out_e_dy_ptr + i, dy, mask=m)
    tl.store(out_e_dz_ptr + i, dz, mask=m)
    tl.store(out_e_E_ptr + i, Ee, mask=m)
    tl.store(out_e_w_ptr + i, w, mask=m)
    tl.store(out_e_rng_key0_ptr + i, key0, mask=m)
    tl.store(out_e_rng_key1_ptr + i, key1, mask=m)
    tl.store(out_e_rng_ctr0_ptr + i, ctr0u.to(tl.int32), mask=m)
    tl.store(out_e_rng_ctr1_ptr + i, ctr1u.to(tl.int32), mask=m)
    tl.store(out_e_rng_ctr2_ptr + i, ctr2u.to(tl.int32), mask=m)
    tl.store(out_e_rng_ctr3_ptr + i, ctr3u.to(tl.int32), mask=m)
    tl.store(out_e_has_ptr + i, has_e.to(tl.int8), mask=m)

    # vacancy staging
    tl.store(out_v_x_ptr + i, x, mask=m)
    tl.store(out_v_y_ptr + i, y, mask=m)
    tl.store(out_v_z_ptr + i, z, mask=m)
    tl.store(out_v_atom_Z_ptr + i, atom_Z, mask=m)
    tl.store(out_v_shell_idx_ptr + i, shell, mask=m)
    tl.store(out_v_has_ptr + i, has_v.to(tl.int8), mask=m)