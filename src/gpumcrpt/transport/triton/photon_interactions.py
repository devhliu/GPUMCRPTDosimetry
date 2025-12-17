from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01


@triton.jit
def photon_classify_kernel(
    real_ptr, pos_ptr, E_ptr, ebin_ptr, rng_ptr,
    material_id_ptr, rho_ptr, ref_rho_ptr,
    sigma_photo_ptr, sigma_compton_ptr, sigma_rayleigh_ptr, sigma_pair_ptr,
    out_type_ptr, out_rng_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr, ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    """
    out_type codes:
      0 = virtual/ignore
      1 = photoelectric
      2 = compton
      3 = rayleigh
      4 = pair
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    real = tl.load(real_ptr + offs, mask=True, other=0).to(tl.int1)
    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    rng = tl.load(rng_ptr + offs, mask=True, other=123456789)
    ebin = tl.load(ebin_ptr + offs, mask=True, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)

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
    s_pe = tl.load(sigma_photo_ptr + off, mask=inside, other=0.0) * rho_scale
    s_co = tl.load(sigma_compton_ptr + off, mask=inside, other=0.0) * rho_scale
    s_ra = tl.load(sigma_rayleigh_ptr + off, mask=inside, other=0.0) * rho_scale
    s_pa = tl.load(sigma_pair_ptr + off, mask=inside, other=0.0) * rho_scale
    s_tot = s_pe + s_co + s_ra + s_pa

    u, rng = rand_uniform_u01(rng)
    t = u * tl.maximum(s_tot, 1e-12)

    is_pe = t < s_pe
    is_co = (~is_pe) & (t < s_pe + s_co)
    is_ra = (~is_pe) & (~is_co) & (t < s_pe + s_co + s_ra)
    is_pa = (~is_pe) & (~is_co) & (~is_ra) & (t <= s_tot)

    typ = tl.zeros([BLOCK], dtype=tl.int8)
    typ = tl.where(is_pe, 1, typ)
    typ = tl.where(is_co, 2, typ)
    typ = tl.where(is_ra, 3, typ)
    typ = tl.where(is_pa, 4, typ)

    typ = tl.where(real & inside, typ, 0)
    tl.store(out_type_ptr + offs, typ, mask=True)
    tl.store(out_rng_ptr + offs, rng, mask=True)


@triton.jit
def photon_photoelectric_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr,
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr,
    edep_ptr,  # flattened [Z*Y*X]
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    """
    Photoelectric:
      - photon killed
      - create photoelectron with T=E (binding ignored MVP, i.e. local binding)
      - no direct Edep here (aside from optional binding); electron will deposit via e- transport
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    w = tl.load(w_ptr + offs, mask=True, other=0.0)
    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)
    uz = tl.load(dir_ptr + offs * 3 + 0, mask=True, other=1.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=True, other=0.0)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=True, other=0.0)

    # output electron (same direction, same position)
    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_e_E_ptr + offs, E, mask=True)
    tl.store(out_e_w_ptr + offs, w, mask=True)