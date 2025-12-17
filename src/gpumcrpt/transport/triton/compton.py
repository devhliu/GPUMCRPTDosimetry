from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01


@triton.jit
def _rotate_dir_kernel(
    uz, uy, ux, cos_t, phi
):
    # Build basis and rotate; returns new (uz,uy,ux)
    sin_t = tl.sqrt(tl.maximum(1.0 - cos_t * cos_t, 0.0))
    cos_p = tl.cos(phi)
    sin_p = tl.sin(phi)

    # pick helper vector v depending on |uz|
    use_a = tl.abs(uz) < 0.9
    va = tl.where(use_a, 1.0, 0.0)
    vb = tl.where(use_a, 0.0, 1.0)

    # e1 = cross(u, v)
    e1z = uy * 0.0 - ux * vb
    e1y = ux * va - uz * 0.0
    e1x = uz * vb - uy * va
    n1 = tl.sqrt(tl.maximum(e1z * e1z + e1y * e1y + e1x * e1x, 1e-20))
    e1z, e1y, e1x = e1z / n1, e1y / n1, e1x / n1

    # e2 = cross(e1, u)
    e2z = e1y * ux - e1x * uy
    e2y = e1x * uz - e1z * ux
    e2x = e1z * uy - e1y * uz

    nz = cos_t * uz + sin_t * (cos_p * e1z + sin_p * e2z)
    ny = cos_t * uy + sin_t * (cos_p * e1y + sin_p * e2y)
    nx = cos_t * ux + sin_t * (cos_p * e1x + sin_p * e2x)

    nn = tl.sqrt(tl.maximum(nz * nz + ny * ny + nx * nx, 1e-20))
    return nz / nn, ny / nn, nx / nn


@triton.jit
def photon_compton_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    compton_inv_cdf_ptr, K: tl.constexpr,
    # outputs:
    out_ph_pos_ptr, out_ph_dir_ptr, out_ph_E_ptr, out_ph_w_ptr, out_ph_rng_ptr, out_ph_ebin_ptr,
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr,
    ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Compton scattering using inverse-CDF sampler stored as cos(theta):
      cos_t = inv_cdf[ebin, lerp(u)]
    K = number of samples in inverse-CDF for each energy bin.

    Kinematics:
      E' = E / (1 + (E/me)*(1 - cosθ))
      T = E - E'
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

    u, rng = rand_uniform_u01(rng)
    t = u * (K - 1)
    i0 = tl.floor(t).to(tl.int32)
    i0 = tl.maximum(0, tl.minimum(i0, K - 2))
    f = t - i0.to(tl.float32)

    base = ebin * K + i0
    c0 = tl.load(compton_inv_cdf_ptr + base, mask=True, other=1.0)
    c1 = tl.load(compton_inv_cdf_ptr + base + 1, mask=True, other=1.0)
    cos_t = c0 + f * (c1 - c0)
    cos_t = tl.maximum(-1.0, tl.minimum(cos_t, 1.0))

    # sample azimuth
    u2, rng = rand_uniform_u01(rng)
    phi = 2.0 * 3.1415926535 * u2

    # scattered photon direction
    nuz, nuy, nux = _rotate_dir_kernel(uz, uy, ux, cos_t, phi)

    me = 0.51099895
    alpha = E / me
    Eratio = 1.0 / (1.0 + alpha * (1.0 - cos_t))
    E_scat = E * Eratio
    T = E - E_scat

    # recoil electron kinematics
    # Conservation of momentum: p_gamma = p_gamma' + p_e
    # p_e = p_gamma - p_gamma'
    # Energy units: p = E * dir
    # Note: uz/uy/ux order in code corresponds to offsets 0, 1, 2 (Z, Y, X)
    
    pez = E * uz - E_scat * nuz
    pey = E * uy - E_scat * nuy
    pex = E * ux - E_scat * nux
    
    pe2 = pez * pez + pey * pey + pex * pex
    pe_norm = tl.sqrt(tl.maximum(pe2, 1e-30))
    
    e_uz = pez / pe_norm
    e_uy = pey / pe_norm
    e_ux = pex / pe_norm

    # write scattered photon
    tl.store(out_ph_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_ph_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_ph_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 0, nuz, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 1, nuy, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 2, nux, mask=True)
    tl.store(out_ph_E_ptr + offs, E_scat, mask=True)
    tl.store(out_ph_w_ptr + offs, w, mask=True)
    tl.store(out_ph_rng_ptr + offs, rng, mask=True)
    tl.store(out_ph_ebin_ptr + offs, ebin, mask=True)

    # recoil electron
    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 0, e_uz, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 1, e_uy, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 2, e_ux, mask=True)
    tl.store(out_e_E_ptr + offs, T, mask=True)
    tl.store(out_e_w_ptr + offs, w, mask=True)