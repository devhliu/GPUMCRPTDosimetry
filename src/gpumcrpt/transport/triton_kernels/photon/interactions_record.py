from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.rng import init_philox_state, rand_uniform4
from gpumcrpt.transport.triton_kernels.utils.geometry import rotate_dir_kernel
from gpumcrpt.utils.constants import ELECTRON_REST_MASS_MEV, PI


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 2}),
    ],
    key=['N'],
    warmup=10,
    rep=20,
)
@triton.jit
def photon_interaction_record_kernel(
    real_ptr, pos_ptr, dir_ptr, E_ptr, w_ptr, ebin_ptr,
    material_id_ptr, rho_ptr, ref_rho_ptr,
    sigma_photo_ptr, sigma_compton_ptr, sigma_rayleigh_ptr, sigma_pair_ptr,
    compton_inv_cdf_ptr, K: tl.constexpr,
    rng_seed: tl.tensor,
    out_ph_pos_ptr, out_ph_dir_ptr, out_ph_E_ptr, out_ph_w_ptr, out_ph_ebin_ptr,
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr,
    out_po_pos_ptr, out_po_dir_ptr, out_po_E_ptr, out_po_w_ptr,
    rec_lin_ph_ptr, rec_val_ph_ptr, rec_lin_e_ptr, rec_val_e_ptr, rec_lin_po_ptr, rec_val_po_ptr,
    rec_offs_ph_ptr, rec_offs_e_ptr, rec_offs_po_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr, ECOUNT: tl.constexpr,
    N: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Kernel for photon classification and interaction with record buffer outputs.
    Used in record mode for sorted_voxel tally to avoid atomic operations.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    real = tl.load(real_ptr + offs, mask=mask, other=0).to(tl.int1)
    E = tl.load(E_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    ebin = tl.load(ebin_ptr + offs, mask=mask, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=0.0)
    uz = tl.load(dir_ptr + offs * 3 + 0, mask=mask, other=1.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=mask, other=0.0)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=mask, other=0.0)

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

    particle_id = offs.to(tl.uint32)
    seed = tl.full(offs.shape, rng_seed, dtype=tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, seed)

    u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
    t = u1 * tl.maximum(s_tot, 1e-12)

    is_pe = t < s_pe
    is_co = (~is_pe) & (t < s_pe + s_co)
    is_ra = (~is_pe) & (~is_co) & (t < s_pe + s_co + s_ra)
    is_pa = (~is_pe) & (~is_co) & (~is_ra) & (t <= s_tot)

    tl.store(out_ph_pos_ptr + offs * 3 + 0, z, mask=mask)
    tl.store(out_ph_pos_ptr + offs * 3 + 1, y, mask=mask)
    tl.store(out_ph_pos_ptr + offs * 3 + 2, x, mask=mask)
    
    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=mask)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=mask)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=mask)
    
    tl.store(out_po_pos_ptr + offs * 3 + 0, z, mask=mask)
    tl.store(out_po_pos_ptr + offs * 3 + 1, y, mask=mask)
    tl.store(out_po_pos_ptr + offs * 3 + 2, x, mask=mask)

    pe_mask = real & inside & is_pe
    tl.store(out_e_dir_ptr + offs * 3 + 0, uz, mask=pe_mask)
    tl.store(out_e_dir_ptr + offs * 3 + 1, uy, mask=pe_mask)
    tl.store(out_e_dir_ptr + offs * 3 + 2, ux, mask=pe_mask)
    tl.store(out_e_E_ptr + offs, E, mask=pe_mask)
    tl.store(out_e_w_ptr + offs, w, mask=pe_mask)
    
    tl.store(out_ph_E_ptr + offs, 0.0, mask=pe_mask)
    tl.store(out_ph_w_ptr + offs, 0.0, mask=pe_mask)
    tl.store(out_ph_ebin_ptr + offs, ebin, mask=pe_mask)

    co_mask = real & inside & is_co
    if tl.sum(co_mask.to(tl.int32)) > 0:
        u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
        t = u1 * (K - 1)
        i0 = tl.floor(t).to(tl.int32)
        i0 = tl.maximum(0, tl.minimum(i0, K - 2))
        f = t - i0.to(tl.float32)

        base = ebin * K + i0
        c0_val = tl.load(compton_inv_cdf_ptr + base, mask=co_mask, other=1.0)
        c1 = tl.load(compton_inv_cdf_ptr + base + 1, mask=co_mask, other=1.0)
        cos_t = c0_val + f * (c1 - c0_val)
        cos_t = tl.maximum(-1.0, tl.minimum(cos_t, 1.0))

        phi = 2.0 * PI * u2

        nuz, nuy, nux = rotate_dir_kernel(uz, uy, ux, cos_t, phi)

        alpha = E / ELECTRON_REST_MASS_MEV
        Eratio = 1.0 / (1.0 + alpha * (1.0 - cos_t))
        E_scat = E * Eratio
        T = E - E_scat

        pez = E * uz - E_scat * nuz
        pey = E * uy - E_scat * nuy
        pex = E * ux - E_scat * nux
        
        pe2 = pez * pez + pey * pey + pex * pex
        pe_norm = tl.sqrt(tl.maximum(pe2, 1e-30))
        
        e_uz = pez / pe_norm
        e_uy = pey / pe_norm
        e_ux = pex / pe_norm

        tl.store(out_ph_dir_ptr + offs * 3 + 0, nuz, mask=co_mask)
        tl.store(out_ph_dir_ptr + offs * 3 + 1, nuy, mask=co_mask)
        tl.store(out_ph_dir_ptr + offs * 3 + 2, nux, mask=co_mask)
        tl.store(out_ph_E_ptr + offs, E_scat, mask=co_mask)
        tl.store(out_ph_w_ptr + offs, w, mask=co_mask)
        tl.store(out_ph_ebin_ptr + offs, ebin, mask=co_mask)

        tl.store(out_e_dir_ptr + offs * 3 + 0, e_uz, mask=co_mask)
        tl.store(out_e_dir_ptr + offs * 3 + 1, e_uy, mask=co_mask)
        tl.store(out_e_dir_ptr + offs * 3 + 2, e_ux, mask=co_mask)
        tl.store(out_e_E_ptr + offs, T, mask=co_mask)
        tl.store(out_e_w_ptr + offs, w, mask=co_mask)

    ra_mask = real & inside & is_ra
    if tl.sum(ra_mask.to(tl.int32)) > 0:
        u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

        u = tl.where(ra_mask, u1, 0.5)
        mu = 2.0 * u - 1.0
        for _ in tl.static_range(4):
            F = 0.5 + 0.375 * (mu + (mu * mu * mu) / 3.0)
            fp = 0.375 * (1.0 + mu * mu)
            mu = mu - (F - u) / tl.maximum(fp, 1e-12)
            mu = tl.maximum(-1.0, tl.minimum(mu, 1.0))

        cos_t = mu
        phi = 2.0 * PI * u2
        nuz, nuy, nux = rotate_dir_kernel(uz, uy, ux, cos_t, phi)

        tl.store(out_ph_dir_ptr + offs * 3 + 0, nuz, mask=ra_mask)
        tl.store(out_ph_dir_ptr + offs * 3 + 1, nuy, mask=ra_mask)
        tl.store(out_ph_dir_ptr + offs * 3 + 2, nux, mask=ra_mask)
        tl.store(out_ph_E_ptr + offs, E, mask=ra_mask)
        tl.store(out_ph_w_ptr + offs, w, mask=ra_mask)
        tl.store(out_ph_ebin_ptr + offs, ebin, mask=ra_mask)

    pa_mask = real & inside & is_pa
    if tl.sum(pa_mask.to(tl.int32)) > 0:
        K_pair = tl.maximum(E - 1.022, 0.0)

        u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
        frac = u1
        Ee = K_pair * frac
        Ep = K_pair - Ee

        tl.store(out_e_dir_ptr + offs * 3 + 0, uz, mask=pa_mask)
        tl.store(out_e_dir_ptr + offs * 3 + 1, uy, mask=pa_mask)
        tl.store(out_e_dir_ptr + offs * 3 + 2, ux, mask=pa_mask)
        tl.store(out_e_E_ptr + offs, Ee, mask=pa_mask)
        tl.store(out_e_w_ptr + offs, w, mask=pa_mask)

        tl.store(out_po_dir_ptr + offs * 3 + 0, -uz, mask=pa_mask)
        tl.store(out_po_dir_ptr + offs * 3 + 1, -uy, mask=pa_mask)
        tl.store(out_po_dir_ptr + offs * 3 + 2, -ux, mask=pa_mask)
        tl.store(out_po_E_ptr + offs, Ep, mask=pa_mask)
        tl.store(out_po_w_ptr + offs, w, mask=pa_mask)

        tl.store(out_ph_E_ptr + offs, 0.0, mask=pa_mask)
        tl.store(out_ph_w_ptr + offs, 0.0, mask=pa_mask)
