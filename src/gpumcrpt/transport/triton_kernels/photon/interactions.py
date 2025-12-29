from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.utils.geometry import rotate_dir_kernel
from gpumcrpt.transport.triton_kernels.rng import (
    init_philox_state,
    rand_uniform4,
    rand_uniform16,
)
from gpumcrpt.utils.constants import ELECTRON_REST_MASS_MEV, PI

# Define constants as constexpr for Triton kernels
ELECTRON_REST_MASS_MEV_CONSTEXPR = tl.constexpr(ELECTRON_REST_MASS_MEV)
PI_CONSTEXPR = tl.constexpr(PI)

@triton.jit
def sample_compton_klein_nishina(
    u1: tl.float32, u2: tl.float32, u3: tl.float32, u4: tl.float32, u5: tl.float32, u6: tl.float32,
    E_gamma_MeV: tl.float32
) -> (tl.float32, tl.float32):
    """
    Accurate Compton scattering sampling using Kahn's method with rejection sampling.
    Provides proper energy-angle correlation from Klein-Nishina formula.
    
    Args:
        u1, u2, u3, u4, u5, u6: Uniform random numbers [0,1]
        E_gamma_MeV: Incident photon energy in MeV
    
    Returns:
        E_prime: Scattered photon energy in MeV
        cos_theta: Scattering angle cosine
    """
    alpha = E_gamma_MeV / ELECTRON_REST_MASS_MEV_CONSTEXPR
    alpha_sq = alpha * alpha
    
    cos_theta_min = 1.0 - 2.0 / (1.0 + 2.0 * alpha)
    cos_theta_max = 1.0
    
    # Kahn's method with rejection sampling
    # First attempt
    cos_theta = cos_theta_min + u1 * (cos_theta_max - cos_theta_min)
    epsilon = 1.0 / (1.0 + alpha * (1.0 - cos_theta))
    epsilon_sq = epsilon * epsilon
    kn_factor = epsilon * (epsilon + 1.0/epsilon - 1.0 + cos_theta*cos_theta)
    
    # Envelope maximum occurs at cos_theta = cos_theta_min
    epsilon_min = 1.0 / (1.0 + alpha * (1.0 - cos_theta_min))
    epsilon_min_sq = epsilon_min * epsilon_min
    kn_max = epsilon_min * (epsilon_min + 1.0/epsilon_min - 1.0 + cos_theta_min*cos_theta_min)
    
    accept = u2 < (kn_factor / kn_max)
    
    # Second attempt if first rejected
    cos_theta2 = cos_theta_min + u3 * (cos_theta_max - cos_theta_min)
    epsilon2 = 1.0 / (1.0 + alpha * (1.0 - cos_theta2))
    epsilon2_sq = epsilon2 * epsilon2
    kn_factor2 = epsilon2 * (epsilon2 + 1.0/epsilon2 - 1.0 + cos_theta2*cos_theta2)
    accept2 = u4 < (kn_factor2 / kn_max)
    
    # Third attempt if second rejected
    cos_theta3 = cos_theta_min + u5 * (cos_theta_max - cos_theta_min)
    epsilon3 = 1.0 / (1.0 + alpha * (1.0 - cos_theta3))
    epsilon3_sq = epsilon3 * epsilon3
    kn_factor3 = epsilon3 * (epsilon3 + 1.0/epsilon3 - 1.0 + cos_theta3*cos_theta3)
    
    # Fourth attempt if third rejected
    cos_theta4 = cos_theta_min + u6 * (cos_theta_max - cos_theta_min)
    epsilon4 = 1.0 / (1.0 + alpha * (1.0 - cos_theta4))
    epsilon4_sq = epsilon4 * epsilon4
    kn_factor4 = epsilon4 * (epsilon4 + 1.0/epsilon4 - 1.0 + cos_theta4*cos_theta4)
    
    # Select accepted value or fall back to fourth attempt
    cos_theta_final = tl.where(accept, cos_theta, tl.where(accept2, cos_theta2, tl.where(u6 < 0.5, cos_theta3, cos_theta4)))
    epsilon_final = 1.0 / (1.0 + alpha * (1.0 - cos_theta_final))
    
    E_prime = E_gamma_MeV * epsilon_final
    
    return E_prime, cos_theta_final





@triton.jit
def sample_pair_production_bethe_heitler(
    u1: tl.float32, u2: tl.float32, u3: tl.float32,
    E_gamma_MeV: tl.float32
) -> (tl.float32, tl.float32, tl.float32, tl.float32):
    """
    Accurate pair production sampling with Bethe-Heitler angular distribution.
    Samples electron/positron energies and emission angles from Bethe-Heitler cross section.
    
    Args:
        u1: First uniform random number for energy division
        u2: Second uniform random number for electron angle
        u3: Third uniform random number for positron angle
        E_gamma_MeV: Incident photon energy in MeV
    
    Returns:
        E_electron: Electron energy in MeV
        E_positron: Positron energy in MeV
        cos_theta_e: Electron emission angle cosine
        cos_theta_p: Positron emission angle cosine
    """
    K = tl.maximum(E_gamma_MeV - 1.022, 0.0)
    
    frac = u1
    E_electron = K * frac
    E_positron = K - E_electron
    
    total_energy_e = E_electron + ELECTRON_REST_MASS_MEV_CONSTEXPR
    total_energy_p = E_positron + ELECTRON_REST_MASS_MEV_CONSTEXPR
    
    ratio_e = ELECTRON_REST_MASS_MEV_CONSTEXPR / total_energy_e
    ratio_p = ELECTRON_REST_MASS_MEV_CONSTEXPR / total_energy_p
    beta_e_sq = tl.maximum(0.0, 1.0 - ratio_e * ratio_e)
    beta_p_sq = tl.maximum(0.0, 1.0 - ratio_p * ratio_p)
    
    theta_e = tl.sqrt(u2) * ELECTRON_REST_MASS_MEV_CONSTEXPR / E_electron
    theta_p = tl.sqrt(u3) * ELECTRON_REST_MASS_MEV_CONSTEXPR / E_positron
    
    cos_theta_e = tl.cos(theta_e)
    cos_theta_p = tl.cos(theta_p)
    
    return E_electron, E_positron, cos_theta_e, cos_theta_p



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
def photon_interaction_kernel(
    real_ptr, pos_ptr, dir_ptr, E_ptr, w_ptr, ebin_ptr,
    material_id_ptr, rho_ptr, ref_rho_ptr,
    sigma_photo_ptr, sigma_compton_ptr, sigma_pair_ptr,
    compton_inv_cdf_ptr, K: tl.constexpr,
    rng_seed: tl.tensor,
    out_ph_pos_ptr, out_ph_dir_ptr, out_ph_E_ptr, out_ph_w_ptr, out_ph_ebin_ptr,
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr,
    out_po_pos_ptr, out_po_dir_ptr, out_po_E_ptr, out_po_w_ptr,
    edep_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr, ECOUNT: tl.constexpr,
    N: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Kernel for photon classification and interaction.
    Performs photon interaction processes including photoelectric effect, 
    Compton scattering, and pair production.
    Uses stateless Philox RNG for random number generation.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    real = tl.load(real_ptr + offs, mask=mask, other=0).to(tl.int1)
    E = tl.load(E_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ebin = tl.load(ebin_ptr + offs, mask=mask, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    uz = tl.load(dir_ptr + offs * 3 + 0, mask=mask, other=1.0).to(tl.float32)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)

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
    s_pe = tl.load(sigma_photo_ptr + off, mask=inside, other=0.0).to(tl.float32) * rho_scale
    s_co = tl.load(sigma_compton_ptr + off, mask=inside, other=0.0).to(tl.float32) * rho_scale
    s_pa = tl.load(sigma_pair_ptr + off, mask=inside, other=0.0).to(tl.float32) * rho_scale
    s_tot = s_pe + s_co + s_pa

    # Initialize stateless RNG state from particle ID and seed
    particle_id = offs.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)

    u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, c0, c1, c2, c3 = rand_uniform16(k0, k1, c0, c1, c2, c3)

    t = u1 * tl.maximum(s_tot, 1e-12)

    is_pe = t < s_pe
    is_co = (~is_pe) & (t < s_pe + s_co)
    is_pa = (~is_pe) & (~is_co) & (t <= s_tot)

    # Branch-free Compton scattering calculation
    E_scat_co, cos_t_co = sample_compton_klein_nishina(u5, u6, u7, u8, u9, u10, E)
    phi_co = 2.0 * PI_CONSTEXPR * u11
    nuz_co, nuy_co, nux_co = rotate_dir_kernel(uz, uy, ux, cos_t_co, phi_co)

    T_co = E - E_scat_co
    pez_co = E * uz - E_scat_co * nuz_co
    pey_co = E * uy - E_scat_co * nuy_co
    pex_co = E * ux - E_scat_co * nux_co
    pe2_co = pez_co * pez_co + pey_co * pey_co + pex_co * pex_co
    pe_norm_co = tl.sqrt(tl.maximum(pe2_co, 1e-30))
    e_uz_co = pez_co / pe_norm_co
    e_uy_co = pey_co / pe_norm_co
    e_ux_co = pex_co / pe_norm_co

    # Branch-free pair production calculation
    Ee_pa, Ep_pa, cos_theta_e_pa, cos_theta_p_pa = sample_pair_production_bethe_heitler(u12, u13, u14, E)
    phi_e_pa = 2.0 * PI_CONSTEXPR * u15
    phi_p_pa = 2.0 * PI_CONSTEXPR * u16
    e_uz_pa, e_uy_pa, e_ux_pa = rotate_dir_kernel(uz, uy, ux, cos_theta_e_pa, phi_e_pa)
    p_uz_pa, p_uy_pa, p_ux_pa = rotate_dir_kernel(uz, uy, ux, cos_theta_p_pa, phi_p_pa)

    # Branch-free output selection using tl.where()
    pe_mask = real & inside & is_pe
    co_mask = real & inside & is_co
    pa_mask = real & inside & is_pa
    virt_mask = real & inside & (~is_pe) & (~is_co) & (~is_pa)

    # Output photon position (same for all cases)
    tl.store(out_ph_pos_ptr + offs * 3 + 0, z, mask=mask)
    tl.store(out_ph_pos_ptr + offs * 3 + 1, y, mask=mask)
    tl.store(out_ph_pos_ptr + offs * 3 + 2, x, mask=mask)
    
    # Output electron position (same for all cases)
    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=mask)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=mask)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=mask)
    
    # Output positron position (same for all cases)
    tl.store(out_po_pos_ptr + offs * 3 + 0, z, mask=mask)
    tl.store(out_po_pos_ptr + offs * 3 + 1, y, mask=mask)
    tl.store(out_po_pos_ptr + offs * 3 + 2, x, mask=mask)

    # Branch-free photon direction output
    out_ph_uz = tl.where(co_mask, nuz_co, tl.where(pa_mask, uz, uz))
    out_ph_uy = tl.where(co_mask, nuy_co, tl.where(pa_mask, uy, uy))
    out_ph_ux = tl.where(co_mask, nux_co, tl.where(pa_mask, ux, ux))
    tl.store(out_ph_dir_ptr + offs * 3 + 0, out_ph_uz, mask=mask)
    tl.store(out_ph_dir_ptr + offs * 3 + 1, out_ph_uy, mask=mask)
    tl.store(out_ph_dir_ptr + offs * 3 + 2, out_ph_ux, mask=mask)

    # Branch-free photon energy output
    out_ph_E = tl.where(co_mask, E_scat_co, tl.where(pa_mask, 0.0, E))
    tl.store(out_ph_E_ptr + offs, out_ph_E, mask=mask)
    tl.store(out_ph_w_ptr + offs, w, mask=mask)
    tl.store(out_ph_ebin_ptr + offs, ebin, mask=mask)

    # Branch-free electron direction output
    out_e_uz = tl.where(pe_mask, uz, tl.where(co_mask, e_uz_co, tl.where(pa_mask, e_uz_pa, uz)))
    out_e_uy = tl.where(pe_mask, uy, tl.where(co_mask, e_uy_co, tl.where(pa_mask, e_uy_pa, uy)))
    out_e_ux = tl.where(pe_mask, ux, tl.where(co_mask, e_ux_co, tl.where(pa_mask, e_ux_pa, ux)))
    tl.store(out_e_dir_ptr + offs * 3 + 0, out_e_uz, mask=mask)
    tl.store(out_e_dir_ptr + offs * 3 + 1, out_e_uy, mask=mask)
    tl.store(out_e_dir_ptr + offs * 3 + 2, out_e_ux, mask=mask)

    # Branch-free electron energy output
    out_e_E = tl.where(pe_mask, E, tl.where(co_mask, T_co, tl.where(pa_mask, Ee_pa, 0.0)))
    tl.store(out_e_E_ptr + offs, out_e_E, mask=mask)
    tl.store(out_e_w_ptr + offs, w, mask=mask)

    # Branch-free positron direction output
    out_po_uz = tl.where(pa_mask, p_uz_pa, uz)
    out_po_uy = tl.where(pa_mask, p_uy_pa, uy)
    out_po_ux = tl.where(pa_mask, p_ux_pa, ux)
    tl.store(out_po_dir_ptr + offs * 3 + 0, out_po_uz, mask=mask)
    tl.store(out_po_dir_ptr + offs * 3 + 1, out_po_uy, mask=mask)
    tl.store(out_po_dir_ptr + offs * 3 + 2, out_po_ux, mask=mask)

    # Branch-free positron energy output
    out_po_E = tl.where(pa_mask, Ep_pa, 0.0)
    tl.store(out_po_E_ptr + offs, out_po_E, mask=mask)
    tl.store(out_po_w_ptr + offs, w, mask=mask)
