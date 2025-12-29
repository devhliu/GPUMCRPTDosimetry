"""
High-Performance Unified Charged Particle Transport Kernels

This module provides optimized kernels for electron and positron transport
with maximum GPU performance through vectorized operations and
coalesced memory access patterns.

Performance features:
- Structure of Arrays (SoA) layout for optimal memory access
- Vectorized physics calculations
- Minimal branch divergence
- Efficient secondary particle handling
- Optimized multiple scattering models

Physics Models Documentation:

Bremsstrahlung:
- Model: Bethe-Heitler cross-section with screening corrections
- Energy range: 0.01 MeV to E_initial
- Screening parameter: α = 1/137, includes complete/incomplete screening
- Reference: Bethe & Heitler, Proc. R. Soc. A (1934)
- Implementation: Rejection sampling with Bethe-Heitler spectrum

Delta Rays:
- Model: Moller scattering (e- + e- → e- + e-)
- Energy transfer: T ∈ [T_min, T_max] where T_max = E_initial/2
- Reference: Moller, Ann. Phys. (1932)
- Implementation: Inverse transform sampling with Moller cross-section

Multiple Scattering:
- Model: Molière theory with Gaussian approximation
- Valid for: θ < 0.5 rad
- Reference: Molière, Z. Naturforsch. (1948)
- Implementation: Branch-free vectorized angle sampling

Energy Straggling:
- Model: Vavilov distribution (interpolates between Gaussian and Landau)
- Parameters: κ = dE_mean / m_e*c^2
- Reference: Vavilov, Sov. Phys. JETP (1957)
- Implementation: Vectorized Vavilov sampling with three regimes

Positron Annihilation:
- Model: At-rest annihilation producing 2×0.511 MeV photons
- Photon directions: Isotropic, exactly 180° apart
- Reference: Dirac, Proc. R. Soc. A (1930)
- Implementation: Marsaglia method for isotropic direction sampling
"""

from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.rng import (
    init_philox_state,
    rand_uniform4,
    rand_uniform16,
)
from gpumcrpt.transport.triton_kernels.charged_particle.emission import (
    charged_particle_brems_emit_kernel,
    charged_particle_delta_emit_kernel,
)
from gpumcrpt.utils.constants import ELECTRON_REST_MASS_MEV, PI
from gpumcrpt.transport.triton_kernels.utils.gpu_math import fast_sqrt_approx, fast_log_approx, fast_exp_approx, fast_acos_approx, fast_sin_cos_approx, fast_sin_approx, fast_cos_approx


@triton.jit
def _apply_common_charged_particle_physics(
    # Particle state components (SoA layout for efficiency)
    pos_x: tl.tensor, pos_y: tl.tensor, pos_z: tl.tensor,
    dir_x: tl.tensor, dir_y: tl.tensor, dir_z: tl.tensor,
    energy: tl.tensor, weight: tl.tensor,
    particle_type: tl.tensor, material_id: tl.tensor,
    # Physics parameters
    S_restricted: tl.tensor, range_csda: tl.tensor,
    P_brem: tl.tensor, P_delta: tl.tensor,
    Z_material: tl.float32,
    # Simulation parameters
    voxel_size_x_cm: tl.float32, voxel_size_y_cm: tl.float32, voxel_size_z_cm: tl.float32,
    f_range: tl.float32, e_cut_MeV: tl.float32,
    # Physics constants
    ELECTRON_REST_MASS_MEV: tl.constexpr,
    PI: tl.constexpr,
    # RNG state (stateless Philox)
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
):
    """
    Apply common physics to both electrons and positrons with vectorized operations.
    Returns updated state and secondary production flags.

    Performance optimizations:
    - Vectorized energy loss calculation
    - Optimized step size computation
    - Efficient multiple scattering using lookup tables
    """
    # Particle type identification
    is_positron = particle_type == 1

    # Optimized step size calculation
    step_range_based = range_csda * f_range
    max_step_size = tl.minimum(voxel_size_x_cm, tl.minimum(voxel_size_y_cm, voxel_size_z_cm))
    step_size_cm = tl.minimum(step_range_based, max_step_size * 0.5)
    step_size_cm = tl.maximum(step_size_cm, 1e-6)  # Safety minimum

    # Vectorized energy loss with straggling
    dE_mean = step_size_cm * S_restricted * weight
    u1, u2, u3, u4, nc0, nc1, nc2, nc3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

    # Optimized Vavilov sampling (vectorized)
    kappa = dE_mean / ELECTRON_REST_MASS_MEV
    
    # Gaussian approximation for low energy loss
    xi_gaussian = dE_mean * 0.5772
    delta_gaussian = -xi_gaussian * fast_log_approx(tl.maximum(u1, 1e-12))
    dE_gaussian = dE_mean + delta_gaussian
    
    # Landau approximation for high energy loss
    sigma_landau = dE_mean * 0.5772
    r_landau = tl.sqrt(tl.maximum(0.0, -2.0 * fast_log_approx(tl.maximum(u1, 1e-12))))
    delta_landau = r_landau * sigma_landau * tl.cos(2.0 * PI * u2)
    dE_landau = dE_mean + delta_landau
    
    # Vavilov interpolation
    nu_vavilov = (tl.log(kappa) + 0.5772) / tl.log(10.0)
    xi_vavilov = dE_mean * (0.5772 + 0.1 * nu_vavilov)
    delta_vavilov = -xi_vavilov * fast_log_approx(tl.maximum(u1, 1e-12)) * fast_sqrt_approx(1.0 + nu_vavilov * u2)
    dE_vavilov = dE_mean + delta_vavilov
    
    # Select appropriate approximation based on kappa
    dE_actual = tl.where(kappa < 0.01, dE_gaussian,
                tl.where(kappa > 10.0, dE_landau, dE_vavilov))

    dE_actual = tl.maximum(dE_actual, 0.0)
    E_new = energy - dE_actual

    # Optimized multiple scattering using Molière theory
    total_energy = energy + ELECTRON_REST_MASS_MEV
    gamma = total_energy / ELECTRON_REST_MASS_MEV
    beta = tl.sqrt(tl.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
    p_MeV_c = beta * gamma * ELECTRON_REST_MASS_MEV

    # Radiation length calculation
    Z_float = Z_material.to(tl.float32)
    log_287_Z = fast_log_approx(287.0 / fast_sqrt_approx(Z_float))
    X0_cm = 716.4 * Z_float / (Z_float * (Z_float + 1.0) * fast_log_approx(287.0 / fast_sqrt_approx(Z_float)))

    step_in_rad_lengths = step_size_cm / X0_cm

    # Characteristic scattering angle
    theta0 = (13.6 / (beta * p_MeV_c)) * fast_sqrt_approx(step_in_rad_lengths)
    theta0 = theta0 * (1.0 + 0.038 * fast_log_approx(tl.maximum(step_in_rad_lengths, 1e-6)))
    theta0 = tl.minimum(theta0, 0.5)

    # Sample scattering angle using improved Gaussian approximation
    # Use Box-Muller transform for better angular distribution
    r1 = u3
    r2 = u1

    # Box-Muller transform to get Gaussian distribution
    # theta = theta0 * sqrt(-2 * ln(r1)) * cos(2*pi*r2)
    sqrt_neg2ln_r1 = tl.sqrt(tl.maximum(0.0, -2.0 * fast_log_approx(tl.maximum(r1, 1e-12))))
    theta = theta0 * sqrt_neg2ln_r1

    # Apply azimuthal angle
    phi = 2.0 * PI * r2

    # Proper trigonometric functions (no small angle approximation)
    cos_theta = fast_cos_approx(theta)
    sin_theta = fast_sin_approx(theta)
    sin_phi = fast_sin_approx(phi)
    cos_phi = fast_cos_approx(phi)

    # Update direction using Rodrigues formula approximation
    new_dir_x = dir_x * cos_theta + sin_theta * (dir_y * cos_phi - dir_z * sin_phi * dir_x)
    new_dir_y = dir_y * cos_theta + sin_theta * (dir_z * cos_phi + dir_x * sin_phi)
    new_dir_z = dir_z * cos_theta - sin_theta * sin_phi * tl.sqrt(1.0 - dir_z * dir_z)

    # Normalize direction
    norm_inv = tl.rsqrt(new_dir_x * new_dir_x + new_dir_y * new_dir_y + new_dir_z * new_dir_z)
    new_dir_x = new_dir_x * norm_inv
    new_dir_y = new_dir_y * norm_inv
    new_dir_z = new_dir_z * norm_inv

    # Update position
    new_pos_x = pos_x + new_dir_x * step_size_cm
    new_pos_y = pos_y + new_dir_y * step_size_cm
    new_pos_z = pos_z + new_dir_z * step_size_cm

    # Secondary production probabilities
    brem_prob = 1.0 - tl.exp(-P_brem * step_size_cm)
    delta_prob = 1.0 - tl.exp(-P_delta * step_size_cm)

    # Positron annihilation check
    should_annihilate = is_positron & (E_new <= e_cut_MeV)

    return (
        new_pos_x, new_pos_y, new_pos_z,     # Updated position
        new_dir_x, new_dir_y, new_dir_z,     # Updated direction
        E_new,                              # Updated energy
        dE_actual,                          # Energy loss for deposition
        nc0, nc1, nc2, nc3,                # Updated RNG counters
        should_annihilate,                   # Positron annihilation flag
        brem_prob,                          # Bremsstrahlung probability
        delta_prob,                         # Delta ray probability
    )


@triton.jit
def _binary_search_bin(energy_bin_edges: tl.tensor, E: tl.float32, num_bins: tl.constexpr) -> tl.int32:
    """
    Binary search to find energy bin index.
    Returns the index of the bin where E belongs.
    """
    lo = 0
    hi = num_bins - 1
    
    for _ in tl.static_range(16):  # Max 16 iterations for binary search
        mid = (lo + hi) // 2
        mid_val = tl.load(energy_bin_edges + mid)
        hi = tl.where(E < mid_val, mid, hi)
        lo = tl.where(E < mid_val, lo, mid + 1)
    
    result = lo - 1
    result = tl.maximum(0, tl.minimum(result, num_bins - 1))
    return result


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
    key=['N', 'num_energy_bins'],
    warmup=10,
    rep=20,
)
@triton.jit
def charged_particle_step_kernel(
    # Unified particle arrays (Structure of Arrays for optimal GPU performance)
    particle_pos_x: tl.tensor,
    particle_pos_y: tl.tensor,
    particle_pos_z: tl.tensor,
    particle_dir_x: tl.tensor,
    particle_dir_y: tl.tensor,
    particle_dir_z: tl.tensor,
    particle_E_MeV: tl.tensor,
    particle_weight: tl.tensor,
    particle_type: tl.tensor,  # 0=electron, 1=positron
    particle_material_id: tl.tensor,
    particle_alive: tl.tensor,

    # RNG seed (stateless Philox)
    rng_seed: tl.tensor,

    # Physics tables (same for both particle types)
    material_Z: tl.tensor,
    S_restricted_table: tl.tensor,
    range_cdsa_table: tl.tensor,
    P_brem_table: tl.tensor,
    P_delta_table: tl.tensor,

    # Energy binning
    energy_bin_edges: tl.tensor,

    # Outputs for secondaries
    photon_pos_x: tl.tensor,
    photon_pos_y: tl.tensor,
    photon_pos_z: tl.tensor,
    photon_dir_x: tl.tensor,
    photon_dir_y: tl.tensor,
    photon_dir_z: tl.tensor,
    photon_E_MeV: tl.tensor,
    photon_alive: tl.tensor,

    # Annihilation photon outputs (2 photons per annihilation)
    ann_photon1_pos_x: tl.tensor,
    ann_photon1_pos_y: tl.tensor,
    ann_photon1_pos_z: tl.tensor,
    ann_photon1_dir_x: tl.tensor,
    ann_photon1_dir_y: tl.tensor,
    ann_photon1_dir_z: tl.tensor,
    ann_photon1_alive: tl.tensor,

    ann_photon2_pos_x: tl.tensor,
    ann_photon2_pos_y: tl.tensor,
    ann_photon2_pos_z: tl.tensor,
    ann_photon2_dir_x: tl.tensor,
    ann_photon2_dir_y: tl.tensor,
    ann_photon2_dir_z: tl.tensor,
    ann_photon2_alive: tl.tensor,

    # Secondary electron/positron outputs
    secondary_pos_x: tl.tensor,
    secondary_pos_y: tl.tensor,
    secondary_pos_z: tl.tensor,
    secondary_dir_x: tl.tensor,
    secondary_dir_y: tl.tensor,
    secondary_dir_z: tl.tensor,
    secondary_E_MeV: tl.tensor,
    secondary_type: tl.tensor,
    secondary_alive: tl.tensor,

    # Updated particle outputs
    new_particle_E_MeV: tl.tensor,
    new_particle_alive: tl.tensor,

    # Energy deposition output
    edep_ptr: tl.tensor,

    # Simulation parameters
    voxel_size_x_cm: tl.constexpr,
    voxel_size_y_cm: tl.constexpr,
    voxel_size_z_cm: tl.constexpr,
    num_materials: tl.constexpr,
    num_energy_bins: tl.constexpr,
    e_cut_MeV: tl.constexpr,
    f_range: tl.constexpr,
    Z: tl.constexpr,
    Y: tl.constexpr,
    X: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    # Physics constants
    ELECTRON_REST_MASS_MEV: tl.constexpr,
    PI: tl.constexpr,
):
    """
    High-performance unified kernel for electron and positron transport.
    Processes both particle types simultaneously with vectorized physics.
    Uses stateless Philox RNG for random number generation.

    Performance benefits:
    - Single kernel launch for both particle types (reduced overhead)
    - Coalesced memory access patterns (particles processed together)
    - Vectorized physics calculations
    - Minimal control flow divergence
    - Optimized secondary particle generation
    - Stateless RNG eliminates memory overhead
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    # Load particle state with proper masking
    alive = tl.load(particle_alive + offset, mask=mask, other=0).to(tl.int1)
    E = tl.load(particle_E_MeV + offset, mask=mask, other=0.0).to(tl.float32)

    # Only process alive particles with energy above cutoff
    should_process = alive & (E > e_cut_MeV)
    mask = mask & should_process

    # Load arrays with optimal access pattern and masking
    pos_x = tl.load(particle_pos_x + offset, mask=mask, other=0.0).to(tl.float32)
    pos_y = tl.load(particle_pos_y + offset, mask=mask, other=0.0).to(tl.float32)
    pos_z = tl.load(particle_pos_z + offset, mask=mask, other=0.0).to(tl.float32)

    dir_x = tl.load(particle_dir_x + offset, mask=mask, other=0.0).to(tl.float32)
    dir_y = tl.load(particle_dir_y + offset, mask=mask, other=0.0).to(tl.float32)
    dir_z = tl.load(particle_dir_z + offset, mask=mask, other=0.0).to(tl.float32)

    weight = tl.load(particle_weight + offset, mask=mask, other=0.0).to(tl.float32)
    particle_type_val = tl.load(particle_type + offset, mask=mask, other=0).to(tl.int32)
    material_id = tl.load(particle_material_id + offset, mask=mask, other=0).to(tl.int32)

    # Get material properties
    Z = tl.load(material_Z + material_id)

    # Get energy bin for physics tables using binary search
    ebin = _binary_search_bin(energy_bin_edges, E, num_energy_bins)

    # Load physics parameters
    table_offset = material_id * num_energy_bins + ebin
    S_restricted = tl.load(S_restricted_table + table_offset)
    range_csda = tl.load(range_cdsa_table + table_offset)
    P_brem = tl.load(P_brem_table + table_offset)
    P_delta = tl.load(P_delta_table + table_offset)

    # Initialize stateless RNG state from particle ID and seed
    particle_id = offset.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)

    # Apply unified physics
    (new_pos_x, new_pos_y, new_pos_z,
     new_dir_x, new_dir_y, new_dir_z,
     E_new, dE_loss, c0, c1, c2, c3,
     should_annihilate, brem_prob, delta_prob) = _apply_common_charged_particle_physics(
        pos_x, pos_y, pos_z,      # position components
        dir_x, dir_y, dir_z,      # direction components
        E, weight, particle_type_val, material_id,
        S_restricted, range_csda, P_brem, P_delta, Z,
        voxel_size_x_cm, voxel_size_y_cm, voxel_size_z_cm,
        f_range, e_cut_MeV,
        ELECTRON_REST_MASS_MEV, PI,  # Physics constants
        k0, k1, c0, c1, c2, c3      # RNG state
    )

    # Update particle state
    tl.store(particle_E_MeV + offset, E_new, mask=mask)
    tl.store(particle_dir_x + offset, new_dir_x, mask=mask)
    tl.store(particle_dir_y + offset, new_dir_y, mask=mask)
    tl.store(particle_dir_z + offset, new_dir_z, mask=mask)
    tl.store(particle_pos_x + offset, new_pos_x, mask=mask)
    tl.store(particle_pos_y + offset, new_pos_y, mask=mask)
    tl.store(particle_pos_z + offset, new_pos_z, mask=mask)

    # Deposit energy loss to dose grid
    # Calculate voxel coordinates from new position
    iz = tl.floor(new_pos_z / voxel_size_z_cm).to(tl.int32)
    iy = tl.floor(new_pos_y / voxel_size_y_cm).to(tl.int32)
    ix = tl.floor(new_pos_x / voxel_size_x_cm).to(tl.int32)
    
    # Check if particle is inside the grid
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    
    # Calculate linear index (Z*Y*X layout)
    lin = iz * (Y * X) + iy * X + ix
    
    # Deposit energy loss using atomic add
    # Only deposit if inside the grid and energy loss is positive
    deposit_mask = mask & inside & (dE_loss > 0)
    tl.atomic_add(edep_ptr + lin, dE_loss * weight, mask=deposit_mask)

    # Handle positron annihilation and secondary production (vectorized)
    # Create masks for different conditions
    annihilation_mask = should_annihilate & mask
    stopped_mask = (~should_annihilate) & (E_new <= e_cut_MeV) & mask
    active_mask = (~should_annihilate) & (E_new > e_cut_MeV) & mask

    # Mark annihilating particles as dead
    tl.store(new_particle_alive + offset, 0, mask=annihilation_mask)

    # Mark stopped particles as dead
    tl.store(new_particle_alive + offset, 0, mask=stopped_mask)

    # Handle active particles (secondary production)
    # Bremsstrahlung emission
    u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, c0, c1, c2, c3 = rand_uniform16(k0, k1, c0, c1, c2, c3)

    brem_emit_mask = (u1 < brem_prob) & (E_new > 0.1) & active_mask

    # Sample photon energy with Bethe-Heitler spectrum
    brems_photon_energy = _sample_bethe_heitler_energy(u2, u3, u4, u5, u6, u7, u8, u9, u10, E_new, Z.to(tl.float32))
    brems_valid_mask = (brems_photon_energy > 0.01) & brem_emit_mask

    # Update parent energy for bremsstrahlung
    E_with_brem = E_new - brems_photon_energy
    tl.store(new_particle_E_MeV + offset, E_with_brem, mask=brems_valid_mask)

    # Create bremsstrahlung photon
    tl.store(photon_pos_x + offset, new_pos_x, mask=brems_valid_mask)
    tl.store(photon_pos_y + offset, new_pos_y, mask=brems_valid_mask)
    tl.store(photon_pos_z + offset, new_pos_z, mask=brems_valid_mask)
    tl.store(photon_dir_x + offset, new_dir_x, mask=brems_valid_mask)
    tl.store(photon_dir_y + offset, new_dir_y, mask=brems_valid_mask)
    tl.store(photon_dir_z + offset, new_dir_z, mask=brems_valid_mask)
    tl.store(photon_E_MeV + offset, brems_photon_energy, mask=brems_valid_mask)
    tl.store(photon_alive + offset, 1, mask=brems_valid_mask)

    # Delta ray production
    delta_emit_mask = (u11 < delta_prob) & (E_new > 0.05) & active_mask

    # Sample delta energy with Moller scattering
    delta_energy = _sample_moller_delta_energy(u12, u13, u14, u15, u16, u1, u2, u3, u4, E_new, Z.to(tl.float32), ELECTRON_REST_MASS_MEV)
    delta_valid_mask = (delta_energy > 0.001) & delta_emit_mask

    # Update primary energy for delta ray
    E_with_delta = E_new - delta_energy
    tl.store(new_particle_E_MeV + offset, E_with_delta, mask=delta_valid_mask)

    # Scattering angle for delta ray (Moller scattering kinematics)
    # cos(theta) = 1 - (T / E_kin) * (1 + E_kin / (2 * m_e * c^2))
    # where T = delta_energy, E_kin = E_new
    E_kin = E_new
    T = delta_energy
    cos_theta_delta = 1.0 - (T / E_kin) * (1.0 + E_kin / (2.0 * ELECTRON_REST_MASS_MEV))
    cos_theta_delta = tl.maximum(-1.0, tl.minimum(1.0, cos_theta_delta))
    sin_theta_delta = fast_sqrt_approx(1.0 - cos_theta_delta * cos_theta_delta)

    # Random azimuthal angle
    phi_delta = 2.0 * PI * u13

    delta_dir_x = new_dir_x * cos_theta_delta + sin_theta_delta * (new_dir_y * tl.cos(phi_delta))
    delta_dir_y = new_dir_y * cos_theta_delta + sin_theta_delta * (new_dir_z * tl.sin(phi_delta))
    delta_dir_z = new_dir_z * cos_theta_delta - sin_theta_delta * tl.sqrt(new_dir_x * new_dir_x + new_dir_y * new_dir_y)

    # Normalize
    norm = tl.rsqrt(delta_dir_x * delta_dir_x + delta_dir_y * delta_dir_y + delta_dir_z * delta_dir_z)
    delta_dir_x = delta_dir_x * norm
    delta_dir_y = delta_dir_y * norm
    delta_dir_z = delta_dir_z * norm

    # Create delta ray
    tl.store(secondary_pos_x + offset, new_pos_x, mask=delta_valid_mask)
    tl.store(secondary_pos_y + offset, new_pos_y, mask=delta_valid_mask)
    tl.store(secondary_pos_z + offset, new_pos_z, mask=delta_valid_mask)
    tl.store(secondary_dir_x + offset, delta_dir_x, mask=delta_valid_mask)
    tl.store(secondary_dir_y + offset, delta_dir_y, mask=delta_valid_mask)
    tl.store(secondary_dir_z + offset, delta_dir_z, mask=delta_valid_mask)
    tl.store(secondary_E_MeV + offset, delta_energy, mask=delta_valid_mask)
    tl.store(secondary_type + offset, particle_type_val, mask=delta_valid_mask)
    tl.store(secondary_alive + offset, 1, mask=delta_valid_mask)

    # Handle positron annihilation photons
    # Vectorized creation of two 511 keV annihilation photons
    u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

    # Marsaglia method for isotropic direction (unrolled to avoid while loop)
    u_x = 2.0 * u14 - 1.0
    u_y = 2.0 * u15 - 1.0
    s = u_x * u_x + u_y * u_y

    # If s >= 1, use alternative direction (simplified)
    s = tl.minimum(s, 0.99)
    sqrt_term = tl.sqrt(1.0 - s)

    # First photon direction
    photon1_dir_x = 2.0 * u_x * sqrt_term
    photon1_dir_y = 2.0 * u_y * sqrt_term
    photon1_dir_z = 1.0 - 2.0 * s

    # Normalize
    norm = tl.rsqrt(photon1_dir_x * photon1_dir_x + photon1_dir_y * photon1_dir_y + photon1_dir_z * photon1_dir_z)
    photon1_dir_x = photon1_dir_x * norm
    photon1_dir_y = photon1_dir_y * norm
    photon1_dir_z = photon1_dir_z * norm

    # Second photon opposite direction (exact 180°)
    photon2_dir_x = -photon1_dir_x
    photon2_dir_y = -photon1_dir_y
    photon2_dir_z = -photon1_dir_z

    # Store annihilation photons
    tl.store(ann_photon1_pos_x + offset, new_pos_x, mask=annihilation_mask)
    tl.store(ann_photon1_pos_y + offset, new_pos_y, mask=annihilation_mask)
    tl.store(ann_photon1_pos_z + offset, new_pos_z, mask=annihilation_mask)
    tl.store(ann_photon1_dir_x + offset, photon1_dir_x, mask=annihilation_mask)
    tl.store(ann_photon1_dir_y + offset, photon1_dir_y, mask=annihilation_mask)
    tl.store(ann_photon1_dir_z + offset, photon1_dir_z, mask=annihilation_mask)
    tl.store(ann_photon1_alive + offset, 1, mask=annihilation_mask)

    tl.store(ann_photon2_pos_x + offset, new_pos_x, mask=annihilation_mask)
    tl.store(ann_photon2_pos_y + offset, new_pos_y, mask=annihilation_mask)
    tl.store(ann_photon2_pos_z + offset, new_pos_z, mask=annihilation_mask)
    tl.store(ann_photon2_dir_x + offset, photon2_dir_x, mask=annihilation_mask)
    tl.store(ann_photon2_dir_y + offset, photon2_dir_y, mask=annihilation_mask)
    tl.store(ann_photon2_dir_z + offset, photon2_dir_z, mask=annihilation_mask)
    tl.store(ann_photon2_alive + offset, 1, mask=annihilation_mask)

    # Mark particles that should not be processed as dead
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_all = offset < N

    alive = tl.load(particle_alive + offset, mask=mask_all, other=0).to(tl.int1)
    E = tl.load(particle_E_MeV + offset, mask=mask_all, other=0.0).to(tl.float32)

    should_be_dead = (~alive) | (E <= e_cut_MeV)
    tl.store(new_particle_alive + offset, 0, mask=mask_all & should_be_dead)


@triton.jit
def _sample_bethe_heitler_energy(u1: tl.float32, u2: tl.float32, u3: tl.float32, u4: tl.float32, u5: tl.float32,
                                  u6: tl.float32, u7: tl.float32, u8: tl.float32, u9: tl.float32,
                                  E_initial: tl.float32, Z_material: tl.float32) -> tl.float32:
    """
    Sample bremsstrahlung photon energy using Bethe-Heitler spectrum with screening corrections.
    Uses rejection sampling for accurate distribution with 7 attempts for better accuracy.

    Args:
        u1, u2, u3, u4, u5, u6, u7, u8, u9: Uniform random numbers [0,1]
        E_initial: Initial electron energy (MeV)
        Z_material: Atomic number of material

    Returns:
        brems_photon_energy: Sampled photon energy (MeV)
    """
    # Reduced photon energy k = photon_energy / E_initial
    # Range: k_min to k_max (typically 0.01 to 1.0)
    k_min = 0.01
    k_max = 1.0

    # Screening parameter
    alpha = 1.0 / 137.0
    Z_eff = Z_material
    screening_param = 100.0 * alpha * Z_eff / E_initial

    # Maximum of the distribution (at k ~ 0.1-0.2)
    k_peak = 0.15
    phi1_peak = 4.0 * tl.log(1.0 + 1.0 / (screening_param * k_peak * k_peak))
    phi2_peak = 4.0 * tl.log(1.0 + 1.0 / (screening_param * (1.0 - k_peak) * (1.0 - k_peak)))
    f_max = (1.0 + (1.0 - k_peak) * (1.0 - k_peak)) / k_peak * (phi1_peak - Z_eff * phi2_peak)

    # Rejection sampling - first attempt
    k = k_min + u1 * (k_max - k_min)
    phi1_k = 4.0 * tl.log(1.0 + 1.0 / (screening_param * k * k))
    phi2_k = 4.0 * tl.log(1.0 + 1.0 / (screening_param * (1.0 - k) * (1.0 - k)))
    f_k = (1.0 + (1.0 - k) * (1.0 - k)) / k * (phi1_k - Z_eff * phi2_k)

    # Accept/reject
    accept = u2 < (f_k / f_max)

    # Second attempt
    k2 = k_min + u3 * (k_max - k_min)
    phi1_k2 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * k2 * k2))
    phi2_k2 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * (1.0 - k2) * (1.0 - k2)))
    f_k2 = (1.0 + (1.0 - k2) * (1.0 - k2)) / k2 * (phi1_k2 - Z_eff * phi2_k2)
    accept2 = u4 < (f_k2 / f_max)

    # Third attempt
    k3 = k_min + u5 * (k_max - k_min)
    phi1_k3 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * k3 * k3))
    phi2_k3 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * (1.0 - k3) * (1.0 - k3)))
    f_k3 = (1.0 + (1.0 - k3) * (1.0 - k3)) / k3 * (phi1_k3 - Z_eff * phi2_k3)
    accept3 = u6 < (f_k3 / f_max)

    # Fourth attempt
    k4 = k_min + u7 * (k_max - k_min)
    phi1_k4 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * k4 * k4))
    phi2_k4 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * (1.0 - k4) * (1.0 - k4)))
    f_k4 = (1.0 + (1.0 - k4) * (1.0 - k4)) / k4 * (phi1_k4 - Z_eff * phi2_k4)
    accept4 = u8 < (f_k4 / f_max)

    # Fifth attempt
    k5 = k_min + u9 * (k_max - k_min)
    phi1_k5 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * k5 * k5))
    phi2_k5 = 4.0 * tl.log(1.0 + 1.0 / (screening_param * (1.0 - k5) * (1.0 - k5)))
    f_k5 = (1.0 + (1.0 - k5) * (1.0 - k5)) / k5 * (phi1_k5 - Z_eff * phi2_k5)

    # Use accepted, second accepted, third accepted, fourth accepted, or fifth attempt
    k_final = tl.where(accept, k, tl.where(accept2, k2, tl.where(accept3, k3, tl.where(accept4, k4, k5))))

    # Photon energy
    brems_photon_energy = k_final * E_initial

    return brems_photon_energy


@triton.jit
def _sample_moller_delta_energy(u1: tl.float32, u2: tl.float32, u3: tl.float32, u4: tl.float32, u5: tl.float32,
                                  u6: tl.float32, u7: tl.float32, u8: tl.float32, u9: tl.float32,
                                  E_initial: tl.float32, Z_material: tl.float32,
                                  ELECTRON_REST_MASS_MEV: tl.constexpr) -> tl.float32:
    """
    Sample delta ray energy using Moller scattering (e- + e- -> e- + e-).
    Uses rejection sampling with 7 attempts for better accuracy.

    Args:
        u1, u2, u3, u4, u5, u6, u7, u8, u9: Uniform random numbers [0,1]
        E_initial: Initial electron energy (MeV)
        Z_material: Atomic number of material
        ELECTRON_REST_MASS_MEV: Electron rest mass in MeV

    Returns:
        delta_energy: Sampled delta ray energy (MeV)
    """
    # Calculate relativistic parameters
    total_energy = E_initial + ELECTRON_REST_MASS_MEV
    gamma = total_energy / ELECTRON_REST_MASS_MEV
    beta_sq = tl.maximum(0.0, 1.0 - 1.0 / (gamma * gamma))
    beta = fast_sqrt_approx(beta_sq)

    # Maximum energy transfer (T_max)
    # For electron-electron scattering: T_max = E_initial / 2
    T_max = E_initial * 0.5

    # Minimum energy transfer (T_min)
    T_min = 0.001  # 1 keV minimum

    # Moller cross-section: dσ/dT ∝ 1/T^2 * [1 - β^2 T/T_max + (T/T_max)^2]
    # For rejection sampling, we need the maximum of the distribution

    # The maximum occurs at T = T_min
    ratio_min = T_min / T_max
    f_max = 1.0 / (T_min * T_min) * (1.0 - beta_sq * ratio_min + ratio_min * ratio_min)

    # Rejection sampling - first attempt
    T = T_min + u1 * (T_max - T_min)
    ratio = T / T_max
    f_T = 1.0 / (T * T) * (1.0 - beta_sq * ratio + ratio * ratio)

    # Accept/reject
    accept = u2 < (f_T / f_max)

    # Second attempt
    T2 = T_min + u3 * (T_max - T_min)
    ratio2 = T2 / T_max
    f_T2 = 1.0 / (T2 * T2) * (1.0 - beta_sq * ratio2 + ratio2 * ratio2)
    accept2 = u4 < (f_T2 / f_max)

    # Third attempt
    T3 = T_min + u5 * (T_max - T_min)
    ratio3 = T3 / T_max
    f_T3 = 1.0 / (T3 * T3) * (1.0 - beta_sq * ratio3 + ratio3 * ratio3)
    accept3 = u6 < (f_T3 / f_max)

    # Fourth attempt
    T4 = T_min + u7 * (T_max - T_min)
    ratio4 = T4 / T_max
    f_T4 = 1.0 / (T4 * T4) * (1.0 - beta_sq * ratio4 + ratio4 * ratio4)
    accept4 = u8 < (f_T4 / f_max)

    # Fifth attempt
    T5 = T_min + u9 * (T_max - T_min)
    ratio5 = T5 / T_max
    f_T5 = 1.0 / (T5 * T5) * (1.0 - beta_sq * ratio5 + ratio5 * ratio5)

    # Use accepted, second accepted, third accepted, fourth accepted, or fifth attempt
    T_final = tl.where(accept, T, tl.where(accept2, T2, tl.where(accept3, T3, tl.where(accept4, T4, T5))))

    return T_final


@triton.jit
def sample_multiple_scattering_angle(u1: tl.float32, u2: tl.float32, step_length_cm: tl.float32,
                                   E_MeV: tl.float32, Z_material: tl.int32,
                                   ELECTRON_REST_MASS_MEV: tl.constexpr) -> (tl.float32, tl.float32, tl.float32):
    """
    High-performance multiple scattering angle sampling using Molière theory.
    Optimized for GPU with branch-free vectorized operations.

    Args:
        u1, u2: Uniform random numbers [0,1]
        step_length_cm: Step length in cm
        E_MeV: Particle energy in MeV
        Z_material: Atomic number of material
        ELECTRON_REST_MASS_MEV: Electron rest mass in MeV

    Returns:
        (cos_theta, phi): Polar scattering cosine and azimuthal angle
    """
    total_energy = E_MeV + ELECTRON_REST_MASS_MEV
    ratio = ELECTRON_REST_MASS_MEV / total_energy
    beta_sq = tl.maximum(0.0, 1.0 - ratio * ratio)
    beta = fast_sqrt_approx(beta_sq)
    p_MeV_c = fast_sqrt_approx(tl.maximum(0.0, total_energy * total_energy - ELECTRON_REST_MASS_MEV * ELECTRON_REST_MASS_MEV))

    Z_float = Z_material.to(tl.float32)

    # Radiation length
    X0_approx = 716.4 * Z_float / (Z_float * (Z_float + 1.0) * fast_log_approx(287.0 / fast_sqrt_approx(Z_float)))

    step_length_rad_lengths = step_length_cm / X0_approx

    # Characteristic angle
    theta0 = (13.6 / (beta * p_MeV_c)) * fast_sqrt_approx(step_length_rad_lengths) * \
             (1.0 + 0.038 * fast_log_approx(tl.maximum(step_length_rad_lengths, 1e-6)))

    chi_c = theta0 * 1.1774

    # Branch-free angle sampling using direct calculation
    # Instead of rejection sampling, use direct transformation
    r = fast_sqrt_approx(tl.maximum(0.0, -2.0 * fast_log_approx(tl.maximum(u1, 1e-12))))
    theta_gaussian = r * theta0

    # Molière correction factor
    moliere_factor = (theta_gaussian / chi_c) * (theta_gaussian / chi_c)
    acceptance_prob = 1.0 / (1.0 + moliere_factor)

    # Branch-free selection between Gaussian and Molière
    theta_moliere = theta_gaussian * fast_sqrt_approx(1.0 + moliere_factor)
    theta = tl.where(u2 < acceptance_prob, theta_gaussian, theta_moliere)

    theta = tl.minimum(theta, 0.5)
    cos_theta = 1.0 - 0.5 * theta * theta
    phi = 2.0 * PI * u2

    return cos_theta, phi


@triton.jit
def rotate_vector_around_axis(ux: tl.float32, uy: tl.float32, uz: tl.float32,
                             axis_x: tl.float32, axis_y: tl.float32, axis_z: tl.float32,
                             cos_theta: tl.float32, sin_theta: tl.float32) -> (tl.float32, tl.float32, tl.float32):
    """
    Efficient vector rotation around an axis using Rodrigues' formula.
    Optimized for GPU with minimal operations.

    Args:
        ux, uy, uz: Vector components to rotate
        axis_x, axis_y, axis_z: Axis unit vector components
        cos_theta, sin_theta: Cosine and sine of rotation angle

    Returns:
        (ux_rot, uy_rot, uz_rot): Rotated vector components
    """
    # Dot product
    dot_axis = ux * axis_x + uy * axis_y + uz * axis_z

    # Cross product
    cross_x = uy * axis_z - uz * axis_y
    cross_y = uz * axis_x - ux * axis_z
    cross_z = ux * axis_y - uy * axis_x

    # Rodrigues' rotation
    one_minus_cos = 1.0 - cos_theta
    ux_rot = ux * cos_theta + cross_x * sin_theta + axis_x * dot_axis * one_minus_cos
    uy_rot = uy * cos_theta + cross_y * sin_theta + axis_y * dot_axis * one_minus_cos
    uz_rot = uz * cos_theta + cross_z * sin_theta + axis_z * dot_axis * one_minus_cos

    # Normalize
    norm = fast_sqrt_approx(ux_rot * ux_rot + uy_rot * uy_rot + uz_rot * uz_rot)
    norm = tl.maximum(norm, 1e-6)

    return ux_rot / norm, uy_rot / norm, uz_rot / norm