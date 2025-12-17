from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01


@triton.jit
def sample_multiple_scattering_angle(u1: tl.float32, u2: tl.float32, step_length_cm: tl.float32, 
                                   E_MeV: tl.float32, Z_material: tl.int32) -> (tl.float32, tl.float32, tl.float32):
    """
    Optimized multiple scattering angle sampling for GPU performance.
    Uses analytical approximations and precomputed constants for efficiency.
    
    Args:
        u1, u2: Uniform random numbers [0,1]
        step_length_cm: Step length in cm
        E_MeV: Electron energy in MeV
        Z_material: Atomic number of material
        
    Returns:
        (cos_theta, phi): Polar scattering cosine and azimuthal angle
    """
    # Precompute constants for better performance
    electron_rest_mass_MeV = 0.5109989461
    hbar_c_MeV_cm = 1.973269804e-11
    a0_cm = 5.29177210903e-9
    
    # Calculate relativistic parameters
    total_energy = E_MeV + electron_rest_mass_MeV
    beta_sq = tl.maximum(0.0, 1.0 - (electron_rest_mass_MeV / total_energy) ** 2)
    beta = tl.sqrt(beta_sq)
    p_MeV_c = tl.sqrt(tl.maximum(0.0, total_energy ** 2 - electron_rest_mass_MeV ** 2))
    
    # Use optimized approximation for characteristic angle
    # Simplified formula based on Highland's extension
    Z_float = tl.float32(Z_material)
    
    # Highland's formula approximation for RMS scattering angle
    # θ_0 ≈ (13.6 MeV / (β p c)) * sqrt(step_length/X0) * (1 + 0.038 * ln(step_length/X0))
    # Simplified for performance while maintaining accuracy
    X0_approx = 716.4 * tl.float32(Z_material) / (Z_float * (Z_float + 1.0) * tl.log(287.0 / tl.sqrt(Z_float)))
    
    # Calculate characteristic angle using optimized formula
    step_length_rad_lengths = step_length_cm / X0_approx
    theta0 = (13.6 / (beta * p_MeV_c)) * tl.sqrt(step_length_rad_lengths) * \
             (1.0 + 0.038 * tl.log(tl.maximum(step_length_rad_lengths, 1e-6)))
    
    # Use more efficient sampling method
    # For small angles, use Gaussian approximation with optimized Box-Muller
    # Precompute log terms to reduce operations
    log_u1 = tl.log(tl.maximum(u1, 1e-12))
    r = tl.sqrt(tl.maximum(0.0, -2.0 * log_u1))
    
    # Sample theta with proper variance scaling
    theta = r * theta0
    
    # Apply physical limits and numerical stability
    theta = tl.minimum(theta, 0.5)  # Limit to 0.5 radian for better approximation
    
    # Use small-angle approximation for cosine: cos(theta) ≈ 1 - theta^2/2
    # This avoids expensive cos() call while maintaining accuracy for small angles
    cos_theta = 1.0 - 0.5 * theta * theta
    
    # Sample azimuthal angle uniformly
    phi = 6.283185307179586 * u2  # 2 * π
    
    return cos_theta, phi


@triton.jit
def rotate_vector_around_axis(ux: tl.float32, uy: tl.float32, uz: tl.float32,
                             axis_x: tl.float32, axis_y: tl.float32, axis_z: tl.float32,
                             cos_theta: tl.float32, sin_theta: tl.float32) -> (tl.float32, tl.float32, tl.float32):
    """
    Rotate a vector around an axis using Rodrigues' rotation formula.
    """
    # Dot product with axis
    dot_axis = ux * axis_x + uy * axis_y + uz * axis_z
    
    # Cross product with axis
    cross_x = uy * axis_z - uz * axis_y
    cross_y = uz * axis_x - ux * axis_z
    cross_z = ux * axis_y - uy * axis_x
    
    # Apply Rodrigues' rotation formula
    ux_rot = ux * cos_theta + cross_x * sin_theta + axis_x * dot_axis * (1.0 - cos_theta)
    uy_rot = uy * cos_theta + cross_y * sin_theta + axis_y * dot_axis * (1.0 - cos_theta)
    uz_rot = uz * cos_theta + cross_z * sin_theta + axis_z * dot_axis * (1.0 - cos_theta)
    
    # Normalize the result
    norm = tl.sqrt(ux_rot * ux_rot + uy_rot * uy_rot + uz_rot * uz_rot)
    ux_rot = ux_rot / tl.maximum(norm, 1e-6)
    uy_rot = uy_rot / tl.maximum(norm, 1e-6)
    uz_rot = uz_rot / tl.maximum(norm, 1e-6)
    
    return ux_rot, uy_rot, uz_rot


# Autotuning configurations for RTX A4000 (Ampere architecture)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'NUM_WARPS': 2}, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['N'],  # Tune based on problem size
)
@triton.jit
def electron_condensed_step_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    material_id_ptr, rho_ptr, ref_rho_ptr,
    S_restricted_ptr, range_csda_ptr,
    P_brem_per_cm_ptr, P_delta_per_cm_ptr,
    Z_material_ptr,  # Atomic numbers for materials
    edep_ptr,  # flattened [Z*Y*X]
    out_pos_ptr, out_dir_ptr, out_E_ptr, out_w_ptr, out_rng_ptr, out_ebin_ptr,
    out_alive_ptr, out_emit_brem_ptr, out_emit_delta_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr, ECOUNT: tl.constexpr,
    N: tl.constexpr,  # number of particles
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,  # Autotuned warp count
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    f_vox: tl.constexpr, f_range: tl.constexpr, max_dE_frac: tl.constexpr,
):
    """
    Optimized Condensed-history electron step using Triton 3.5.1 features:
    - Block pointers for efficient memory access
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    - Implicit boundary checking
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Create block pointers for efficient memory access
    pos_block = tl.make_block_ptr(
        base=pos_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    dir_block = tl.make_block_ptr(
        base=dir_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    
    # Load particle data using block pointers with cache hints
    pos_data = tl.load(pos_block, boundary_check=(0, 1), mask=mask, other=0.0, cache_modifier=".cg")
    dir_data = tl.load(dir_block, boundary_check=(0, 1), mask=mask, other=0.0, cache_modifier=".cg")
    
    # Extract position and direction components
    z = pos_data[:, 0]
    y = pos_data[:, 1]
    x = pos_data[:, 2]
    uz = dir_data[:, 0]
    uy = dir_data[:, 1]
    ux = dir_data[:, 2]

    # Load other particle properties with cache hints
    E = tl.load(E_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    w = tl.load(w_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    rng = tl.load(rng_ptr + offs, mask=mask, other=123456789, cache_modifier=".cg")
    ebin = tl.load(ebin_ptr + offs, mask=mask, other=0, cache_modifier=".cg").to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

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
    S = tl.load(S_restricted_ptr + off, mask=inside, other=0.0) * rho_scale  # MeV/cm
    R = tl.load(range_csda_ptr + off, mask=inside, other=1e-3)  # cm at ref; (simple scaling handled by rho_scale outside in host if needed)

    vox_mean = (voxel_z_cm + voxel_y_cm + voxel_x_cm) * (1.0 / 3.0)
    ds1 = f_vox * vox_mean
    ds2 = f_range * tl.maximum(R, 1e-6)
    ds = tl.minimum(ds1, ds2)
    ds = tl.maximum(ds, 1e-5)

    dE = S * ds
    dE = tl.minimum(dE, max_dE_frac * E)
    dE = tl.minimum(dE, E)

    # Score continuous deposition
    tl.atomic_add(edep_ptr + lin, dE * w, mask=inside)

    E2 = E - dE

    # Apply Multiple Scattering (MSC) approximation
    # Get atomic number for current material
    Z_mat = tl.load(Z_material_ptr + mat, mask=inside, other=8.0).to(tl.int32)  # Default to oxygen (Z=8)
    
    # Sample multiple scattering angles
    u_msc1, rng = rand_uniform_u01(rng)
    u_msc2, rng = rand_uniform_u01(rng)
    
    cos_theta, phi = sample_multiple_scattering_angle(u_msc1, u_msc2, ds, E, Z_mat)
    sin_theta = tl.sqrt(tl.maximum(0.0, 1.0 - cos_theta * cos_theta))
    
    # Apply scattering to direction vector
    # Find a perpendicular axis to rotate around
    # Use a reference vector that's not parallel to current direction
    ref_x = 1.0
    ref_y = 0.0
    ref_z = 0.0
    
    # If direction is close to x-axis, use different reference
    dot_ref = ux * ref_x + uy * ref_y + uz * ref_z
    if tl.abs(dot_ref) > 0.9:
        ref_x = 0.0
        ref_y = 1.0
        ref_z = 0.0
        dot_ref = ux * ref_x + uy * ref_y + uz * ref_z
    
    # Calculate rotation axis (cross product of reference and direction)
    axis_x = ref_y * uz - ref_z * uy
    axis_y = ref_z * ux - ref_x * uz
    axis_z = ref_x * uy - ref_y * ux
    axis_norm = tl.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    axis_x = axis_x / tl.maximum(axis_norm, 1e-6)
    axis_y = axis_y / tl.maximum(axis_norm, 1e-6)
    axis_z = axis_z / tl.maximum(axis_norm, 1e-6)
    
    # Rotate direction vector by the scattering angle
    uz_scattered, uy_scattered, ux_scattered = rotate_vector_around_axis(
        uz, uy, ux, axis_x, axis_y, axis_z, cos_theta, sin_theta
    )
    
    # Apply azimuthal rotation around the original direction
    # This adds the second scattering component
    cos_phi = tl.cos(phi)
    sin_phi = tl.sin(phi)
    
    # Find a vector perpendicular to the scattered direction for azimuthal rotation
    # Use the same axis calculation but with the scattered direction
    dot_ref_scat = ux_scattered * ref_x + uy_scattered * ref_y + uz_scattered * ref_z
    if tl.abs(dot_ref_scat) > 0.9:
        ref_x = 0.0
        ref_y = 1.0
        ref_z = 0.0
        dot_ref_scat = ux_scattered * ref_x + uy_scattered * ref_y + uz_scattered * ref_z
    
    axis_azim_x = ref_y * uz_scattered - ref_z * uy_scattered
    axis_azim_y = ref_z * ux_scattered - ref_x * uz_scattered
    axis_azim_z = ref_x * uy_scattered - ref_y * ux_scattered
    axis_azim_norm = tl.sqrt(axis_azim_x * axis_azim_x + axis_azim_y * axis_azim_y + axis_azim_z * axis_azim_z)
    axis_azim_x = axis_azim_x / tl.maximum(axis_azim_norm, 1e-6)
    axis_azim_y = axis_azim_y / tl.maximum(axis_azim_norm, 1e-6)
    axis_azim_z = axis_azim_z / tl.maximum(axis_azim_norm, 1e-6)
    
    # Apply azimuthal rotation
    uz_final, uy_final, ux_final = rotate_vector_around_axis(
        uz_scattered, uy_scattered, ux_scattered, 
        axis_azim_x, axis_azim_y, axis_azim_z, cos_phi, sin_phi
    )

    # Move with scattered direction
    z2 = z + ds * uz_final
    y2 = y + ds * uy_final
    x2 = x + ds * ux_final

    alive = inside & (E2 > 0.0)

    # Hard-event probabilities (if pointers are null-like, host should pass tensors of zeros)
    lam_b = tl.load(P_brem_per_cm_ptr + off, mask=inside, other=0.0) * rho_scale
    lam_d = tl.load(P_delta_per_cm_ptr + off, mask=inside, other=0.0) * rho_scale

    # P = 1-exp(-lam*ds)
    Pb = 1.0 - tl.exp(-lam_b * ds)
    Pd = 1.0 - tl.exp(-lam_d * ds)

    u1, rng = rand_uniform_u01(rng)
    u2, rng = rand_uniform_u01(rng)

    emit_brem = alive & (u1 < Pb)
    emit_delta = alive & (u2 < Pd)

    # Create output block pointers for efficient memory storage
    out_pos_block = tl.make_block_ptr(
        base=out_pos_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    out_dir_block = tl.make_block_ptr(
        base=out_dir_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    
    # Prepare output data for block storage
    out_pos_data = tl.zeros((BLOCK_SIZE, 3), dtype=tl.float32)
    out_dir_data = tl.zeros((BLOCK_SIZE, 3), dtype=tl.float32)
    
    out_pos_data[:, 0] = z2
    out_pos_data[:, 1] = y2
    out_pos_data[:, 2] = x2
    out_dir_data[:, 0] = uz_final
    out_dir_data[:, 1] = uy_final
    out_dir_data[:, 2] = ux_final
    
    # Write out using block pointers with cache hints
    tl.store(out_pos_block, out_pos_data, boundary_check=(0, 1), mask=mask, cache_modifier=".cg")
    tl.store(out_dir_block, out_dir_data, boundary_check=(0, 1), mask=mask, cache_modifier=".cg")
    
    tl.store(out_E_ptr + offs, E2, mask=mask, cache_modifier=".cg")
    tl.store(out_w_ptr + offs, w, mask=mask, cache_modifier=".cg")
    tl.store(out_rng_ptr + offs, rng, mask=mask, cache_modifier=".cg")
    tl.store(out_ebin_ptr + offs, ebin, mask=mask, cache_modifier=".cg")

    tl.store(out_alive_ptr + offs, alive.to(tl.int8), mask=mask, cache_modifier=".cg")
    tl.store(out_emit_brem_ptr + offs, emit_brem.to(tl.int8), mask=mask, cache_modifier=".cg")
    tl.store(out_emit_delta_ptr + offs, emit_delta.to(tl.int8), mask=mask, cache_modifier=".cg")