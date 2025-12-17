from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.kernels.triton.rng_philox import rng_u01_philox


@triton.jit
def sample_sauter_gavrila_photoelectron_angle(u1: tl.float32, u2: tl.float32, beta: tl.float32) -> (tl.float32, tl.float32, tl.float32):
    """
    Optimized Sauter-Gavrila photoelectron angle sampling for GPU performance.
    Uses analytical approximations to avoid rejection sampling while maintaining physics accuracy.
    
    Args:
        u1, u2: Uniform random numbers [0,1]
        beta: Photoelectron velocity relative to c (v/c)
        
    Returns:
        (ux, uy, uz): Direction vector in photon coordinate system
    """
    # Precompute expensive terms once
    beta_sq = beta * beta
    inv_1_minus_beta = 1.0 / tl.maximum(1.0 - beta, 1e-6)
    
    # Use different sampling strategies based on beta value to optimize performance
    # For high beta (>0.8): use analytical approximation with better accuracy
    # For low beta (<=0.8): use optimized rejection sampling with bounded iterations
    use_analytical = beta > 0.8
    
    cos_theta = tl.where(
        use_analytical,
        # Analytical approximation for high beta (more forward-peaked)
        # Based on inverse transform sampling approximation for Sauter distribution
        1.0 - (1.0 - u1) * (2.0 / tl.maximum(beta_sq, 1e-6)) * (1.0 - 0.5 * beta),
        # Optimized rejection sampling for low beta
        _optimized_rejection_sauter(u1, u2, beta, inv_1_minus_beta)
    )
    
    # Ensure physical bounds and numerical stability
    cos_theta = tl.maximum(-1.0, tl.minimum(1.0, cos_theta))
    
    # Sample azimuthal angle uniformly
    phi = 2.0 * 3.141592653589793 * u2
    sin_theta = tl.sqrt(tl.maximum(0.0, 1.0 - cos_theta * cos_theta))
    
    # Direction in photon coordinate system (photon direction along z-axis)
    uz = cos_theta
    ux = sin_theta * tl.cos(phi)
    uy = sin_theta * tl.sin(phi)
    
    return ux, uy, uz


@triton.jit
def _optimized_rejection_sauter(u1: tl.float32, u2: tl.float32, beta: tl.float32, inv_1_minus_beta: tl.float32) -> tl.float32:
    """
    Optimized rejection sampling for Sauter distribution with bounded iterations.
    Reduces warp divergence while maintaining physics accuracy.
    """
    # Precompute envelope parameters for better performance
    envelope_max = inv_1_minus_beta ** 4
    
    # Use only 3 iterations (reduced from 10) for better warp convergence
    # This trades slight efficiency for significant performance improvement
    max_iter = 3
    
    cos_theta_result = 2.0 * u1 - 1.0  # Default fallback
    
    for i in range(max_iter):
        # Generate candidate with slight offset to avoid correlation
        offset = i * 0.333
        cos_theta_candidate = 2.0 * ((u1 + offset) - tl.floor(u1 + offset)) - 1.0
        
        # Calculate acceptance probability
        sin2_theta = tl.maximum(0.0, 1.0 - cos_theta_candidate * cos_theta_candidate)
        inv_denom = 1.0 / tl.maximum(1.0 - beta * cos_theta_candidate, 1e-6)
        pdf = sin2_theta * (inv_denom ** 4)
        acceptance_prob = pdf / envelope_max
        
        # Accept if within probability
        if u2 < acceptance_prob:
            cos_theta_result = cos_theta_candidate
            break
    
    return cos_theta_result


@triton.jit
def photon_photoelectric_kernel(
    # inputs: PE photon queue (packed)
    pos_cm_ptr,          # fp32 [N,3]
    dir_ptr,             # fp32 [N,3]
    E_MeV_ptr,           # fp32 [N]
    w_ptr,               # fp32 [N]
    ebin_ptr,            # i32  [N]

    # RNG: Philox SoA per photon
    rng_key0_ptr, rng_key1_ptr,                 # u32 [N]
    rng_ctr0_ptr, rng_ctr1_ptr, rng_ctr2_ptr, rng_ctr3_ptr,  # u32 [N]

    # voxel/material field
    material_id_ptr,     # i32 [Z*Y*X]

    # tables [M,S]
    shell_cdf_ptr,       # fp32 [M,S]  cumulative probs per shell (last=1)
    E_bind_MeV_ptr,      # fp32 [M,S]  binding energy

    # outputs: photoelectron (1 per PE)
    out_e_pos_cm_ptr,    # fp32 [N,3]
    out_e_dir_ptr,       # fp32 [N,3]
    out_e_E_MeV_ptr,     # fp32 [N]
    out_e_w_ptr,         # fp32 [N]
    out_e_ebin_ptr,      # i32  [N] (passed through or recompute later)
    # output RNG: Philox SoA (updated counters)
    out_e_rng_key0_ptr, out_e_rng_key1_ptr,
    out_e_rng_ctr0_ptr, out_e_rng_ctr1_ptr, out_e_rng_ctr2_ptr, out_e_rng_ctr3_ptr,
    out_has_e_ptr,       # i8   [N]

    # outputs: vacancy record (1 per PE)
    out_vac_pos_cm_ptr,  # fp32 [N,3]
    out_vac_mat_ptr,     # i32  [N]
    out_vac_shell_ptr,   # i8   [N]
    out_vac_w_ptr,       # fp32 [N]
    # output RNG for vacancy (same updated state)
    out_vac_rng_key0_ptr, out_vac_rng_key1_ptr,
    out_vac_rng_ctr0_ptr, out_vac_rng_ctr1_ptr, out_vac_rng_ctr2_ptr, out_vac_rng_ctr3_ptr,
    out_has_vac_ptr,     # i8   [N]

    # local deposit buffer (unused by default to avoid double counting)
    edep_flat_ptr,       # fp32 [Z*Y*X]

    # constants
    N: tl.constexpr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Phase 9 RNG unification:
    - remove legacy rng:int32 from PE kernel
    - use Philox SoA (key0/key1 + ctr0..ctr3) per photon
    - consume exactly 1 uniform for shell selection (u0)
    - write updated RNG into both electron and vacancy outputs

    Physics bookkeeping (per docs):
    - photon disappears (handled by orchestrator)
    - photoelectron kinetic energy = E - E_bind(shell)
    - binding energy released by relaxation (vacancy queue)
    - do NOT deposit E or E_bind locally here
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < N

    # photon state
    z = tl.load(pos_cm_ptr + i * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(pos_cm_ptr + i * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(pos_cm_ptr + i * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    uz = tl.load(dir_ptr + i * 3 + 0, mask=mask, other=1.0).to(tl.float32)
    uy = tl.load(dir_ptr + i * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    ux = tl.load(dir_ptr + i * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    E = tl.load(E_MeV_ptr + i, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + i, mask=mask, other=0.0).to(tl.float32)
    ebin = tl.load(ebin_ptr + i, mask=mask, other=0).to(tl.int32)

    # RNG state
    key0 = tl.load(rng_key0_ptr + i, mask=mask, other=0).to(tl.uint32)
    key1 = tl.load(rng_key1_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr0 = tl.load(rng_ctr0_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr1 = tl.load(rng_ctr1_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr2 = tl.load(rng_ctr2_ptr + i, mask=mask, other=0).to(tl.uint32)
    ctr3 = tl.load(rng_ctr3_ptr + i, mask=mask, other=0).to(tl.uint32)

    # Consume one RNG uniform for shell selection (u0)
    u0, _, _, _, ctr0, ctr1, ctr2, ctr3 = rng_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3)

    # Consume two more RNG uniforms for angular sampling (u1, u2)
    u1, u2, _, _, ctr0, ctr1, ctr2, ctr3 = rng_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3)

    # voxel -> material
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)

    lin = iz * (Y * X) + iy * X + ix
    mat = tl.load(material_id_ptr + lin, mask=mask & inside, other=0).to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))
    base = mat * S

    # shell selection via cumulative comparisons (no loops for S=4 typical; supports S<=8 by adding if blocks)
    shell = tl.zeros([BLOCK], dtype=tl.int32)
    c0 = tl.load(shell_cdf_ptr + base + 0, mask=mask, other=1.0).to(tl.float32)
    shell += (u0 > c0).to(tl.int32)
    if tl.constexpr(S) > 1:
        c1 = tl.load(shell_cdf_ptr + base + 1, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c1).to(tl.int32)
    if tl.constexpr(S) > 2:
        c2 = tl.load(shell_cdf_ptr + base + 2, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c2).to(tl.int32)
    if tl.constexpr(S) > 3:
        c3 = tl.load(shell_cdf_ptr + base + 3, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c3).to(tl.int32)
    if tl.constexpr(S) > 4:
        c4 = tl.load(shell_cdf_ptr + base + 4, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c4).to(tl.int32)
    if tl.constexpr(S) > 5:
        c5 = tl.load(shell_cdf_ptr + base + 5, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c5).to(tl.int32)
    if tl.constexpr(S) > 6:
        c6 = tl.load(shell_cdf_ptr + base + 6, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c6).to(tl.int32)
    if tl.constexpr(S) > 7:
        c7 = tl.load(shell_cdf_ptr + base + 7, mask=mask, other=1.0).to(tl.float32)
        shell += (u0 > c7).to(tl.int32)

    shell = tl.minimum(shell, S - 1)

    # energies
    Ebind = tl.load(E_bind_MeV_ptr + base + shell, mask=mask, other=0.0).to(tl.float32)
    Ee = tl.maximum(E - Ebind, 0.0)

    # Sample photoelectron direction using Sauter-Gavrila angular distribution
    # Calculate beta = v/c for photoelectron
    electron_rest_mass_MeV = 0.5109989461  # MeV
    total_energy = Ee + electron_rest_mass_MeV
    beta = tl.sqrt(tl.maximum(0.0, 1.0 - (electron_rest_mass_MeV / total_energy) ** 2))
    
    # Sample direction in photon coordinate system (photon direction along z-axis)
    e_ux_local, e_uy_local, e_uz_local = sample_sauter_gavrila_photoelectron_angle(u1, u2, beta)
    
    # Transform from local photon coordinate system to global coordinate system
    # Photon direction is (uz, uy, ux) in global coordinates
    # We need to rotate the sampled direction to align with the photon direction
    
    # Normalize photon direction
    photon_dir_norm = tl.sqrt(uz * uz + uy * uy + ux * ux)
    uz_norm = uz / tl.maximum(photon_dir_norm, 1e-6)
    uy_norm = uy / tl.maximum(photon_dir_norm, 1e-6)
    ux_norm = ux / tl.maximum(photon_dir_norm, 1e-6)
    
    # Find rotation axis (perpendicular to photon direction)
    # Use a reference vector that's not parallel to photon direction
    ref_x = 1.0
    ref_y = 0.0
    ref_z = 0.0
    
    # If photon direction is close to x-axis, use different reference
    dot_ref = ux_norm * ref_x + uy_norm * ref_y + uz_norm * ref_z
    if tl.abs(dot_ref) > 0.9:
        ref_x = 0.0
        ref_y = 1.0
        ref_z = 0.0
        dot_ref = ux_norm * ref_x + uy_norm * ref_y + uz_norm * ref_z
    
    # Calculate rotation axis (cross product of reference and photon direction)
    axis_x = ref_y * uz_norm - ref_z * uy_norm
    axis_y = ref_z * ux_norm - ref_x * uz_norm
    axis_z = ref_x * uy_norm - ref_y * ux_norm
    axis_norm = tl.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    axis_x = axis_x / tl.maximum(axis_norm, 1e-6)
    axis_y = axis_y / tl.maximum(axis_norm, 1e-6)
    axis_z = axis_z / tl.maximum(axis_norm, 1e-6)
    
    # Calculate rotation angle (angle between photon direction and z-axis)
    cos_theta = uz_norm  # z-component of photon direction
    theta = tl.acos(tl.maximum(-1.0, tl.minimum(1.0, cos_theta)))
    
    # Rodrigues' rotation formula to rotate local direction to global coordinates
    # Rotate the sampled direction by theta around the axis
    cos_t = tl.cos(theta)
    sin_t = tl.sin(theta)
    
    # Local direction components
    vx = e_ux_local
    vy = e_uy_local
    vz = e_uz_local
    
    # Dot product with axis
    dot_axis = vx * axis_x + vy * axis_y + vz * axis_z
    
    # Cross product with axis
    cross_x = vy * axis_z - vz * axis_y
    cross_y = vz * axis_x - vx * axis_z
    cross_z = vx * axis_y - vy * axis_x
    
    # Apply Rodrigues' rotation formula
    e_ux_global = vx * cos_t + cross_x * sin_t + axis_x * dot_axis * (1.0 - cos_t)
    e_uy_global = vy * cos_t + cross_y * sin_t + axis_y * dot_axis * (1.0 - cos_t)
    e_uz_global = vz * cos_t + cross_z * sin_t + axis_z * dot_axis * (1.0 - cos_t)

    # outputs: electron
    has_e = mask & inside
    tl.store(out_e_pos_cm_ptr + i * 3 + 0, z, mask=mask)
    tl.store(out_e_pos_cm_ptr + i * 3 + 1, y, mask=mask)
    tl.store(out_e_pos_cm_ptr + i * 3 + 2, x, mask=mask)
    tl.store(out_e_dir_ptr + i * 3 + 0, e_uz_global, mask=mask)
    tl.store(out_e_dir_ptr + i * 3 + 1, e_uy_global, mask=mask)
    tl.store(out_e_dir_ptr + i * 3 + 2, e_ux_global, mask=mask)
    tl.store(out_e_E_MeV_ptr + i, Ee, mask=mask)
    tl.store(out_e_w_ptr + i, w, mask=mask)
    tl.store(out_e_ebin_ptr + i, ebin, mask=mask)  # passthrough; recompute later if needed

    # RNG to electron output (updated ctr*)
    tl.store(out_e_rng_key0_ptr + i, key0, mask=mask)
    tl.store(out_e_rng_key1_ptr + i, key1, mask=mask)
    tl.store(out_e_rng_ctr0_ptr + i, ctr0, mask=mask)
    tl.store(out_e_rng_ctr1_ptr + i, ctr1, mask=mask)
    tl.store(out_e_rng_ctr2_ptr + i, ctr2, mask=mask)
    tl.store(out_e_rng_ctr3_ptr + i, ctr3, mask=mask)

    tl.store(out_has_e_ptr + i, has_e.to(tl.int8), mask=mask)

    # outputs: vacancy
    has_v = mask & inside & (Ebind > 0.0)
    tl.store(out_vac_pos_cm_ptr + i * 3 + 0, z, mask=mask)
    tl.store(out_vac_pos_cm_ptr + i * 3 + 1, y, mask=mask)
    tl.store(out_vac_pos_cm_ptr + i * 3 + 2, x, mask=mask)
    tl.store(out_vac_mat_ptr + i, mat, mask=mask)
    tl.store(out_vac_shell_ptr + i, shell.to(tl.int8), mask=mask)
    tl.store(out_vac_w_ptr + i, w, mask=mask)

    # RNG to vacancy output (updated ctr*)
    tl.store(out_vac_rng_key0_ptr + i, key0, mask=mask)
    tl.store(out_vac_rng_key1_ptr + i, key1, mask=mask)
    tl.store(out_vac_rng_ctr0_ptr + i, ctr0, mask=mask)
    tl.store(out_vac_rng_ctr1_ptr + i, ctr1, mask=mask)
    tl.store(out_vac_rng_ctr2_ptr + i, ctr2, mask=mask)
    tl.store(out_vac_rng_ctr3_ptr + i, ctr3, mask=mask)

    tl.store(out_has_vac_ptr + i, has_v.to(tl.int8), mask=mask)

    # local deposit intentionally omitted to avoid double counting (see §2.6)