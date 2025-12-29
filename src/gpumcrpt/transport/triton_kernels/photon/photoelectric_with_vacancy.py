"""
Photoelectric absorption with atomic vacancy Triton kernels.
"""

from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.rng.philox import init_philox_state, rand_uniform4


@triton.jit
def photon_photoelectric_with_vacancy_kernel(
    # Photon state (SoA format)
    photon_pos: tl.tensor,
    photon_dir: tl.tensor,
    photon_E: tl.tensor,
    photon_w: tl.tensor,
    photon_ebin: tl.tensor,
    # Physics tables
    material_id_flat: tl.tensor,
    shell_cdf: tl.tensor,
    E_bind: tl.tensor,
    # Output: electron state
    e_pos: tl.tensor,
    e_dir: tl.tensor,
    e_E: tl.tensor,
    e_w: tl.tensor,
    # Output: vacancy information (for atomic relaxation)
    v_pos: tl.tensor,
    v_mat: tl.tensor,
    v_shell: tl.tensor,
    v_w: tl.tensor,
    v_has: tl.tensor,
    # Output: local energy deposit
    edep: tl.tensor,
    # RNG seed
    rng_seed: tl.tensor,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    # Number of shells
    S: tl.constexpr,
):
    """
    Photoelectric absorption kernel with vacancy creation.
    Creates photoelectrons and atomic vacancies for relaxation.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load photon state
    E = tl.load(photon_E + offs)
    w = tl.load(photon_w + offs)

    # Create mask for valid photons (E > 0 and w > 0)
    valid_mask = (E > 0) & (w > 0)

    pos_z = tl.load(photon_pos + offs * 3 + 0, mask=valid_mask, other=0.0)
    pos_y = tl.load(photon_pos + offs * 3 + 1, mask=valid_mask, other=0.0)
    pos_x = tl.load(photon_pos + offs * 3 + 2, mask=valid_mask, other=0.0)
    
    dir_z = tl.load(photon_dir + offs * 3 + 0, mask=valid_mask, other=0.0)
    dir_y = tl.load(photon_dir + offs * 3 + 1, mask=valid_mask, other=0.0)
    dir_x = tl.load(photon_dir + offs * 3 + 2, mask=valid_mask, other=0.0)
    ebin = tl.load(photon_ebin + offs)

    # Load material ID for each photon
    mat_id = tl.load(material_id_flat + offs, mask=valid_mask, other=0)

    # Initialize stateless Philox RNG
    particle_id = offs.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)

    # Sample uniform random numbers
    u1, u2, u3, _, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

    # Sample binding energy from shell CDF
    # shell_cdf is indexed as [mat_id, shell_idx] with linear index: mat_id * S + shell_idx
    # We'll load the shell CDF for each photon's material
    # Create a 2D index array where each row is a photon's shell indices
    shell_idx = tl.arange(0, S)
    
    # Expand shell_idx to 2D: (BLOCK_SIZE, S) where each row is the same shell indices
    # Then add mat_id * S to get the linear indices
    shell_idx_2d = shell_idx[None, :]  # Shape: (1, S)
    shell_idx_2d = shell_idx_2d + tl.zeros((BLOCK_SIZE, 1), dtype=tl.int32)  # Shape: (BLOCK_SIZE, S)
    
    # Compute linear indices: mat_id * S + shell_idx
    mat_id_expanded = mat_id[:, None]  # Shape: (BLOCK_SIZE, 1)
    linear_indices = mat_id_expanded * S + shell_idx_2d  # Shape: (BLOCK_SIZE, S)
    
    # Load shell CDF values
    shell_probs = tl.load(shell_cdf + linear_indices, mask=valid_mask[:, None], other=0.0)

    # Sample shell using inverse CDF
    # Get the last shell CDF value (total probability) for each material
    # We need to compute the offset for the last shell: mat_id * S + (S - 1)
    last_shell_offset = mat_id * S + (S - 1)
    last_shell_cdf = tl.load(shell_cdf + last_shell_offset, mask=valid_mask, other=0.0)
    u_shell = u1 * last_shell_cdf
    
    # Create mask for comparison (all shells except the last one)
    shell_idx_all = tl.arange(0, S)
    shell_mask = shell_idx_all < (S - 1)
    
    # Compare u_shell with all shell CDF values except the last
    # We need to broadcast u_shell to (BLOCK_SIZE, S) for comparison
    u_shell_expanded = u_shell[:, None]
    comparison = u_shell_expanded >= shell_probs
    
    # Apply mask to exclude the last column
    comparison = tl.where(shell_mask[None, :], comparison, False)
    
    # Count how many CDF values are less than u_shell
    selected_shell = tl.sum(comparison, axis=1)
    binding_energy = tl.load(E_bind + mat_id * S + selected_shell, mask=valid_mask, other=0.0)

    # Create mask for photons with sufficient energy to overcome binding energy
    energy_mask = valid_mask & (E > binding_energy)

    # Initialize outputs with default values for all valid photons
    tl.store(e_pos + offs * 3 + 0, tl.zeros_like(pos_z), mask=valid_mask)
    tl.store(e_pos + offs * 3 + 1, tl.zeros_like(pos_y), mask=valid_mask)
    tl.store(e_pos + offs * 3 + 2, tl.zeros_like(pos_x), mask=valid_mask)
    tl.store(e_dir + offs * 3 + 0, tl.zeros_like(dir_z), mask=valid_mask)
    tl.store(e_dir + offs * 3 + 1, tl.zeros_like(dir_y), mask=valid_mask)
    tl.store(e_dir + offs * 3 + 2, tl.zeros_like(dir_x), mask=valid_mask)
    tl.store(e_E + offs, tl.zeros_like(E), mask=valid_mask)
    tl.store(e_w + offs, tl.zeros_like(w), mask=valid_mask)
    tl.store(v_has + offs, tl.zeros(offs.shape, dtype=tl.int32), mask=valid_mask)

    # Process photons with sufficient energy
    photoelectron_energy = tl.where(energy_mask, E - binding_energy, 0.0)
    tl.store(e_E + offs, photoelectron_energy, mask=energy_mask)
    tl.store(e_w + offs, w, mask=energy_mask)

    # Store vacancy information for photons with sufficient energy
    tl.store(v_pos + offs * 3 + 0, pos_z, mask=energy_mask)
    tl.store(v_pos + offs * 3 + 1, pos_y, mask=energy_mask)
    tl.store(v_pos + offs * 3 + 2, pos_x, mask=energy_mask)
    tl.store(v_mat + offs, mat_id, mask=energy_mask)
    tl.store(v_shell + offs, selected_shell, mask=energy_mask)
    tl.store(v_w + offs, w, mask=energy_mask)
    tl.store(v_has + offs, 1, mask=energy_mask)
