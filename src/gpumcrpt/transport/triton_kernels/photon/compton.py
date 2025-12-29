"""
Compton scattering Triton kernels.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ..rng.philox import init_philox_state, rand_uniform4
from ..utils.gpu_math import fast_sqrt_approx, fast_sin_cos_approx


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
    key=['K'],
    warmup=10,
    rep=20,
)
@triton.jit
def photon_compton_kernel(
    # Photon state (SoA format)
    photon_pos: tl.tensor,
    photon_dir: tl.tensor,
    photon_E: tl.tensor,
    photon_w: tl.tensor,
    photon_ebin: tl.tensor,
    # Physics tables
    compton_inv_cdf: tl.tensor,
    K: tl.constexpr,
    # Output: scattered photon state
    scat_pos: tl.tensor,
    scat_dir: tl.tensor,
    scat_E: tl.tensor,
    scat_w: tl.tensor,
    scat_ebin: tl.tensor,
    # Output: recoil electron state
    e_pos: tl.tensor,
    e_dir: tl.tensor,
    e_E: tl.tensor,
    e_w: tl.tensor,
    # RNG seed
    rng_seed: tl.tensor,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Compton scattering kernel for photons with branch-free optimizations.
    Creates Compton recoil electrons using vectorized operations.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load photon state
    pos = tl.load(photon_pos + offs, mask=mask, other=0.0)
    dir = tl.load(photon_dir + offs, mask=mask, other=0.0)
    E = tl.load(photon_E + offs, mask=mask, other=0.0)
    w = tl.load(photon_w + offs, mask=mask, other=0.0)
    ebin = tl.load(photon_ebin + offs, mask=mask, other=0)
    
    # Branch-free energy cutoff check
    above_cutoff = E > 0.01
    
    # Stateless RNG initialization
    particle_id = offs.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)
    u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

    # Branch-free Compton scattering probability
    compton_prob = 0.3
    should_scatter = (u1 < compton_prob) & above_cutoff
    
    # Vectorized energy transfer calculation
    # Simplified Klein-Nishina sampling
    alpha = E / 0.511
    epsilon_min = 1.0 / (1.0 + 2.0 * alpha)
    epsilon = epsilon_min + (1.0 - epsilon_min) * u2
    recoil_energy = E * (1.0 - epsilon)
    
    # Branch-free scattered photon energy
    scat_E_val = tl.where(should_scatter, E * epsilon, E)
    
    # Branch-free recoil electron energy
    e_E_val = tl.where(should_scatter, recoil_energy, 0.0)
    
    # Vectorized direction sampling for scattered photon
    # Simplified isotropic scattering
    phi = 2.0 * 3.14159 * u3
    cos_theta = 1.0 - 2.0 * u4
    sin_theta = fast_sqrt_approx(1.0 - cos_theta * cos_theta)
    
    # Rotate direction vector using fast sin/cos
    cos_phi, sin_phi = fast_sin_cos_approx(phi)
    
    scat_dir_x = dir * cos_theta + fast_sqrt_approx(1.0 - dir * dir) * sin_theta * cos_phi
    scat_dir_y = fast_sqrt_approx(1.0 - dir * dir) * sin_theta * sin_phi
    scat_dir_z = dir * cos_theta
    
    # Normalize scattered direction
    norm = fast_sqrt_approx(scat_dir_x * scat_dir_x + scat_dir_y * scat_dir_y + scat_dir_z * scat_dir_z)
    norm = tl.maximum(norm, 1e-6)
    scat_dir_x = scat_dir_x / norm
    scat_dir_y = scat_dir_y / norm
    scat_dir_z = scat_dir_z / norm
    
    # Branch-free direction selection
    final_dir_x = tl.where(should_scatter, scat_dir_x, dir)
    final_dir_y = tl.where(should_scatter, scat_dir_y, 0.0)
    final_dir_z = tl.where(should_scatter, scat_dir_z, 0.0)
    
    # Branch-free electron direction (opposite to scattered photon)
    e_dir_x = tl.where(should_scatter, -final_dir_x, 0.0)
    e_dir_y = tl.where(should_scatter, -final_dir_y, 0.0)
    e_dir_z = tl.where(should_scatter, -final_dir_z, 0.0)
    
    # Store scattered photon state
    tl.store(scat_pos + offs, pos, mask=mask)
    tl.store(scat_dir + offs, final_dir_x, mask=mask)
    tl.store(scat_E + offs, scat_E_val, mask=mask)
    tl.store(scat_w + offs, w, mask=mask)
    tl.store(scat_ebin + offs, ebin, mask=mask)

    # Store recoil electron state
    tl.store(e_pos + offs, pos, mask=mask)
    tl.store(e_dir + offs, e_dir_x, mask=mask)
    tl.store(e_E + offs, e_E_val, mask=mask)
    tl.store(e_w + offs, w, mask=mask)