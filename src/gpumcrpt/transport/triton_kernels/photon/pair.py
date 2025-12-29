"""
Pair production Triton kernels.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ..rng.philox import init_philox_state, rand_uniform4


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
def photon_pair_kernel(
    # Photon state (SoA format)
    photon_pos: tl.tensor,
    photon_dir: tl.tensor,
    photon_E: tl.tensor,
    photon_w: tl.tensor,
    photon_ebin: tl.tensor,
    # Output: electron state
    e_pos: tl.tensor,
    e_dir: tl.tensor,
    e_E: tl.tensor,
    e_w: tl.tensor,
    # Output: positron state
    p_pos: tl.tensor,
    p_dir: tl.tensor,
    p_E: tl.tensor,
    p_w: tl.tensor,
    # RNG seed
    rng_seed: tl.tensor,
    # Number of elements
    N: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Pair production kernel for photons (>1.022 MeV).
    Creates electron-positron pairs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Boundary check: only process valid indices
    valid_idx_mask = offs < N

    # Load photon state with boundary check
    pos = tl.load(photon_pos + offs, mask=valid_idx_mask[:, None], other=0.0)
    dir = tl.load(photon_dir + offs, mask=valid_idx_mask[:, None], other=0.0)
    E = tl.load(photon_E + offs, mask=valid_idx_mask, other=0.0)
    w = tl.load(photon_w + offs, mask=valid_idx_mask, other=0.0)
    ebin = tl.load(photon_ebin + offs, mask=valid_idx_mask, other=0)
    
    # Mask for valid photons (E > 0 and w > 0)
    valid_mask = valid_idx_mask & (E > 0.0) & (w > 0.0)
    
    # Mask for photons above pair production threshold
    pair_threshold_mask = E > 1.022

    # Stateless RNG initialization
    particle_id = offs.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)
    u1, u2, u3, _, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

    # Simplified pair production - replace with proper physics
    pair_prob = 0.1  # Simplified probability
    pair_mask = valid_mask & pair_threshold_mask & (u1 < pair_prob)
    
    # Expand pair_mask to 2D for broadcasting with position/direction tensors
    pair_mask_2d = tl.expand_dims(pair_mask, 1)
    
    # Expand valid_idx_mask to 2D for position/direction tensors
    valid_idx_mask_2d = tl.expand_dims(valid_idx_mask, 1)
    
    # Energy available for kinetic energy after pair production
    kinetic_energy = tl.where(pair_mask, E - 1.022, 0.0)

    # Simplified energy split
    electron_energy = kinetic_energy * 0.5
    positron_energy = kinetic_energy * 0.5

    # Create electron with boundary check
    tl.store(e_pos + offs, tl.where(pair_mask_2d, pos, 0.0), mask=valid_idx_mask_2d)
    tl.store(e_dir + offs, tl.where(pair_mask_2d, dir, 0.0), mask=valid_idx_mask_2d)
    tl.store(e_E + offs, electron_energy, mask=valid_idx_mask)
    tl.store(e_w + offs, tl.where(pair_mask, w, 0.0), mask=valid_idx_mask)

    # Create positron with boundary check
    tl.store(p_pos + offs, tl.where(pair_mask_2d, pos, 0.0), mask=valid_idx_mask_2d)
    tl.store(p_dir + offs, tl.where(pair_mask_2d, dir, 0.0), mask=valid_idx_mask_2d)
    tl.store(p_E + offs, positron_energy, mask=valid_idx_mask)
    tl.store(p_w + offs, tl.where(pair_mask, w, 0.0), mask=valid_idx_mask)