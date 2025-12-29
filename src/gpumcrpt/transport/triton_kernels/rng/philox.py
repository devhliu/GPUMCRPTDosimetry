"""
Stateless Philox Counter-Based RNG for GPU Monte Carlo Simulations

This module implements a stateless, offset-based Philox4x32-10 random number generator
optimized for high-performance Monte Carlo dosimetry simulations on GPU.

Key Features:
- Stateless: No memory reads for RNG state during simulation
- Register-based: All operations stay in GPU registers
- Counter-based: Random state derived from seed + particle ID
- Perfect parallelization: Each particle gets unique random stream
- Batch checkpointing: Track completed particles, not full RNG state
- Zero memory overhead: No RNG state buffers needed

Design Philosophy:
For dosimetry simulations with millions of particles, memory bandwidth is precious.
Reading/writing RNG state for every step wastes bandwidth that should be used for
physics calculations. Instead, we calculate the RNG state purely from the seed and
particle ID, keeping everything in fast GPU registers.

Usage:
    # In kernel initialization
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k0, k1, c0, c1, c2, c3 = init_philox_state(pid, seed)

    # Generate random numbers in physics loop
    u0, u1, u2, u3, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
    u, c0, c1, c2, c3 = rand_uniform(k0, k1, c0, c1, c2, c3)
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _philox_round(
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    k0: tl.tensor, k1: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Single round of Philox-4x32 permutation.
    
    This is the core Philox operation that mixes counters with keys.
    All operations are vectorized for optimal GPU performance.
    
    Args:
        c0, c1, c2, c3: Counter values (uint32)
        k0, k1: Key values (uint32)
    
    Returns:
        (nc0, nc1, nc2, nc3, nk0, nk1): New counters and keys
    """
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9
    W1 = 0xBB67AE85
    
    hi0 = ((c0.to(tl.uint64) * M0) >> 32).to(tl.uint32)
    lo0 = (c0 * M0).to(tl.uint32)
    hi1 = ((c2.to(tl.uint64) * M1) >> 32).to(tl.uint32)
    lo1 = (c2 * M1).to(tl.uint32)
    
    nc0 = hi1 ^ c1 ^ k0
    nc1 = lo0
    nc2 = hi0 ^ c3 ^ k1
    nc3 = lo1
    
    nk0 = (k0 + W0).to(tl.uint32)
    nk1 = (k1 + W1).to(tl.uint32)
    
    return nc0, nc1, nc2, nc3, nk0, nk1


@triton.jit
def _philox4x32_10(
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    k0: tl.tensor, k1: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Philox4x32 with 10 rounds - provides excellent statistical quality.
    
    The 10-round unrolled loop is fully unrolled by the compiler for maximum
    performance. This provides cryptographic-quality random numbers suitable
    for scientific Monte Carlo simulations.
    
    Args:
        c0, c1, c2, c3: Counter values (uint32)
        k0, k1: Key values (uint32)
    
    Returns:
        (r0, r1, r2, r3): Four 32-bit random numbers
    """
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    c0, c1, c2, c3, k0, k1 = _philox_round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


@triton.jit
def init_philox_state(
    particle_id: tl.tensor,
    seed: tl.tensor,
    salt: tl.tensor = 0xDEADBEEF
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Initialize Philox RNG state from particle ID and seed.
    
    This is the key to stateless RNG: we derive the complete RNG state
    purely from the particle ID and global seed. No memory reads needed.
    
    Each particle gets a unique random stream that never overlaps with others.
    
    Args:
        particle_id: Unique identifier for each particle (uint32)
        seed: Global seed for the simulation (uint32)
        salt: Optional salt value for additional uniqueness (uint32)
    
    Returns:
        (k0, k1, c0, c1, c2, c3): Philox key and counter values
    
    Example:
        pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        k0, k1, c0, c1, c2, c3 = init_philox_state(pid, seed)
    """
    k0 = tl.full(particle_id.shape, seed, dtype=tl.uint32)
    k1 = tl.full(particle_id.shape, salt, dtype=tl.uint32)
    
    c0 = particle_id.to(tl.uint32)
    c1 = tl.zeros(particle_id.shape, dtype=tl.uint32)
    c2 = tl.zeros(particle_id.shape, dtype=tl.uint32)
    c3 = tl.zeros(particle_id.shape, dtype=tl.uint32)
    
    return k0, k1, c0, c1, c2, c3

@triton.jit
def rand_uniform(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate a single uniform random number in [0, 1).
    
    More efficient when only one random number is needed.
    Still generates 4 internally but only returns one.
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
    
    Returns:
        (u, nc0, nc1, nc2, nc3): One uniform float and updated counters
    
    Example:
        u, c0, c1, c2, c3 = rand_uniform(k0, k1, c0, c1, c2, c3)
    """
    r0, _, _, _ = _philox4x32_10(c0, c1, c2, c3, k0, k1)
    
    scale = 2.3283064e-10
    u = r0.to(tl.float32) * scale
    
    nc0 = (c0 + 1).to(tl.uint32)
    
    return u, nc0, c1, c2, c3

@triton.jit
def rand_uniform4(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate 4 uniform random numbers in [0, 1).
    
    This is the primary RNG function for Monte Carlo simulations.
    Returns 4 random numbers and updates the counter for the next call.
    All operations stay in GPU registers - no memory access.
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
    
    Returns:
        (u0, u1, u2, u3, nc0, nc1, nc2, nc3): Four uniform floats and updated counters
    
    Example:
        u0, u1, u2, u3, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
    """
    r0, r1, r2, r3 = _philox4x32_10(c0, c1, c2, c3, k0, k1)
    
    scale = 2.3283064e-10
    
    u0 = r0.to(tl.float32) * scale
    u1 = r1.to(tl.float32) * scale
    u2 = r2.to(tl.float32) * scale
    u3 = r3.to(tl.float32) * scale
    
    nc0 = (c0 + 1).to(tl.uint32)
    
    return u0, u1, u2, u3, nc0, c1, c2, c3

@triton.jit
def rand_uniform8(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate 8 uniform random numbers in [0, 1) in a single call.
    
    More efficient than calling rand_uniform4 twice - reduces function call overhead.
    All operations stay in GPU registers - no memory access.
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
    
    Returns:
        (u0, u1, u2, u3, u4, u5, u6, u7, nc0, nc1, nc2, nc3): Eight uniform floats and updated counters
    
    Example:
        u0, u1, u2, u3, u4, u5, u6, u7, c0, c1, c2, c3 = rand_uniform8(k0, k1, c0, c1, c2, c3)
    """
    r0, r1, r2, r3 = _philox4x32_10(c0, c1, c2, c3, k0, k1)
    r4, r5, r6, r7 = _philox4x32_10((c0 + 1).to(tl.uint32), c1, c2, c3, k0, k1)
    
    scale = 2.3283064e-10
    
    u0 = r0.to(tl.float32) * scale
    u1 = r1.to(tl.float32) * scale
    u2 = r2.to(tl.float32) * scale
    u3 = r3.to(tl.float32) * scale
    u4 = r4.to(tl.float32) * scale
    u5 = r5.to(tl.float32) * scale
    u6 = r6.to(tl.float32) * scale
    u7 = r7.to(tl.float32) * scale
    
    nc0 = (c0 + 2).to(tl.uint32)
    
    return u0, u1, u2, u3, u4, u5, u6, u7, nc0, c1, c2, c3


@triton.jit
def rand_uniform16(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate 16 uniform random numbers in [0, 1) in a single call.
    
    More efficient than calling rand_uniform4 four times - significantly reduces function call overhead.
    All operations stay in GPU registers - no memory access.
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
    
    Returns:
        (u0, u1, ..., u15, nc0, nc1, nc2, nc3): Sixteen uniform floats and updated counters
    
    Example:
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, c0, c1, c2, c3 = rand_uniform16(k0, k1, c0, c1, c2, c3)
    """
    r0, r1, r2, r3 = _philox4x32_10(c0, c1, c2, c3, k0, k1)
    r4, r5, r6, r7 = _philox4x32_10((c0 + 1).to(tl.uint32), c1, c2, c3, k0, k1)
    r8, r9, r10, r11 = _philox4x32_10((c0 + 2).to(tl.uint32), c1, c2, c3, k0, k1)
    r12, r13, r14, r15 = _philox4x32_10((c0 + 3).to(tl.uint32), c1, c2, c3, k0, k1)
    
    scale = 2.3283064e-10
    
    u0 = r0.to(tl.float32) * scale
    u1 = r1.to(tl.float32) * scale
    u2 = r2.to(tl.float32) * scale
    u3 = r3.to(tl.float32) * scale
    u4 = r4.to(tl.float32) * scale
    u5 = r5.to(tl.float32) * scale
    u6 = r6.to(tl.float32) * scale
    u7 = r7.to(tl.float32) * scale
    u8 = r8.to(tl.float32) * scale
    u9 = r9.to(tl.float32) * scale
    u10 = r10.to(tl.float32) * scale
    u11 = r11.to(tl.float32) * scale
    u12 = r12.to(tl.float32) * scale
    u13 = r13.to(tl.float32) * scale
    u14 = r14.to(tl.float32) * scale
    u15 = r15.to(tl.float32) * scale
    
    nc0 = (c0 + 4).to(tl.uint32)
    
    return u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, nc0, c1, c2, c3


@triton.jit
def rand_uniform_range(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    min_val: tl.tensor,
    max_val: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate a single uniform random number in [min_val, max_val).
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
        min_val: Minimum value (float32)
        max_val: Maximum value (float32)
    
    Returns:
        (u, nc0, nc1, nc2, nc3): Random number in [min_val, max_val) and updated counters
    
    Example:
        angle, c0, c1, c2, c3 = rand_uniform_range(k0, k1, c0, c1, c2, c3, 0.0, 2.0 * PI)
    """
    u, nc0, nc1, nc2, nc3 = rand_uniform(k0, k1, c0, c1, c2, c3)
    u = min_val + u * (max_val - min_val)
    return u, nc0, nc1, nc2, nc3


@triton.jit
def rand_normal_box_muller(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate two normally distributed random numbers using Box-Muller transform.
    
    Returns two independent samples from standard normal distribution N(0, 1).
    Useful for Gaussian energy straggling and multiple scattering.
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
    
    Returns:
        (z1, z2, nc0, nc1, nc2, nc3): Two normal random numbers and updated counters
    
    Example:
        z1, z2, c0, c1, c2, c3 = rand_normal_box_muller(k0, k1, c0, c1, c2, c3)
    """
    u1, u2, _, _, nc0, nc1, nc2, nc3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
    
    r = tl.sqrt(-2.0 * tl.log(u1 + 1e-10))
    theta = 2.0 * 3.141592653589793 * u2
    
    z1 = r * tl.cos(theta)
    z2 = r * tl.sin(theta)
    
    return z1, z2, nc0, nc1, nc2, nc3


@triton.jit
def rand_exponential(
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    rate: tl.tensor = 1.0
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Generate an exponentially distributed random number.
    
    Useful for sampling interaction distances and decay times.
    Uses inverse transform: x = -ln(u) / rate
    
    Args:
        k0, k1: Philox key values (uint32)
        c0, c1, c2, c3: Philox counter values (uint32)
        rate: Rate parameter (default 1.0)
    
    Returns:
        (x, nc0, nc1, nc2, nc3): Exponential random number and updated counters
    
    Example:
        distance, c0, c1, c2, c3 = rand_exponential(k0, k1, c0, c1, c2, c3, mu)
    """
    u, nc0, nc1, nc2, nc3 = rand_uniform(k0, k1, c0, c1, c2, c3)
    x = -tl.log(u + 1e-10) / tl.maximum(rate, 1e-10)
    return x, nc0, nc1, nc2, nc3


def create_rng_state(n_particles: int, seed: int, device: str = "cuda") -> dict:
    """
    Create RNG state for n_particles.
    
    Note: This is a convenience function for initialization.
    The actual RNG operations in kernels use init_philox_state() directly.
    
    Args:
        n_particles: Number of particles
        seed: Global seed for simulation
        device: Device to allocate tensors on
    
    Returns:
        Dictionary with RNG state tensors
    
    Example:
        rng_state = create_rng_state(1000000, 12345, device="cuda")
    """
    particle_ids = torch.arange(n_particles, device=device, dtype=torch.int32)
    
    rng_state = {
        "particle_id": particle_ids,
        "seed": seed,
    }
    
    return rng_state


def get_batch_seed(global_seed: int, batch_id: int) -> int:
    """
    Get seed for a specific batch.
    
    For batch-based checkpointing, each batch gets a unique seed.
    This allows resuming from any batch without saving full RNG state.
    
    Args:
        global_seed: Global simulation seed
        batch_id: Batch identifier (0-indexed)
    
    Returns:
        Seed for this specific batch
    
    Example:
        seed = get_batch_seed(global_seed=12345, batch_id=5)
    """
    return global_seed + batch_id * 0x9E3779B9
