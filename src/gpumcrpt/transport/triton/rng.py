from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def xorshift32(state: tl.tensor) -> tl.tensor:
    """
    Simple xorshift RNG for MVP.
    NOTE: For production, replace with Philox/Threefry-like counter RNG.
    """
    x = state
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    return x


@triton.jit
def rand_uniform_u01(state: tl.tensor) -> tuple[tl.tensor, tl.tensor]:
    """
    Returns (u in (0,1), new_state).
    """
    new_state = xorshift32(state)
    # convert uint32 -> float in (0,1)
    u = tl.view(new_state, tl.uint32).to(tl.float32) * (1.0 / 4294967296.0)
    # avoid exact 0
    u = tl.maximum(u, 1e-12)
    return u, new_state


# Copy Philox implementation directly to avoid import issues in Triton 3.5.1
@triton.jit
def _mulhi_u32(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    return ((a.to(tl.uint64) * b.to(tl.uint64)) >> 32).to(tl.uint32)


@triton.jit
def philox4x32_round(c0, c1, c2, c3, k0, k1):
    M0 = tl.full(c0.shape, 0xD2511F53, tl.uint32)
    M1 = tl.full(c0.shape, 0xCD9E8D57, tl.uint32)
    W0 = tl.full(c0.shape, 0x9E3779B9, tl.uint32)
    W1 = tl.full(c0.shape, 0xBB67AE85, tl.uint32)

    hi0 = _mulhi_u32(M0, c0)
    lo0 = (M0 * c0).to(tl.uint32)
    hi1 = _mulhi_u32(M1, c2)
    lo1 = (M1 * c2).to(tl.uint32)

    nc0 = hi1 ^ c1 ^ k0
    nc1 = lo1
    nc2 = hi0 ^ c3 ^ k1
    nc3 = lo0

    nk0 = (k0 + W0).to(tl.uint32)
    nk1 = (k1 + W1).to(tl.uint32)
    return nc0, nc1, nc2, nc3, nk0, nk1


@triton.jit
def philox4x32_10(c0, c1, c2, c3, k0, k1):
    for _ in range(10):
        c0, c1, c2, c3, k0, k1 = philox4x32_round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


@triton.jit
def rng_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3):
    r0, r1, r2, r3 = philox4x32_10(
        ctr0.to(tl.uint32), ctr1.to(tl.uint32), ctr2.to(tl.uint32), ctr3.to(tl.uint32),
        key0.to(tl.uint32), key1.to(tl.uint32)
    )

    # u in [0,1)
    u0 = ((r0 >> 8).to(tl.float32)) * (1.0 / 16777216.0)
    u1 = ((r1 >> 8).to(tl.float32)) * (1.0 / 16777216.0)
    u2 = ((r2 >> 8).to(tl.float32)) * (1.0 / 16777216.0)
    u3 = ((r3 >> 8).to(tl.float32)) * (1.0 / 16777216.0)

    # advance ctr0 by 1 per call (fixed cost)
    ctr0 = (ctr0 + 1).to(tl.uint32)
    return u0, u1, u2, u3, ctr0, ctr1, ctr2, ctr3


@triton.jit
def rand_uniform_u01_philox(
    key0: tl.tensor, key1: tl.tensor,
    ctr0: tl.tensor, ctr1: tl.tensor, ctr2: tl.tensor, ctr3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Philox-based uniform random number generation.
    Returns (u0, u1, u2, u3, new_ctr0, new_ctr1, new_ctr2, new_ctr3)
    """
    u0, u1, u2, u3, new_ctr0, new_ctr1, new_ctr2, new_ctr3 = rng_u01_philox(
        key0, key1, ctr0, ctr1, ctr2, ctr3
    )
    
    # Avoid exact 0
    u0 = tl.maximum(u0, 1e-12)
    u1 = tl.maximum(u1, 1e-12)
    u2 = tl.maximum(u2, 1e-12)
    u3 = tl.maximum(u3, 1e-12)
    
    return u0, u1, u2, u3, new_ctr0, new_ctr1, new_ctr2, new_ctr3


@triton.jit
def philox_advance_counter(
    ctr0: tl.tensor, ctr1: tl.tensor, ctr2: tl.tensor, ctr3: tl.tensor,
    steps: tl.constexpr = 1
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    """
    Advance Philox counter by specified number of steps.
    """
    new_ctr0 = ctr0 + steps
    carry = (new_ctr0 < ctr0).to(tl.uint32)
    new_ctr1 = ctr1 + carry
    carry = (new_ctr1 < ctr1).to(tl.uint32)
    new_ctr2 = ctr2 + carry
    carry = (new_ctr2 < ctr2).to(tl.uint32)
    new_ctr3 = ctr3 + carry
    
    return new_ctr0, new_ctr1, new_ctr2, new_ctr3