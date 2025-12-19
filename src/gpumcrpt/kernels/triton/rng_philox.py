from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _mulhi_u32(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    return ((a.to(tl.uint64) * b.to(tl.uint64)) >> 32).to(tl.uint32)


@triton.jit
def philox4x32_round(
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    k0: tl.tensor, k1: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
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
def philox4x32_10(
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    k0: tl.tensor, k1: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    for _ in range(10):
        c0, c1, c2, c3, k0, k1 = philox4x32_round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3, k0, k1


@triton.jit
def rng_u01_philox(
    key0: tl.tensor, key1: tl.tensor,
    ctr0: tl.tensor, ctr1: tl.tensor, ctr2: tl.tensor, ctr3: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
    r0, r1, r2, r3, k0, k1 = philox4x32_10(
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