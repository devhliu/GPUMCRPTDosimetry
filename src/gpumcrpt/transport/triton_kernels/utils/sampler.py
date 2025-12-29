from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.rng import rand_uniform4


@triton.jit
def sample_inv_cdf_1d_philox(
    inv_ptr,
    ebin: tl.tensor,
    k0: tl.tensor, k1: tl.tensor,
    c0: tl.tensor, c1: tl.tensor, c2: tl.tensor, c3: tl.tensor,
    K: tl.constexpr,
):
    """
    Sample value from inverse-CDF table at given ebin using stateless Philox RNG:
      t = u*(K-1)
      i0 = floor(t), f = t - i0
      x = lerp(inv[ebin,i0], inv[ebin,i0+1], f)

    Returns: (x, nc0, nc1, nc2, nc3)
    """
    u1, u2, u3, u4, nc0, nc1, nc2, nc3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
    t = u1 * (K - 1)
    i0 = tl.floor(t).to(tl.int32)
    i0 = tl.maximum(0, tl.minimum(i0, K - 2))
    f = t - i0.to(tl.float32)

    base = ebin * K + i0
    x0 = tl.load(inv_ptr + base, mask=True, other=0.0)
    x1 = tl.load(inv_ptr + base + 1, mask=True, other=1.0)
    x = x0 + f * (x1 - x0)
    x = tl.maximum(0.0, tl.minimum(x, 1.0))
    return x, nc0, nc1, nc2, nc3
