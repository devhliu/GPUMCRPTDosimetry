from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01


@triton.jit
def sample_inv_cdf_1d(
    inv_ptr,  # flattened [ECOUNT*K]
    ebin: tl.tensor,
    rng: tl.tensor,
    K: tl.constexpr,
):
    """
    Sample value from inverse-CDF table at given ebin:
      t = u*(K-1)
      i0 = floor(t), f = t - i0
      x = lerp(inv[ebin,i0], inv[ebin,i0+1], f)

    Returns: (x, new_rng)
    """
    u, rng = rand_uniform_u01(rng)
    t = u * (K - 1)
    i0 = tl.floor(t).to(tl.int32)
    i0 = tl.maximum(0, tl.minimum(i0, K - 2))
    f = t - i0.to(tl.float32)

    base = ebin * K + i0
    x0 = tl.load(inv_ptr + base, mask=True, other=0.0)
    x1 = tl.load(inv_ptr + base + 1, mask=True, other=1.0)
    x = x0 + f * (x1 - x0)
    x = tl.maximum(0.0, tl.minimum(x, 1.0))
    return x, rng