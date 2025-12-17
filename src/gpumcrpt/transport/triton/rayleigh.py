from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01
from .compton import _rotate_dir_kernel


@triton.jit
def photon_rayleigh_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    # outputs (scattered photon only):
    out_pos_ptr, out_dir_ptr, out_E_ptr, out_w_ptr, out_rng_ptr, out_ebin_ptr,
    ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Rayleigh (coherent) scattering:
      - photon energy unchanged
      - direction changed

    MVP sampling: isotropic cosθ in [-1,1].
    Phase 5+ upgrade: use tabulated form-factor sampler from `.h5`.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    w = tl.load(w_ptr + offs, mask=True, other=0.0)
    rng = tl.load(rng_ptr + offs, mask=True, other=123456789)
    ebin = tl.load(ebin_ptr + offs, mask=True, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)
    uz = tl.load(dir_ptr + offs * 3 + 0, mask=True, other=1.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=True, other=0.0)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=True, other=0.0)

    u1, rng = rand_uniform_u01(rng)
    u2, rng = rand_uniform_u01(rng)

    # Thomson scattering angular distribution for coherent scattering baseline:
    #   pdf(mu)=3/8*(1+mu^2), mu in [-1,1]
    # Invert via a few Newton iterations (no rejection sampling).
    u = u1
    mu = 2.0 * u - 1.0
    # CDF(mu) = 1/2 + 3/8*(mu + mu^3/3)
    for _ in tl.static_range(4):
        F = 0.5 + 0.375 * (mu + (mu * mu * mu) / 3.0)
        fp = 0.375 * (1.0 + mu * mu)
        mu = mu - (F - u) / tl.maximum(fp, 1e-12)
        mu = tl.maximum(-1.0, tl.minimum(mu, 1.0))

    cos_t = mu
    phi = 2.0 * 3.1415926535 * u2
    nuz, nuy, nux = _rotate_dir_kernel(uz, uy, ux, cos_t, phi)

    tl.store(out_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_dir_ptr + offs * 3 + 0, nuz, mask=True)
    tl.store(out_dir_ptr + offs * 3 + 1, nuy, mask=True)
    tl.store(out_dir_ptr + offs * 3 + 2, nux, mask=True)
    tl.store(out_E_ptr + offs, E, mask=True)
    tl.store(out_w_ptr + offs, w, mask=True)
    tl.store(out_rng_ptr + offs, rng, mask=True)
    tl.store(out_ebin_ptr + offs, ebin, mask=True)