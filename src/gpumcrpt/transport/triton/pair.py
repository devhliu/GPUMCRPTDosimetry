from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01


@triton.jit
def photon_pair_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    # outputs
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr,
    out_p_pos_ptr, out_p_dir_ptr, out_p_E_ptr, out_p_w_ptr,
    out_rng_ptr,
    ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Pair production:
      - photon is absorbed
      - create e- and e+ with total kinetic energy K = E - 1.022 MeV
      - NO local deposit of 1.022 MeV (it is carried by annihilation photons later)

    MVP energy split: uniform fraction.
    Phase 5+ upgrade: tabulated energy split sampler.

    Output:
      e- kinetic energy: K * frac
      e+ kinetic energy: K * (1-frac)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    w = tl.load(w_ptr + offs, mask=True, other=0.0)
    rng = tl.load(rng_ptr + offs, mask=True, other=123456789)

    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)

    uz = tl.load(dir_ptr + offs * 3 + 0, mask=True, other=1.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=True, other=0.0)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=True, other=0.0)

    # kinetic available
    K = tl.maximum(E - 1.022, 0.0)

    u1, rng = rand_uniform_u01(rng)
    frac = u1
    Ee = K * frac
    Ep = K - Ee

    # e-
    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_e_E_ptr + offs, Ee, mask=True)
    tl.store(out_e_w_ptr + offs, w, mask=True)

    # e+
    tl.store(out_p_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_p_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_p_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_p_dir_ptr + offs * 3 + 0, -uz, mask=True)  # opposite direction MVP
    tl.store(out_p_dir_ptr + offs * 3 + 1, -uy, mask=True)
    tl.store(out_p_dir_ptr + offs * 3 + 2, -ux, mask=True)
    tl.store(out_p_E_ptr + offs, Ep, mask=True)
    tl.store(out_p_w_ptr + offs, w, mask=True)

    tl.store(out_rng_ptr + offs, rng, mask=True)