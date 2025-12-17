from __future__ import annotations

import triton
import triton.language as tl

from .samplers import sample_inv_cdf_1d
from .rng import rand_uniform_u01


@triton.jit
def electron_brems_emit_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    brem_inv_cdf_ptr, K: tl.constexpr,
    # outputs: updated parent energy + spawned photon
    out_parent_E_ptr, out_rng_ptr,
    out_ph_pos_ptr, out_ph_dir_ptr, out_ph_E_ptr, out_ph_w_ptr,
    ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
    # Parent direction is currently unused for the emitted photon direction in MVP
    _uz = tl.load(dir_ptr + offs * 3 + 0, mask=True, other=1.0)
    _uy = tl.load(dir_ptr + offs * 3 + 1, mask=True, other=0.0)
    _ux = tl.load(dir_ptr + offs * 3 + 2, mask=True, other=0.0)

    Efrac, rng = sample_inv_cdf_1d(brem_inv_cdf_ptr, ebin, rng, K=K)
    Eph = Efrac * E
    Eph = tl.minimum(Eph, E)
    E2 = E - Eph

    # Isotropic photon direction:
    # mu ~ U[-1,1], phi ~ U[0, 2pi)
    u_mu, rng = rand_uniform_u01(rng)
    u_phi, rng = rand_uniform_u01(rng)
    mu = 2.0 * u_mu - 1.0
    phi = u_phi * 6.283185307179586
    sin_theta = tl.sqrt(tl.maximum(0.0, 1.0 - mu * mu))
    uz = mu
    uy = sin_theta * tl.sin(phi)
    ux = sin_theta * tl.cos(phi)

    tl.store(out_parent_E_ptr + offs, E2, mask=True)
    tl.store(out_rng_ptr + offs, rng, mask=True)

    # photon
    tl.store(out_ph_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_ph_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_ph_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_ph_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_ph_E_ptr + offs, Eph, mask=True)
    tl.store(out_ph_w_ptr + offs, w, mask=True)


@triton.jit
def electron_delta_emit_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    delta_inv_cdf_ptr, K: tl.constexpr,
    # outputs: updated parent energy + spawned delta electron
    out_parent_E_ptr, out_rng_ptr,
    out_de_pos_ptr, out_de_dir_ptr, out_de_E_ptr, out_de_w_ptr,
    ECOUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
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

    Efrac, rng = sample_inv_cdf_1d(delta_inv_cdf_ptr, ebin, rng, K=K)
    Ed = Efrac * E
    Ed = tl.minimum(Ed, E)
    E2 = E - Ed

    tl.store(out_parent_E_ptr + offs, E2, mask=True)
    tl.store(out_rng_ptr + offs, rng, mask=True)

    # delta electron: MVP emit along current dir
    tl.store(out_de_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_de_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_de_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_de_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_de_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_de_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_de_E_ptr + offs, Ed, mask=True)
    tl.store(out_de_w_ptr + offs, w, mask=True)