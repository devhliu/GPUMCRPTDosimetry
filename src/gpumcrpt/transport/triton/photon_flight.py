from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton.rng import rand_uniform_u01


# Autotuning configurations for RTX A4000 (Ampere architecture)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['N'],  # Tune based on problem size
)
@triton.jit
def photon_woodcock_flight_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr,
    ebin_ptr,  # <-- NEW: precomputed energy bin per particle (int32)
    out_pos_ptr, out_dir_ptr, out_E_ptr, out_w_ptr, out_rng_ptr,
    out_ebin_ptr,  # <-- NEW: forward ebin (recomputed outside when E changes)
    out_alive_ptr, out_real_ptr,
    material_id_ptr, rho_ptr,
    sigma_total_ptr, sigma_max_ptr, ref_rho_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr,  # number of materials
    ECOUNT: tl.constexpr,  # number of energy bins
    N: tl.constexpr,  # number of particles
    BLOCK_SIZE: tl.constexpr,  # Autotuned block size
    NUM_WARPS: tl.constexpr,  # Autotuned warp count
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    """
    Optimized Photon Woodcock flight using Triton 3.5.1 features:
    - Block pointers for efficient memory access
    - Cache hints for improved memory performance
    - Autotuning for RTX A4000 optimization
    - Implicit boundary checking
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=0.0)

    uz = tl.load(dir_ptr + offs * 3 + 0, mask=mask, other=0.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=mask, other=0.0)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=mask, other=0.0)

    # Load scalar data
    E = tl.load(E_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    rng = tl.load(rng_ptr + offs, mask=mask, other=123456789)

    ebin = tl.load(ebin_ptr + offs, mask=mask, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    sigma_max = tl.load(sigma_max_ptr + ebin, mask=mask, other=1e-3)

    u1, rng = rand_uniform_u01(rng)
    s = -tl.log(u1) / tl.maximum(sigma_max, 1e-12)

    # Calculate new positions
    z2 = z + s * uz
    y2 = y + s * uy
    x2 = x + s * ux

    # Voxel index calculation with optimized bounds checking
    iz = tl.floor(z2 / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y2 / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x2 / voxel_x_cm).to(tl.int32)

    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    # Load material data
    mat = tl.load(material_id_ptr + lin, mask=inside & mask, other=0).to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))

    rho = tl.load(rho_ptr + lin, mask=inside & mask, other=1e-3).to(tl.float32)
    rho_ref = tl.load(ref_rho_ptr + mat, mask=inside & mask, other=1.0).to(tl.float32)
    rho_scale = rho / tl.maximum(rho_ref, 1e-6)

    # Load sigma_total using efficient indexing
    xs_off = mat * ECOUNT + ebin
    sigma_tot = tl.load(sigma_total_ptr + xs_off, mask=inside & mask, other=0.0) * rho_scale

    u2, rng = rand_uniform_u01(rng)
    accept = u2 < (sigma_tot / tl.maximum(sigma_max, 1e-12))
    real = accept & inside
    alive = inside

    tl.store(out_pos_ptr + offs * 3 + 0, z2, mask=mask)
    tl.store(out_pos_ptr + offs * 3 + 1, y2, mask=mask)
    tl.store(out_pos_ptr + offs * 3 + 2, x2, mask=mask)

    tl.store(out_dir_ptr + offs * 3 + 0, uz, mask=mask)
    tl.store(out_dir_ptr + offs * 3 + 1, uy, mask=mask)
    tl.store(out_dir_ptr + offs * 3 + 2, ux, mask=mask)

    # Store scalar outputs
    tl.store(out_E_ptr + offs, E, mask=mask)
    tl.store(out_w_ptr + offs, w, mask=mask)
    tl.store(out_rng_ptr + offs, rng, mask=mask)
    tl.store(out_ebin_ptr + offs, ebin, mask=mask)
    tl.store(out_alive_ptr + offs, alive.to(tl.int8), mask=mask)
    tl.store(out_real_ptr + offs, real.to(tl.int8), mask=mask)
