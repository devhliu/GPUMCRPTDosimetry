from __future__ import annotations

import triton
import triton.language as tl

from .rng import rand_uniform_u01


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

    # Create block pointers for efficient memory access
    pos_block = tl.make_block_ptr(
        base=pos_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    dir_block = tl.make_block_ptr(
        base=dir_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    
    # Load particle data using block pointers with cache hints
    pos_data = tl.load(pos_block, boundary_check=(0, 1), mask=mask, other=0.0, cache_modifier=".cg")
    dir_data = tl.load(dir_block, boundary_check=(0, 1), mask=mask, other=0.0, cache_modifier=".cg")
    
    z = pos_data[:, 0]
    y = pos_data[:, 1]
    x = pos_data[:, 2]
    
    uz = dir_data[:, 0]
    uy = dir_data[:, 1]
    ux = dir_data[:, 2]

    # Load scalar data with cache hints
    E = tl.load(E_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    w = tl.load(w_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
    rng = tl.load(rng_ptr + offs, mask=mask, other=123456789, cache_modifier=".cg")

    ebin = tl.load(ebin_ptr + offs, mask=mask, other=0, cache_modifier=".cg").to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))

    # Load sigma_max using block pointer for better cache utilization
    sigma_max_block = tl.make_block_ptr(
        base=sigma_max_ptr, shape=(ECOUNT,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK_SIZE,), order=(0,)
    )
    sigma_max = tl.load(sigma_max_block, boundary_check=(0,), mask=mask, other=1e-3, cache_modifier=".ca")

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

    # Load material data with cache hints
    mat = tl.load(material_id_ptr + lin, mask=inside & mask, other=0, cache_modifier=".cg").to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))

    rho = tl.load(rho_ptr + lin, mask=inside & mask, other=1e-3, cache_modifier=".cg").to(tl.float32)
    rho_ref = tl.load(ref_rho_ptr + mat, mask=inside & mask, other=1.0, cache_modifier=".ca").to(tl.float32)
    rho_scale = rho / tl.maximum(rho_ref, 1e-6)

    # Load sigma_total using efficient indexing
    xs_off = mat * ECOUNT + ebin
    sigma_tot = tl.load(sigma_total_ptr + xs_off, mask=inside & mask, other=0.0, cache_modifier=".ca") * rho_scale

    u2, rng = rand_uniform_u01(rng)
    accept = u2 < (sigma_tot / tl.maximum(sigma_max, 1e-12))
    real = accept & inside
    alive = inside

    # Store results using block pointers for efficient writes
    out_pos_data = tl.zeros((BLOCK_SIZE, 3), dtype=tl.float32)
    out_pos_data[:, 0] = z2
    out_pos_data[:, 1] = y2
    out_pos_data[:, 2] = x2
    
    out_pos_block = tl.make_block_ptr(
        base=out_pos_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    tl.store(out_pos_block, out_pos_data, boundary_check=(0, 1), mask=mask)

    out_dir_data = tl.zeros((BLOCK_SIZE, 3), dtype=tl.float32)
    out_dir_data[:, 0] = uz
    out_dir_data[:, 1] = uy
    out_dir_data[:, 2] = ux
    
    out_dir_block = tl.make_block_ptr(
        base=out_dir_ptr, shape=(N, 3), strides=(3, 1), offsets=(pid * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, 3), order=(1, 0)
    )
    tl.store(out_dir_block, out_dir_data, boundary_check=(0, 1), mask=mask)

    # Store scalar outputs
    tl.store(out_E_ptr + offs, E, mask=mask)
    tl.store(out_w_ptr + offs, w, mask=mask)
    tl.store(out_rng_ptr + offs, rng, mask=mask)
    tl.store(out_ebin_ptr + offs, ebin, mask=mask)
    tl.store(out_alive_ptr + offs, alive.to(tl.int8), mask=mask)
    tl.store(out_real_ptr + offs, real.to(tl.int8), mask=mask)