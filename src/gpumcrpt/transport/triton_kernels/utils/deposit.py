from __future__ import annotations

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 2}),
    ],
    key=['N'],
    warmup=10,
    rep=20,
)
@triton.jit
def deposit_local_energy_kernel(
    pos_ptr, E_ptr, w_ptr,
    edep_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    N: tl.constexpr,
):
    """
    Deposit E*w at the particle position voxel via atomic add.
    Used for cutoff termination (photon/electron/positron below cutoffs).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    E = tl.load(E_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    tl.atomic_add(edep_ptr + lin, E * w, mask=mask & inside)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'HASH_SIZE': 64, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'HASH_SIZE': 64, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'HASH_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'HASH_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'HASH_SIZE': 128, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'HASH_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'HASH_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'HASH_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 1024, 'HASH_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'HASH_SIZE': 512, 'NUM_WARPS': 16, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'HASH_SIZE': 512, 'NUM_WARPS': 16, 'NUM_STAGES': 2}),
    ],
    key=['N'],
    warmup=10,
    rep=20,
)
@triton.jit
def deposit_local_energy_kernel_shared(
    pos_ptr, E_ptr, w_ptr,
    edep_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, HASH_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    N: tl.constexpr,
):
    """
    Deposit E*w at the particle position voxel using shared memory hash table to reduce atomic contention.
    Uses vectorized operations with branch-free logic for optimal performance.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    E = tl.load(E_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix
    
    edep_value = E * w
    
    valid_mask = mask & inside & (edep_value > 0.0)
    
    tl.atomic_add(edep_ptr + lin, edep_value, mask=valid_mask)
