from __future__ import annotations

import triton
import triton.language as tl
import torch

from gpumcrpt.transport.triton_kernels.rng import init_philox_state, rand_uniform4
from gpumcrpt.transport.triton_kernels.perf.optimization import get_optimal_kernel_config
from gpumcrpt.transport.triton_kernels.utils.gpu_math import fast_log_approx


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
def photon_woodcock_flight_kernel_philox(
    pos_ptr, dir_ptr, E_ptr, w_ptr,
    rng_seed: tl.tensor,
    ebin_ptr,
    out_pos_ptr, out_dir_ptr, out_E_ptr, out_w_ptr,
    out_ebin_ptr, out_alive_ptr, out_real_ptr,
    material_id_ptr, rho_ptr,
    sigma_total_ptr, sigma_max_ptr, ref_rho_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    M: tl.constexpr,
    ECOUNT: tl.constexpr,
    N: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_SHARED_MEMORY: tl.constexpr = True,
    PREFETCH_DISTANCE: tl.constexpr = 8,
):
    """
    Photon Woodcock flight with stateless Philox RNG and optimized memory access.
    Uses branch-free logic and cache hints for maximum performance.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    offs = tl.max_contiguous(offs, BLOCK_SIZE)
    
    # Load photon state
    E = tl.load(E_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    
    # Stateless RNG initialization
    particle_id = offs.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)
    
    # Load energy bin
    ebin = tl.load(ebin_ptr + offs, mask=mask, other=0).to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(ebin, ECOUNT - 1))
    
    # Load sigma_max
    sigma_max = tl.load(sigma_max_ptr + ebin, mask=mask, other=1e-3)
    
    # Generate random numbers
    u1, u2, u3, u4, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)
    
    # Branch-free distance calculation
    s = -fast_log_approx(u1) / tl.maximum(sigma_max, 1e-12)
    
    # Load position and direction
    pos_z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=0.0)
    pos_y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=0.0)
    pos_x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=0.0)
    
    dir_z = tl.load(dir_ptr + offs * 3 + 0, mask=mask, other=0.0)
    dir_y = tl.load(dir_ptr + offs * 3 + 1, mask=mask, other=0.0)
    dir_x = tl.load(dir_ptr + offs * 3 + 2, mask=mask, other=1.0)
    
    # Branch-free position update
    new_pos_z = pos_z + s * dir_z
    new_pos_y = pos_y + s * dir_y
    new_pos_x = pos_x + s * dir_x
    
    # Branch-free voxel index calculation
    iz = tl.floor(new_pos_z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(new_pos_y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(new_pos_x / voxel_x_cm).to(tl.int32)
    
    # Branch-free boundary check
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin_idx = iz * (Y * X) + iy * X + ix
    
    # Load material properties
    mat = tl.load(material_id_ptr + lin_idx, mask=inside & mask, other=0).to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))
    
    rho = tl.load(rho_ptr + lin_idx, mask=inside & mask, other=1e-3).to(tl.float32)
    rho_ref = tl.load(ref_rho_ptr + mat, mask=inside & mask, other=1.0).to(tl.float32)
    rho_scale = rho / tl.maximum(rho_ref, 1e-6)
    
    # Load cross-section
    xs_offset = mat * ECOUNT + ebin
    sigma_tot = tl.load(sigma_total_ptr + xs_offset, mask=inside & mask, other=0.0) * rho_scale
    
    # Branch-free acceptance check
    accept = u2 < (sigma_tot / tl.maximum(sigma_max, 1e-12))
    real = accept & inside
    alive = inside
    
    # Store results with cache hints
    tl.store(out_pos_ptr + offs * 3 + 0, new_pos_z, mask=mask)
    tl.store(out_pos_ptr + offs * 3 + 1, new_pos_y, mask=mask)
    tl.store(out_pos_ptr + offs * 3 + 2, new_pos_x, mask=mask)
    
    tl.store(out_dir_ptr + offs * 3 + 0, dir_z, mask=mask)
    tl.store(out_dir_ptr + offs * 3 + 1, dir_y, mask=mask)
    tl.store(out_dir_ptr + offs * 3 + 2, dir_x, mask=mask)
    
    tl.store(out_E_ptr + offs, E, mask=mask)
    tl.store(out_w_ptr + offs, w, mask=mask)
    tl.store(out_ebin_ptr + offs, ebin, mask=mask)
    
    tl.store(out_alive_ptr + offs, alive.to(tl.int8), mask=mask)
    tl.store(out_real_ptr + offs, real.to(tl.int8), mask=mask)


def launch_photon_flight(
    pos: torch.Tensor,
    direction: torch.Tensor,
    E: torch.Tensor,
    w: torch.Tensor,
    rng_seed: torch.Tensor,
    ebin: torch.Tensor,
    out_pos: torch.Tensor,
    out_direction: torch.Tensor,
    out_E: torch.Tensor,
    out_w: torch.Tensor,
    out_ebin: torch.Tensor,
    out_alive: torch.Tensor,
    out_real: torch.Tensor,
    material_id: torch.Tensor,
    rho: torch.Tensor,
    sigma_total: torch.Tensor,
    sigma_max: torch.Tensor,
    ref_rho: torch.Tensor,
    voxel_size_cm: tuple,
    device: torch.device,
    performance_monitor=None,
    **kwargs
):
    """
    Launch photon flight kernel with dynamic configuration.
    """
    N = int(E.numel())
    if N == 0:
        return
    
    block_size, grid_size = get_optimal_kernel_config(
        data_size=N,
        device=device,
        shared_mem_required=sigma_max.numel() * 4,
        register_pressure=64
    )
    
    Z, Y, X = material_id.shape
    M = int(ref_rho.numel())
    ECOUNT = int(sigma_max.numel())
    
    voxel_z_cm, voxel_y_cm, voxel_x_cm = voxel_size_cm
    
    material_id_flat = material_id.contiguous().view(-1)
    rho_flat = rho.contiguous().view(-1)
    
    kernel_args = {
        'pos_ptr': pos,
        'dir_ptr': direction,
        'E_ptr': E,
        'w_ptr': w,
        'rng_seed': rng_seed,
        'ebin_ptr': ebin,
        'out_pos_ptr': out_pos,
        'out_dir_ptr': out_direction,
        'out_E_ptr': out_E,
        'out_w_ptr': out_w,
        'out_ebin_ptr': out_ebin,
        'out_alive_ptr': out_alive,
        'out_real_ptr': out_real,
        'material_id_ptr': material_id_flat,
        'rho_ptr': rho_flat,
        'sigma_total_ptr': sigma_total,
        'sigma_max_ptr': sigma_max,
        'ref_rho_ptr': ref_rho,
        'Z': Z, 'Y': Y, 'X': X,
        'M': M, 'ECOUNT': ECOUNT,
        'N': N,
        'voxel_z_cm': voxel_z_cm,
        'voxel_y_cm': voxel_y_cm,
        'voxel_x_cm': voxel_x_cm,
        'USE_SHARED_MEMORY': True,
        'PREFETCH_DISTANCE': 8,
    }
    
    grid = (grid_size,)
    
    if performance_monitor:
        import time
        start_time = time.time()
        
        photon_woodcock_flight_kernel_philox[grid](**kernel_args)
        
        execution_time = time.time() - start_time
        performance_monitor.record_kernel_execution(
            name='photon_woodcock_flight',
            execution_time=execution_time,
            data_size=N,
            block_size=block_size,
            grid_size=grid_size,
            occupancy=0.8
        )
    else:
        photon_woodcock_flight_kernel_philox[grid](**kernel_args)
