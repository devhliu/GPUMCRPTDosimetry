from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.rng.philox import init_philox_state, rand_uniform4
from gpumcrpt.transport.triton_kernels.utils.gpu_math import fast_sqrt_approx, fast_sin_cos_approx


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
    key=['Nv'],
    warmup=10,
    rep=20,
)
@triton.jit
def atomic_relaxation_kernel(
    vac_x_ptr, vac_y_ptr, vac_z_ptr,
    vac_Z_ptr,
    vac_shell_ptr,
    vac_w_ptr,
    vac_id_ptr,

    fluor_yield_ptr,
    E_xray_MeV_ptr,
    E_auger_MeV_ptr,
    Zmax: tl.constexpr,
    S: tl.constexpr,
    rng_seed: tl.tensor,

    out_ph_x_ptr, out_ph_y_ptr, out_ph_z_ptr,
    out_ph_dx_ptr, out_ph_dy_ptr, out_ph_dz_ptr,
    out_ph_E_ptr, out_ph_w_ptr,
    out_ph_has_ptr,

    out_e_x_ptr, out_e_y_ptr, out_e_z_ptr,
    out_e_dx_ptr, out_e_dy_ptr, out_e_dz_ptr,
    out_e_E_ptr, out_e_w_ptr,
    out_e_has_ptr,

    edep_flat_ptr,
    Zdim: tl.constexpr, Ydim: tl.constexpr, Xdim: tl.constexpr,
    voxel_x_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_z_cm: tl.constexpr,

    photon_cut_MeV: tl.constexpr,
    e_cut_MeV: tl.constexpr,

    Nv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < Nv

    x = tl.load(vac_x_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(vac_y_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(vac_z_ptr + i, mask=m, other=0.0).to(tl.float32)
    Zatom = tl.load(vac_Z_ptr + i, mask=m, other=1).to(tl.int32)
    shell = tl.load(vac_shell_ptr + i, mask=m, other=0).to(tl.int32)
    w = tl.load(vac_w_ptr + i, mask=m, other=0.0).to(tl.float32)
    vac_id = tl.load(vac_id_ptr + i, mask=m, other=0).to(tl.int64)

    Zatom = tl.maximum(1, tl.minimum(Zmax, Zatom))
    shell = tl.maximum(0, tl.minimum(S - 1, shell))
    base = Zatom * S + shell

    particle_id = vac_id.to(tl.uint32)
    k0, k1, c0, c1, c2, c3 = init_philox_state(particle_id, rng_seed)

    u0, u1, u2, _, c0, c1, c2, c3 = rand_uniform4(k0, k1, c0, c1, c2, c3)

    fy = tl.load(fluor_yield_ptr + base, mask=m, other=0.0).to(tl.float32)
    Ex = tl.load(E_xray_MeV_ptr + base, mask=m, other=0.0).to(tl.float32)
    Ea = tl.load(E_auger_MeV_ptr + base, mask=m, other=0.0).to(tl.float32)

    emit_x = u0 < fy

    cost = 2.0 * u1 - 1.0
    sint = fast_sqrt_approx(tl.maximum(0.0, 1.0 - cost * cost))
    phi = 6.28318530718 * u2
    cos_phi, sin_phi = fast_sin_cos_approx(phi)
    dx = sint * cos_phi
    dy = sint * sin_phi
    dz = cost

    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    inside = (ix >= 0) & (ix < Xdim) & (iy >= 0) & (iy < Ydim) & (iz >= 0) & (iz < Zdim)
    lin = iz * (Ydim * Xdim) + iy * Xdim + ix

    has_ph = m & inside & emit_x & (Ex >= photon_cut_MeV)
    has_e = m & inside & (~emit_x) & (Ea >= e_cut_MeV)

    dep_x = m & emit_x & (~has_ph)
    dep_e = m & (~emit_x) & (~has_e)
    depE = tl.where(dep_x, Ex * w, 0.0) + tl.where(dep_e, Ea * w, 0.0)
    tl.atomic_add(edep_flat_ptr + lin, depE, mask=inside & (depE > 0.0))

    tl.store(out_ph_x_ptr + i, x, mask=m)
    tl.store(out_ph_y_ptr + i, y, mask=m)
    tl.store(out_ph_z_ptr + i, z, mask=m)
    tl.store(out_ph_dx_ptr + i, dx, mask=m)
    tl.store(out_ph_dy_ptr + i, dy, mask=m)
    tl.store(out_ph_dz_ptr + i, dz, mask=m)
    tl.store(out_ph_E_ptr + i, Ex, mask=m)
    tl.store(out_ph_w_ptr + i, w, mask=m)
    tl.store(out_ph_has_ptr + i, has_ph.to(tl.int8), mask=m)

    tl.store(out_e_x_ptr + i, x, mask=m)
    tl.store(out_e_y_ptr + i, y, mask=m)
    tl.store(out_e_z_ptr + i, z, mask=m)
    tl.store(out_e_dx_ptr + i, dx, mask=m)
    tl.store(out_e_dy_ptr + i, dy, mask=m)
    tl.store(out_e_dz_ptr + i, dz, mask=m)
    tl.store(out_e_E_ptr + i, Ea, mask=m)
    tl.store(out_e_w_ptr + i, w, mask=m)
    tl.store(out_e_has_ptr + i, has_e.to(tl.int8), mask=m)
