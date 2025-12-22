from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton.rng import rand_uniform_u01


@triton.jit
def positron_annihilation_at_rest_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr,
    edep_ptr,  # flattened voxels
    out_ph1_pos_ptr, out_ph1_dir_ptr, out_ph1_E_ptr, out_ph1_w_ptr,
    out_ph2_pos_ptr, out_ph2_dir_ptr, out_ph2_E_ptr, out_ph2_w_ptr,
    out_rng_ptr,
    material_id_ptr,  # for voxel index (only bounds check)
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    """
    For positrons below cutoff:
      - deposit remaining kinetic energy E locally (edep += E*w)
      - emit 2 photons of 0.511 MeV back-to-back
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

    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    tl.atomic_add(edep_ptr + lin, E * w, mask=inside)

    E511 = 0.511

    # photon 1 uses direction = current dir, photon 2 opposite
    tl.store(out_ph1_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_ph1_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_ph1_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_ph1_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_ph1_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_ph1_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_ph1_E_ptr + offs, E511, mask=True)
    tl.store(out_ph1_w_ptr + offs, w, mask=True)

    tl.store(out_ph2_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_ph2_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_ph2_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_ph2_dir_ptr + offs * 3 + 0, -uz, mask=True)
    tl.store(out_ph2_dir_ptr + offs * 3 + 1, -uy, mask=True)
    tl.store(out_ph2_dir_ptr + offs * 3 + 2, -ux, mask=True)
    tl.store(out_ph2_E_ptr + offs, E511, mask=True)
    tl.store(out_ph2_w_ptr + offs, w, mask=True)

    tl.store(out_rng_ptr + offs, rng, mask=True)