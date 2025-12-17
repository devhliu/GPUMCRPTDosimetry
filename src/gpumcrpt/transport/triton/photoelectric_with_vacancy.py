from __future__ import annotations

import triton
import triton.language as tl
from gpumcrpt.transport.triton.rng import rand_uniform_u01


@triton.jit
def photon_photoelectric_with_vacancy_kernel(
    pos_ptr, dir_ptr, E_ptr, w_ptr, rng_ptr, ebin_ptr,
    material_id_ptr,
    # relaxation tables:
    shell_cdf_ptr,   # float32 [M, S] cumulative probabilities for shell selection (last=1)
    E_bind_ptr,      # float32 [M, S] binding energies (MeV)
    M: tl.constexpr,
    S: tl.constexpr,
    # outputs:
    out_e_pos_ptr, out_e_dir_ptr, out_e_E_ptr, out_e_w_ptr, out_e_rng_ptr,
    # vacancy queue outputs (1 per PE event):
    out_vac_pos_ptr, out_vac_mat_ptr, out_vac_shell_ptr, out_vac_w_ptr,
    out_has_vac_ptr,   # int8 mask (1 if vacancy valid)
    # edep for local deposits (e.g., if E_e below cutoff handled outside, you can omit):
    edep_ptr,
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # Assume caller uses mask by N; if not, pass N and mask. Here keep mask=True placeholders.
    E = tl.load(E_ptr + offs, mask=True, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=True, other=0.0).to(tl.float32)
    rng = tl.load(rng_ptr + offs, mask=True, other=1234567).to(tl.int32)

    # position
    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)

    # voxel lin for material lookup
    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix
    mat = tl.load(material_id_ptr + lin, mask=inside, other=0).to(tl.int32)
    mat = tl.maximum(0, tl.minimum(mat, M - 1))

    # sample shell
    u, rng = rand_uniform_u01(rng)
    # compute shell index via CDF comparisons (branch-minimized)
    # shell_cdf layout: [M*S]
    base = mat * S
    cdf0 = tl.load(shell_cdf_ptr + base + 0, mask=True, other=1.0)
    shell = (u > cdf0).to(tl.int32)
    # unroll up to S=4/8 typical; for general S use a loop but keep S small
    for s in range(1, 8):  # supports S<=8
        c = tl.load(shell_cdf_ptr + base + s, mask=True, other=1.0)
        shell += (u > c).to(tl.int32)

    shell = tl.minimum(shell, S - 1)

    E_bind = tl.load(E_bind_ptr + base + shell, mask=True, other=0.0).to(tl.float32)
    Ee = tl.maximum(E - E_bind, 0.0)

    # create photoelectron (direction: reuse photon direction; you can isotropize later if desired)
    ux = tl.load(dir_ptr + offs * 3 + 2, mask=True, other=1.0)
    uy = tl.load(dir_ptr + offs * 3 + 1, mask=True, other=0.0)
    uz = tl.load(dir_ptr + offs * 3 + 0, mask=True, other=0.0)

    tl.store(out_e_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_e_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 0, uz, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 1, uy, mask=True)
    tl.store(out_e_dir_ptr + offs * 3 + 2, ux, mask=True)
    tl.store(out_e_E_ptr + offs, Ee, mask=True)
    tl.store(out_e_w_ptr + offs, w, mask=True)
    tl.store(out_e_rng_ptr + offs, rng, mask=True)

    # vacancy output
    has_vac = inside & (E_bind > 0.0)
    tl.store(out_vac_pos_ptr + offs * 3 + 0, z, mask=True)
    tl.store(out_vac_pos_ptr + offs * 3 + 1, y, mask=True)
    tl.store(out_vac_pos_ptr + offs * 3 + 2, x, mask=True)
    tl.store(out_vac_mat_ptr + offs, mat.to(tl.int32), mask=True)
    tl.store(out_vac_shell_ptr + offs, shell.to(tl.int8), mask=True)
    tl.store(out_vac_w_ptr + offs, w, mask=True)
    tl.store(out_has_vac_ptr + offs, has_vac.to(tl.int8), mask=True)

    # If you still want any local deposit here, keep it minimal; otherwise leave as 0.
    # (Binding energy is NOT deposited here; it is released by relaxation kernel.)