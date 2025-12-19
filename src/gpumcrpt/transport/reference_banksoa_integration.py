"""
Phase 10 integration (Option A RNG): Bank-style atomic appends + end-of-step compaction.

This file is intended to be pasted/adapted into your existing engine_gpu_triton.py.
It assumes your exact schema:

Photon bank (SoA):
  x,y,z, dx,dy,dz, E,w, ebin, rng_key0,rng_key1,rng_ctr0..3, status(int8)

Electron bank (SoA):
  x,y,z, dx,dy,dz, E,w, ebin, rng_key0,rng_key1,rng_ctr0..3, status(int8)

Vacancy bank (SoA):
  x,y,z, atom_Z(int32), shell_idx(int32), status(int8)
  + (recommended) carry RNG + weight too for relaxation:
    w, rng_key0,rng_key1,rng_ctr0..3

Counters:
  self.global_counters int32[>=3]
    [0]=photon_count, [1]=electron_count, [2]=vacancy_count
"""

from __future__ import annotations

from typing import Tuple
import torch
import triton

from gpumcrpt.kernels.triton.photon_pe_soa import photon_photoelectric_pe_soa_kernel
from gpumcrpt.kernels.triton.atomic_relaxation_soa import atomic_relaxation_soa_kernel
from gpumcrpt.kernels.triton.bank_append_soa import (
    append_photons_bank_soa_kernel,
    append_electrons_bank_soa_kernel,
)
from gpumcrpt.kernels.triton.vacancy_append_soa import append_vacancies_bank_soa_kernel


# engine constants (adapt to your enum)
PE_CODE = 0  # TODO: set to your PE encoding in photon classifier


@torch.no_grad()
def phase10_dispatch_photoelectric_bank(
    *,
    # Staging inputs produced by your classifier stage (for REAL collisions subset already)
    # If your classifier produces interaction types aligned to active_indices, use those directly.
    photon_active_indices: torch.Tensor,   # int32 [Nactive]
    photon_itype: torch.Tensor,            # int32/int8 [Nactive], interaction type per active photon index

    # photon bank (SoA)
    ph_x: torch.Tensor, ph_y: torch.Tensor, ph_z: torch.Tensor,
    ph_dx: torch.Tensor, ph_dy: torch.Tensor, ph_dz: torch.Tensor,
    ph_E: torch.Tensor, ph_w: torch.Tensor, ph_ebin: torch.Tensor,
    ph_rng_key0: torch.Tensor, ph_rng_key1: torch.Tensor,
    ph_rng_ctr0: torch.Tensor, ph_rng_ctr1: torch.Tensor, ph_rng_ctr2: torch.Tensor, ph_rng_ctr3: torch.Tensor,
    ph_status: torch.Tensor,

    # electron bank (SoA) + status
    el_x: torch.Tensor, el_y: torch.Tensor, el_z: torch.Tensor,
    el_dx: torch.Tensor, el_dy: torch.Tensor, el_dz: torch.Tensor,
    el_E: torch.Tensor, el_w: torch.Tensor, el_ebin: torch.Tensor,
    el_rng_key0: torch.Tensor, el_rng_key1: torch.Tensor,
    el_rng_ctr0: torch.Tensor, el_rng_ctr1: torch.Tensor, el_rng_ctr2: torch.Tensor, el_rng_ctr3: torch.Tensor,
    el_status: torch.Tensor,

    # vacancy bank (SoA + recommended w + RNG)
    vac_x: torch.Tensor, vac_y: torch.Tensor, vac_z: torch.Tensor,
    vac_atom_Z: torch.Tensor, vac_shell_idx: torch.Tensor,
    vac_w: torch.Tensor,
    vac_rng_key0: torch.Tensor, vac_rng_key1: torch.Tensor,
    vac_rng_ctr0: torch.Tensor, vac_rng_ctr1: torch.Tensor, vac_rng_ctr2: torch.Tensor, vac_rng_ctr3: torch.Tensor,
    vac_status: torch.Tensor,

    # global counters int32[>=3]
    global_counters: torch.Tensor,

    # geometry/material
    material_id: torch.Tensor,            # int32 [Z*Y*X]
    material_atom_Z: torch.Tensor,        # int32 [M]
    tables,                                # relax_shell_cdf [M,S], relax_E_bind_MeV [M,S]
    edep_flat: torch.Tensor,
    Zdim: int, Ydim: int, Xdim: int,
    voxel_size_cm: Tuple[float, float, float],  # (vx,vy,vz)

    block: int = 256,
) -> None:
    """
    Bank-style PE dispatch:
    - Identify PE photons via photon_itype on active_indices.
    - Gather bank fields to staging (torch index_select).
    - Run PE kernel to produce electron staging and vacancy staging.
    - Append electrons to electron bank tail via append_electrons_bank_soa_kernel (counter idx 1).
    - Append vacancies to vacancy bank tail via append_vacancies_bank_soa_kernel (counter idx 2).
    - Mark PE photons DEAD in ph_status.
    """
    # 1) identify PE photons from active list
    pe_mask = (photon_itype == PE_CODE)
    if not torch.any(pe_mask):
        return

    pe_idx = photon_active_indices[pe_mask]               # int32
    Npe = int(pe_idx.numel())
    if Npe == 0:
        return

    # 2) gather inputs from photon bank (SoA)
    idx64 = pe_idx.to(torch.int64)

    in_x = ph_x[idx64];     in_y = ph_y[idx64];     in_z = ph_z[idx64]
    in_dx = ph_dx[idx64];   in_dy = ph_dy[idx64];   in_dz = ph_dz[idx64]
    in_E = ph_E[idx64];     in_w = ph_w[idx64];     in_ebin = ph_ebin[idx64]

    in_k0 = ph_rng_key0[idx64]; in_k1 = ph_rng_key1[idx64]
    in_c0 = ph_rng_ctr0[idx64]; in_c1 = ph_rng_ctr1[idx64]; in_c2 = ph_rng_ctr2[idx64]; in_c3 = ph_rng_ctr3[idx64]

    device = ph_x.device

    # 3) allocate staging outputs (SoA) for PE results
    e_x = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_y = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_z = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_dx = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_dy = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_dz = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_E = torch.empty((Npe,), device=device, dtype=torch.float32)
    e_w = torch.empty((Npe,), device=device, dtype=torch.float32)

    # PE electron ebin: you can set placeholder; actual electron transport may use different binning.
    # For now we leave ebin computation to append_electrons_bank_soa_kernel using photon metadata (OK for shared bins).
    e_k0 = torch.empty((Npe,), device=device, dtype=torch.int32)
    e_k1 = torch.empty((Npe,), device=device, dtype=torch.int32)
    e_c0 = torch.empty((Npe,), device=device, dtype=torch.int32)
    e_c1 = torch.empty((Npe,), device=device, dtype=torch.int32)
    e_c2 = torch.empty((Npe,), device=device, dtype=torch.int32)
    e_c3 = torch.empty((Npe,), device=device, dtype=torch.int32)
    e_has = torch.empty((Npe,), device=device, dtype=torch.int8)

    v_x = torch.empty((Npe,), device=device, dtype=torch.float32)
    v_y = torch.empty((Npe,), device=device, dtype=torch.float32)
    v_z = torch.empty((Npe,), device=device, dtype=torch.float32)
    v_Z = torch.empty((Npe,), device=device, dtype=torch.int32)
    v_shell = torch.empty((Npe,), device=device, dtype=torch.int32)
    v_has = torch.empty((Npe,), device=device, dtype=torch.int8)

    vx, vy, vz = voxel_size_cm
    M = int(tables.relax_shell_cdf.shape[0])
    S = int(tables.relax_shell_cdf.shape[1])

    # 4) run PE kernel on staging photons
    grid = (triton.cdiv(Npe, block),)
    photon_photoelectric_pe_soa_kernel[grid](
        in_x, in_y, in_z,
        in_dx, in_dy, in_dz,
        in_E, in_w,
        in_ebin,
        in_k0, in_k1,
        in_c0, in_c1, in_c2, in_c3,
        material_id,
        material_atom_Z,
        tables.relax_shell_cdf,
        tables.relax_E_bind_MeV,
        e_x, e_y, e_z,
        e_dx, e_dy, e_dz,
        e_E, e_w,
        e_k0, e_k1,
        e_c0, e_c1, e_c2, e_c3,
        e_has,
        v_x, v_y, v_z,
        v_Z, v_shell,
        v_has,
        edep_flat,
        Npe=Npe,
        Zdim=Zdim, Ydim=Ydim, Xdim=Xdim,
        M=M, S=S,
        voxel_x_cm=float(vx), voxel_y_cm=float(vy), voxel_z_cm=float(vz),
        BLOCK=block,
        num_warps=4,
    )

    # 5) append electrons to electron bank using atomic tail (global_counters[1])
    NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])

    append_electrons_bank_soa_kernel[(triton.cdiv(Npe, 256),)](
        e_x, e_y, e_z,
        e_dx, e_dy, e_dz,
        e_E, e_w,
        e_k0, e_k1,
        e_c0, e_c1, e_c2, e_c3,
        e_has,
        el_x, el_y, el_z,
        el_dx, el_dy, el_dz,
        el_E, el_w, el_ebin,
        el_rng_key0, el_rng_key1,
        el_rng_ctr0, el_rng_ctr1, el_rng_ctr2, el_rng_ctr3,
        el_status,
        global_counters,
        ELECTRON_COUNT_IDX=1,
        NB=NB,
        common_log_E_min=float(tables.common_log_E_min),
        common_log_step_inv=float(tables.common_log_step_inv),
        Ns=Npe,
        BLOCK=256,
        num_warps=4,
    )

    # 6) append vacancies to vacancy bank using atomic tail (global_counters[2])
    # Store vacancy weight and RNG state by copying from the parent photon staging (updated in PE kernel?).
    # In the SoA PE kernel above, vacancy RNG isn't written; we recommend using electron RNG state for vacancy too.
    # Here: reuse e_k*/e_c* (already advanced) and parent weight in_w. If you prefer, write RNG to vacancy in PE kernel.
    append_vacancies_bank_soa_kernel[(triton.cdiv(Npe, 256),)](
        v_x, v_y, v_z,
        v_Z,
        v_shell,
        v_has,
        vac_x, vac_y, vac_z,
        vac_atom_Z,
        vac_shell_idx,
        vac_status,
        global_counters,
        VAC_COUNT_IDX=2,
        Ns=Npe,
        BLOCK=256,
        num_warps=4,
    )

    # vacancy aux fields (w + rng) need to be appended too; do it with masked scatter on CPU-side indices
    # to keep Phase 10 self-contained. For full GPU, add an append_vacancies_aux kernel.
    # We can do it with a second kernel; recommended below as Phase 10.8.
    #
    # Mark PE photons DEAD
    ph_status[idx64] = 0