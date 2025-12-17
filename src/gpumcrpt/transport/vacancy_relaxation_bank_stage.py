from __future__ import annotations

from typing import Tuple
import torch
import triton

from gpumcrpt.kernels.triton.atomic_relaxation_soa import atomic_relaxation_soa_kernel
from gpumcrpt.kernels.triton.bank_append_soa import append_photons_bank_soa_kernel, append_electrons_bank_soa_kernel


@torch.no_grad()
def phase10_dispatch_vacancy_relaxation_bank(
    *,
    vac_active_indices: torch.Tensor,     # int32 [Nv_active]
    # vacancy bank SoA + RNG + weight
    vac_x: torch.Tensor, vac_y: torch.Tensor, vac_z: torch.Tensor,
    vac_atom_Z: torch.Tensor, vac_shell_idx: torch.Tensor,
    vac_w: torch.Tensor,
    vac_rng_key0: torch.Tensor, vac_rng_key1: torch.Tensor,
    vac_rng_ctr0: torch.Tensor, vac_rng_ctr1: torch.Tensor, vac_rng_ctr2: torch.Tensor, vac_rng_ctr3: torch.Tensor,
    vac_status: torch.Tensor,

    # photon bank SoA
    ph_x: torch.Tensor, ph_y: torch.Tensor, ph_z: torch.Tensor,
    ph_dx: torch.Tensor, ph_dy: torch.Tensor, ph_dz: torch.Tensor,
    ph_E: torch.Tensor, ph_w: torch.Tensor, ph_ebin: torch.Tensor,
    ph_rng_key0: torch.Tensor, ph_rng_key1: torch.Tensor,
    ph_rng_ctr0: torch.Tensor, ph_rng_ctr1: torch.Tensor, ph_rng_ctr2: torch.Tensor, ph_rng_ctr3: torch.Tensor,
    ph_status: torch.Tensor,

    # electron bank SoA
    el_x: torch.Tensor, el_y: torch.Tensor, el_z: torch.Tensor,
    el_dx: torch.Tensor, el_dy: torch.Tensor, el_dz: torch.Tensor,
    el_E: torch.Tensor, el_w: torch.Tensor, el_ebin: torch.Tensor,
    el_rng_key0: torch.Tensor, el_rng_key1: torch.Tensor,
    el_rng_ctr0: torch.Tensor, el_rng_ctr1: torch.Tensor, el_rng_ctr2: torch.Tensor, el_rng_ctr3: torch.Tensor,
    el_status: torch.Tensor,

    global_counters: torch.Tensor,   # int32[>=3]
    tables,                           # relaxation tables indexed by atomic Z: [Zmax+1,S]

    edep_flat: torch.Tensor,
    Zdim: int, Ydim: int, Xdim: int,
    voxel_size_cm: Tuple[float, float, float],
    photon_cut_MeV: float,
    e_cut_MeV: float,

    block: int = 256,
) -> None:
    Nv = int(vac_active_indices.numel())
    if Nv == 0:
        return

    idx64 = vac_active_indices.to(torch.int64)
    device = vac_x.device
    vx, vy, vz = voxel_size_cm

    # gather vacancy staging
    in_x = vac_x[idx64]; in_y = vac_y[idx64]; in_z = vac_z[idx64]
    in_Z = vac_atom_Z[idx64]; in_shell = vac_shell_idx[idx64]
    in_w = vac_w[idx64]
    in_k0 = vac_rng_key0[idx64]; in_k1 = vac_rng_key1[idx64]
    in_c0 = vac_rng_ctr0[idx64]; in_c1 = vac_rng_ctr1[idx64]; in_c2 = vac_rng_ctr2[idx64]; in_c3 = vac_rng_ctr3[idx64]

    # staging outputs
    phx = torch.empty((Nv,), device=device, dtype=torch.float32)
    phy = torch.empty((Nv,), device=device, dtype=torch.float32)
    phz = torch.empty((Nv,), device=device, dtype=torch.float32)
    phd_x = torch.empty((Nv,), device=device, dtype=torch.float32)
    phd_y = torch.empty((Nv,), device=device, dtype=torch.float32)
    phd_z = torch.empty((Nv,), device=device, dtype=torch.float32)
    phE = torch.empty((Nv,), device=device, dtype=torch.float32)
    phw = torch.empty((Nv,), device=device, dtype=torch.float32)
    phk0 = torch.empty((Nv,), device=device, dtype=torch.int32)
    phk1 = torch.empty((Nv,), device=device, dtype=torch.int32)
    phc0 = torch.empty((Nv,), device=device, dtype=torch.int32)
    phc1 = torch.empty((Nv,), device=device, dtype=torch.int32)
    phc2 = torch.empty((Nv,), device=device, dtype=torch.int32)
    phc3 = torch.empty((Nv,), device=device, dtype=torch.int32)
    ph_has = torch.empty((Nv,), device=device, dtype=torch.int8)

    ex = torch.empty((Nv,), device=device, dtype=torch.float32)
    ey = torch.empty((Nv,), device=device, dtype=torch.float32)
    ez = torch.empty((Nv,), device=device, dtype=torch.float32)
    edx = torch.empty((Nv,), device=device, dtype=torch.float32)
    edy = torch.empty((Nv,), device=device, dtype=torch.float32)
    edz = torch.empty((Nv,), device=device, dtype=torch.float32)
    eE = torch.empty((Nv,), device=device, dtype=torch.float32)
    ew = torch.empty((Nv,), device=device, dtype=torch.float32)
    ek0 = torch.empty((Nv,), device=device, dtype=torch.int32)
    ek1 = torch.empty((Nv,), device=device, dtype=torch.int32)
    ec0 = torch.empty((Nv,), device=device, dtype=torch.int32)
    ec1 = torch.empty((Nv,), device=device, dtype=torch.int32)
    ec2 = torch.empty((Nv,), device=device, dtype=torch.int32)
    ec3 = torch.empty((Nv,), device=device, dtype=torch.int32)
    e_has = torch.empty((Nv,), device=device, dtype=torch.int8)

    Zmax = int(tables.relax_fluor_yield.shape[0] - 1)
    S = int(tables.relax_fluor_yield.shape[1])

    grid = (triton.cdiv(Nv, block),)
    atomic_relaxation_soa_kernel[grid](
        in_x, in_y, in_z,
        in_Z, in_shell, in_w,
        in_k0, in_k1, in_c0, in_c1, in_c2, in_c3,
        tables.relax_fluor_yield,
        tables.relax_E_xray_MeV,
        tables.relax_E_auger_MeV,
        Zmax=Zmax,
        S=S,
        out_ph_x_ptr=phx, out_ph_y_ptr=phy, out_ph_z_ptr=phz,
        out_ph_dx_ptr=phd_x, out_ph_dy_ptr=phd_y, out_ph_dz_ptr=phd_z,
        out_ph_E_ptr=phE, out_ph_w_ptr=phw,
        out_ph_rng_key0_ptr=phk0, out_ph_rng_key1_ptr=phk1,
        out_ph_rng_ctr0_ptr=phc0, out_ph_rng_ctr1_ptr=phc1, out_ph_rng_ctr2_ptr=phc2, out_ph_rng_ctr3_ptr=phc3,
        out_ph_has_ptr=ph_has,
        out_e_x_ptr=ex, out_e_y_ptr=ey, out_e_z_ptr=ez,
        out_e_dx_ptr=edx, out_e_dy_ptr=edy, out_e_dz_ptr=edz,
        out_e_E_ptr=eE, out_e_w_ptr=ew,
        out_e_rng_key0_ptr=ek0, out_e_rng_key1_ptr=ek1,
        out_e_rng_ctr0_ptr=ec0, out_e_rng_ctr1_ptr=ec1, out_e_rng_ctr2_ptr=ec2, out_e_rng_ctr3_ptr=ec3,
        out_e_has_ptr=e_has,
        edep_flat_ptr=edep_flat,
        Zdim=Zdim, Ydim=Ydim, Xdim=Xdim,
        voxel_x_cm=float(vx), voxel_y_cm=float(vy), voxel_z_cm=float(vz),
        photon_cut_MeV=float(photon_cut_MeV),
        e_cut_MeV=float(e_cut_MeV),
        Nv=Nv,
        BLOCK=block,
        num_warps=4,
    )

    # append emitted photons/electrons into banks (compute log-uniform ebin in-kernel)
    NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])

    append_photons_bank_soa_kernel[(triton.cdiv(Nv, 256),)](
        phx, phy, phz,
        phd_x, phd_y, phd_z,
        phE, phw,
        phk0, phk1, phc0, phc1, phc2, phc3,
        ph_has,
        ph_x, ph_y, ph_z,
        ph_dx, ph_dy, ph_dz,
        ph_E, ph_w, ph_ebin,
        ph_rng_key0, ph_rng_key1, ph_rng_ctr0, ph_rng_ctr1, ph_rng_ctr2, ph_rng_ctr3,
        ph_status,
        global_counters,
        PHOTON_COUNT_IDX=0,
        NB=NB,
        common_log_E_min=float(tables.common_log_E_min),
        common_log_step_inv=float(tables.common_log_step_inv),
        Ns=Nv,
        BLOCK=256,
        num_warps=4,
    )

    append_electrons_bank_soa_kernel[(triton.cdiv(Nv, 256),)](
        ex, ey, ez,
        edx, edy, edz,
        eE, ew,
        ek0, ek1, ec0, ec1, ec2, ec3,
        e_has,
        el_x, el_y, el_z,
        el_dx, el_dy, el_dz,
        el_E, el_w, el_ebin,
        el_rng_key0, el_rng_key1, el_rng_ctr0, el_rng_ctr1, el_rng_ctr2, el_rng_ctr3,
        el_status,
        global_counters,
        ELECTRON_COUNT_IDX=1,
        NB=NB,
        common_log_E_min=float(tables.common_log_E_min),
        common_log_step_inv=float(tables.common_log_step_inv),
        Ns=Nv,
        BLOCK=256,
        num_warps=4,
    )

    # mark processed vacancies dead
    vac_status[idx64] = 0