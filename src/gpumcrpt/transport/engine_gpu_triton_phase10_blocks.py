from __future__ import annotations

from typing import Dict, Tuple
import torch
import triton

from gpumcrpt.kernels.triton.photon import photon_photoelectric_kernel
from gpumcrpt.kernels.triton.bank_append_vacancy import bank_append_vacancies_kernel
from gpumcrpt.kernels.triton.bank_append import bank_append_electrons_kernel


@torch.no_grad()
def dispatch_photoelectric_bank(
    *,
    # Photon bank + active list
    photon_bank: Dict[str, torch.Tensor],
    photon_active_indices: torch.Tensor,         # int32
    photon_itype: torch.Tensor,                  # int8/int32 per active photon (produced by your classify kernel)
    PE_CODE: int,

    # Electron bank + tail counter
    electron_bank: Dict[str, torch.Tensor],
    electron_count: torch.Tensor,                # int32[1]

    # Vacancy bank + tail counter
    vacancy_bank: Dict[str, torch.Tensor],
    vacancy_count: torch.Tensor,                 # int32[1]

    # tables/phantom
    tables,
    material_id: torch.Tensor,
    edep_flat: torch.Tensor,
    Z: int, Y: int, X: int,
    voxel_size_cm: Tuple[float, float, float],

    block: int = 256,
    num_warps: int = 4,
) -> None:
    """
    Bank-style PE:
      - gather PE photons by mask on photon_active_indices
      - run PE kernel into staging (Npe)
      - append electrons into electron_bank via bank_append_electrons_kernel
      - append vacancies into vacancy_bank via bank_append_vacancies_kernel
      - mark processed photons dead in photon_bank['alive'] (so compaction removes)
    """
    device = edep_flat.device
    vx, vy, vz = voxel_size_cm

    pe_mask = (photon_itype == PE_CODE)
    if not torch.any(pe_mask):
        return

    pe_active = photon_active_indices[pe_mask]
    Npe = int(pe_active.numel())
    if Npe == 0:
        return

    idx64 = pe_active.to(torch.int64)

    # Gather PE photon fields from bank
    pe = {
        "pos_cm": photon_bank["pos_cm"][idx64],
        "dir": photon_bank["dir"][idx64],
        "E_MeV": photon_bank["E_MeV"][idx64],
        "w": photon_bank["w"][idx64],
        "ebin": photon_bank["ebin"][idx64],
        "rng_key0": photon_bank["rng_key0"][idx64],
        "rng_key1": photon_bank["rng_key1"][idx64],
        "rng_ctr0": photon_bank["rng_ctr0"][idx64],
        "rng_ctr1": photon_bank["rng_ctr1"][idx64],
        "rng_ctr2": photon_bank["rng_ctr2"][idx64],
        "rng_ctr3": photon_bank["rng_ctr3"][idx64],
    }

    # STAGING outputs (length Npe) for electrons + vacancies
    e_out = {
        "pos_cm": torch.empty((Npe, 3), device=device, dtype=torch.float32),
        "dir": torch.empty((Npe, 3), device=device, dtype=torch.float32),
        "E_MeV": torch.empty((Npe,), device=device, dtype=torch.float32),
        "w": torch.empty((Npe,), device=device, dtype=torch.float32),
        "ebin": torch.empty((Npe,), device=device, dtype=torch.int32),  # PE passes through / optional
        "rng_key0": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_key1": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr0": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr1": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr2": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr3": torch.empty((Npe,), device=device, dtype=torch.int32),
        "has": torch.empty((Npe,), device=device, dtype=torch.int8),
    }
    v_out = {
        "pos_cm": torch.empty((Npe, 3), device=device, dtype=torch.float32),
        "mat": torch.empty((Npe,), device=device, dtype=torch.int32),
        "shell": torch.empty((Npe,), device=device, dtype=torch.int8),
        "w": torch.empty((Npe,), device=device, dtype=torch.float32),
        "rng_key0": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_key1": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr0": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr1": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr2": torch.empty((Npe,), device=device, dtype=torch.int32),
        "rng_ctr3": torch.empty((Npe,), device=device, dtype=torch.int32),
        "has": torch.empty((Npe,), device=device, dtype=torch.int8),
    }

    M = int(tables.relax_shell_cdf.shape[0])
    S = int(tables.relax_shell_cdf.shape[1])

    grid = (triton.cdiv(Npe, block),)
    photon_photoelectric_kernel[grid](
        pe["pos_cm"], pe["dir"], pe["E_MeV"], pe["w"], pe["ebin"],
        pe["rng_key0"], pe["rng_key1"], pe["rng_ctr0"], pe["rng_ctr1"], pe["rng_ctr2"], pe["rng_ctr3"],
        material_id,
        tables.relax_shell_cdf, tables.relax_E_bind_MeV,
        e_out["pos_cm"], e_out["dir"], e_out["E_MeV"], e_out["w"], e_out["ebin"],
        e_out["rng_key0"], e_out["rng_key1"], e_out["rng_ctr0"], e_out["rng_ctr1"], e_out["rng_ctr2"], e_out["rng_ctr3"],
        e_out["has"],
        v_out["pos_cm"], v_out["mat"], v_out["shell"], v_out["w"],
        v_out["rng_key0"], v_out["rng_key1"], v_out["rng_ctr0"], v_out["rng_ctr1"], v_out["rng_ctr2"], v_out["rng_ctr3"],
        v_out["has"],
        edep_flat,
        N=Npe,
        Z=Z, Y=Y, X=X,
        M=M, S=S,
        voxel_z_cm=float(vz),
        voxel_y_cm=float(vy),
        voxel_x_cm=float(vx),
        BLOCK=block,
        num_warps=num_warps,
    )

    # Append photoelectrons to electron bank (compute ebin inside append kernel)
    NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])
    bank_append_electrons_kernel[(triton.cdiv(Npe, 256),)](
        e_out["pos_cm"], e_out["dir"], e_out["E_MeV"], e_out["w"],
        e_out["rng_key0"], e_out["rng_key1"],
        e_out["rng_ctr0"], e_out["rng_ctr1"], e_out["rng_ctr2"], e_out["rng_ctr3"],
        e_out["has"],
        electron_bank["pos_cm"], electron_bank["dir"], electron_bank["E_MeV"], electron_bank["w"], electron_bank["ebin"],
        electron_bank["rng_key0"], electron_bank["rng_key1"],
        electron_bank["rng_ctr0"], electron_bank["rng_ctr1"], electron_bank["rng_ctr2"], electron_bank["rng_ctr3"],
        electron_bank["alive"],
        electron_count,
        Ns=Npe,
        NB=NB,
        common_log_E_min=float(tables.common_log_E_min),
        common_log_step_inv=float(tables.common_log_step_inv),
        BLOCK=256,
        num_warps=4,
    )

    # Append vacancies to vacancy bank
    bank_append_vacancies_kernel[(triton.cdiv(Npe, 256),)](
        v_out["pos_cm"], v_out["mat"], v_out["shell"], v_out["w"],
        v_out["rng_key0"], v_out["rng_key1"],
        v_out["rng_ctr0"], v_out["rng_ctr1"], v_out["rng_ctr2"], v_out["rng_ctr3"],
        v_out["has"],
        vacancy_bank["pos_cm"], vacancy_bank["mat"], vacancy_bank["shell"], vacancy_bank["w"],
        vacancy_bank["rng_key0"], vacancy_bank["rng_key1"],
        vacancy_bank["rng_ctr0"], vacancy_bank["rng_ctr1"], vacancy_bank["rng_ctr2"], vacancy_bank["rng_ctr3"],
        vacancy_bank["alive"],
        vacancy_count,
        Ns=Npe,
        BLOCK=256,
        num_warps=4,
    )

    # Mark PE photons dead (so compaction removes them)
    photon_bank["alive"][idx64] = 0