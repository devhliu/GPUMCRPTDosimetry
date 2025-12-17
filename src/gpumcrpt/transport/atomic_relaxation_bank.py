from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import triton

from gpumcrpt.kernels.triton.atomic_relaxation import atomic_relaxation_kernel
from gpumcrpt.kernels.triton.bank_append import bank_append_photons_kernel, bank_append_electrons_kernel


@dataclass
class AtomicRelaxationBankConfig:
    block: int = 256
    num_warps: int = 4
    photon_cut_MeV: float = 0.005
    e_cut_MeV: float = 0.010


@torch.no_grad()
def run_atomic_relaxation_to_banks(
    *,
    vac_bank: Dict[str, torch.Tensor],
    vac_active_indices: torch.Tensor,     # int32 [Nv_active]
    tables,
    edep_flat: torch.Tensor,
    # photon/electron banks + counters
    photon_bank: Dict[str, torch.Tensor],
    photon_count: torch.Tensor,           # int32[1]
    electron_bank: Dict[str, torch.Tensor],
    electron_count: torch.Tensor,         # int32[1]
    # geometry
    Z: int, Y: int, X: int,
    voxel_size_cm: Tuple[float, float, float],
    cfg: AtomicRelaxationBankConfig,
) -> None:
    """
    For each active vacancy:
      - atomic relaxation creates either X-ray or Auger (or deposits locally)
      - emits above cutoff are appended into photon/electron banks via bank_append kernels
      - vacancy is marked dead in vac_bank['alive']
    """
    Nv = int(vac_active_indices.numel())
    if Nv == 0:
        return

    device = edep_flat.device
    vx, vy, vz = voxel_size_cm

    # Gather vacancy fields by active indices (this is a standard bank pattern).
    # If you already have a gather kernel, use it; torch indexing is OK if Nv is not huge,
    # but for performance you'd typically do gather in Triton/CUDA.
    idx = vac_active_indices.to(torch.int64)
    vac = {
        "pos_cm": vac_bank["pos_cm"][idx],
        "mat": vac_bank["mat"][idx],
        "shell": vac_bank["shell"][idx],
        "w": vac_bank["w"][idx],
        "rng_key0": vac_bank["rng_key0"][idx],
        "rng_key1": vac_bank["rng_key1"][idx],
        "rng_ctr0": vac_bank["rng_ctr0"][idx],
        "rng_ctr1": vac_bank["rng_ctr1"][idx],
        "rng_ctr2": vac_bank["rng_ctr2"][idx],
        "rng_ctr3": vac_bank["rng_ctr3"][idx],
    }

    # staging outputs (length Nv)
    ph_out = {
        "pos_cm": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "dir": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "E_MeV": torch.empty((Nv,), device=device, dtype=torch.float32),
        "w": torch.empty((Nv,), device=device, dtype=torch.float32),
        "rng_key0": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_key1": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr0": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr1": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr2": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr3": torch.empty((Nv,), device=device, dtype=torch.int32),
        "has": torch.empty((Nv,), device=device, dtype=torch.int8),
    }
    e_out = {
        "pos_cm": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "dir": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "E_MeV": torch.empty((Nv,), device=device, dtype=torch.float32),
        "w": torch.empty((Nv,), device=device, dtype=torch.float32),
        "rng_key0": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_key1": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr0": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr1": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr2": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr3": torch.empty((Nv,), device=device, dtype=torch.int32),
        "has": torch.empty((Nv,), device=device, dtype=torch.int8),
    }

    M = int(tables.relax_fluor_yield.shape[0])
    S = int(tables.relax_fluor_yield.shape[1])

    grid = (triton.cdiv(Nv, int(cfg.block)),)
    atomic_relaxation_kernel[grid](
        vac["pos_cm"], vac["mat"], vac["shell"], vac["w"],
        vac["rng_key0"], vac["rng_key1"],
        vac["rng_ctr0"], vac["rng_ctr1"], vac["rng_ctr2"], vac["rng_ctr3"],
        tables.relax_fluor_yield, tables.relax_E_xray_MeV, tables.relax_E_auger_MeV,
        ph_out["pos_cm"], ph_out["dir"], ph_out["E_MeV"], ph_out["w"],
        ph_out["rng_key0"], ph_out["rng_key1"],
        ph_out["rng_ctr0"], ph_out["rng_ctr1"], ph_out["rng_ctr2"], ph_out["rng_ctr3"],
        ph_out["has"],
        e_out["pos_cm"], e_out["dir"], e_out["E_MeV"], e_out["w"],
        e_out["rng_key0"], e_out["rng_key1"],
        e_out["rng_ctr0"], e_out["rng_ctr1"], e_out["rng_ctr2"], e_out["rng_ctr3"],
        e_out["has"],
        edep_flat,
        Nv=Nv,
        Z=Z, Y=Y, X=X,
        M=M, S=S,
        photon_cut_MeV=float(cfg.photon_cut_MeV),
        e_cut_MeV=float(cfg.e_cut_MeV),
        voxel_z_cm=float(vz),
        voxel_y_cm=float(vy),
        voxel_x_cm=float(vx),
        BLOCK=int(cfg.block),
        num_warps=int(cfg.num_warps),
    )

    # Append emitted photons/electrons into banks (compute ebin in append kernels)
    NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])

    bank_append_photons_kernel[(triton.cdiv(Nv, 256),)](
        ph_out["pos_cm"], ph_out["dir"], ph_out["E_MeV"], ph_out["w"],
        ph_out["rng_key0"], ph_out["rng_key1"],
        ph_out["rng_ctr0"], ph_out["rng_ctr1"], ph_out["rng_ctr2"], ph_out["rng_ctr3"],
        ph_out["has"],
        photon_bank["pos_cm"], photon_bank["dir"], photon_bank["E_MeV"], photon_bank["w"], photon_bank["ebin"],
        photon_bank["rng_key0"], photon_bank["rng_key1"],
        photon_bank["rng_ctr0"], photon_bank["rng_ctr1"], photon_bank["rng_ctr2"], photon_bank["rng_ctr3"],
        photon_bank["alive"],
        photon_count,
        Ns=Nv,
        NB=NB,
        common_log_E_min=float(tables.common_log_E_min),
        common_log_step_inv=float(tables.common_log_step_inv),
        BLOCK=256,
        num_warps=4,
    )

    bank_append_electrons_kernel[(triton.cdiv(Nv, 256),)](
        e_out["pos_cm"], e_out["dir"], e_out["E_MeV"], e_out["w"],
        e_out["rng_key0"], e_out["rng_key1"],
        e_out["rng_ctr0"], e_out["rng_ctr1"], e_out["rng_ctr2"], e_out["rng_ctr3"],
        e_out["has"],
        electron_bank["pos_cm"], electron_bank["dir"], electron_bank["E_MeV"], electron_bank["w"], electron_bank["ebin"],
        electron_bank["rng_key0"], electron_bank["rng_key1"],
        electron_bank["rng_ctr0"], electron_bank["rng_ctr1"], electron_bank["rng_ctr2"], electron_bank["rng_ctr3"],
        electron_bank["alive"],
        electron_count,
        Ns=Nv,
        NB=NB,
        common_log_E_min=float(tables.common_log_E_min),
        common_log_step_inv=float(tables.common_log_step_inv),
        BLOCK=256,
        num_warps=4,
    )

    # Mark processed vacancies dead in the bank (so compaction removes them)
    vac_bank["alive"][idx] = 0