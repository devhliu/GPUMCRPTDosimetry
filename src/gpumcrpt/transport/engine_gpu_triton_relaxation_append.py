from __future__ import annotations

from typing import Dict, Optional
import torch
import triton

from gpumcrpt.kernels.triton.energy_bin import compute_ebin_log_uniform_kernel


@torch.no_grad()
def compute_ebin_log_uniform(E_MeV: torch.Tensor, out_ebin: torch.Tensor, *, common_log_E_min: float, common_log_step_inv: float, NB: int) -> None:
    n = int(E_MeV.numel())
    if n == 0:
        return
    grid = (triton.cdiv(n, 256),)
    compute_ebin_log_uniform_kernel[grid](
        E_MeV, out_ebin,
        n=n,
        common_log_E_min=float(common_log_E_min),
        common_log_step_inv=float(common_log_step_inv),
        NB=int(NB),
        BLOCK=256,
        num_warps=4,
    )


@torch.no_grad()
def compact_and_append_relaxation_products(
    *,
    tables,
    q_p: Optional[Dict[str, torch.Tensor]],
    q_e: Optional[Dict[str, torch.Tensor]],
    ph_out: Dict[str, torch.Tensor],
    e_out: Dict[str, torch.Tensor],
) -> tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    # --- photons ---
    ph_mask = ph_out["has"].bool()
    if torch.any(ph_mask):
        ph = {k: ph_out[k][ph_mask] for k in ["pos_cm","dir","E_MeV","w","rng_key0","rng_key1","rng_ctr0","rng_ctr1","rng_ctr2","rng_ctr3"]}
        ph["ebin"] = torch.empty((ph["E_MeV"].shape[0],), device=ph["E_MeV"].device, dtype=torch.int32)
        NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])
        compute_ebin_log_uniform(ph["E_MeV"], ph["ebin"], common_log_E_min=float(tables.common_log_E_min), common_log_step_inv=float(tables.common_log_step_inv), NB=NB)

        if q_p is None:
            q_p = ph
        else:
            for k in ph.keys():
                q_p[k] = torch.cat([q_p[k], ph[k]], dim=0)

    # --- electrons ---
    e_mask = e_out["has"].bool()
    if torch.any(e_mask):
        ee = {k: e_out[k][e_mask] for k in ["pos_cm","dir","E_MeV","w","rng_key0","rng_key1","rng_ctr0","rng_ctr1","rng_ctr2","rng_ctr3"]}
        ee["ebin"] = torch.empty((ee["E_MeV"].shape[0],), device=ee["E_MeV"].device, dtype=torch.int32)
        NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])
        compute_ebin_log_uniform(ee["E_MeV"], ee["ebin"], common_log_E_min=float(tables.common_log_E_min), common_log_step_inv=float(tables.common_log_step_inv), NB=NB)

        if q_e is None:
            q_e = ee
        else:
            for k in ee.keys():
                if k in q_e:
                    q_e[k] = torch.cat([q_e[k], ee[k]], dim=0)
                else:
                    q_e[k] = ee[k]

    return q_p, q_e