from __future__ import annotations

from typing import Dict
import torch

from gpumcrpt.transport.queue_buffer import QueueBuffer
from gpumcrpt.transport.engine_gpu_triton_relaxation_append import compute_ebin_log_uniform


@torch.no_grad()
def append_relaxation_products_buffered(
    *,
    tables,
    q_p: QueueBuffer,
    q_e: QueueBuffer,
    ph_out: Dict[str, torch.Tensor],
    e_out: Dict[str, torch.Tensor],
) -> None:
    device = q_p.bufs["E_MeV"].device
    NB = int(tables.NB) if hasattr(tables, "NB") else int(tables.Sigma_total.shape[1])

    # photons
    ph_mask = ph_out["has"].bool()
    if torch.any(ph_mask):
        ph = {k: ph_out[k][ph_mask] for k in [
            "pos_cm","dir","E_MeV","w","rng_key0","rng_key1","rng_ctr0","rng_ctr1","rng_ctr2","rng_ctr3"
        ]}
        ph["ebin"] = torch.empty((ph["E_MeV"].shape[0],), device=device, dtype=torch.int32)
        compute_ebin_log_uniform(ph["E_MeV"], ph["ebin"],
                                 common_log_E_min=float(tables.common_log_E_min),
                                 common_log_step_inv=float(tables.common_log_step_inv),
                                 NB=NB)
        q_p.append(ph)

    # electrons
    e_mask = e_out["has"].bool()
    if torch.any(e_mask):
        ee = {k: e_out[k][e_mask] for k in [
            "pos_cm","dir","E_MeV","w","rng_key0","rng_key1","rng_ctr0","rng_ctr1","rng_ctr2","rng_ctr3"
        ]}
        ee["ebin"] = torch.empty((ee["E_MeV"].shape[0],), device=device, dtype=torch.int32)
        compute_ebin_log_uniform(ee["E_MeV"], ee["ebin"],
                                 common_log_E_min=float(tables.common_log_E_min),
                                 common_log_step_inv=float(tables.common_log_step_inv),
                                 NB=NB)
        q_e.append(ee)