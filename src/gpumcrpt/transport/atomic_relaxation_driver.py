from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import triton

from gpumcrpt.kernels.triton.atomic_relaxation import atomic_relaxation_kernel


@dataclass
class AtomicRelaxationConfig:
    block: int = 256
    num_warps: int = 4
    photon_cut_MeV: float = 0.005
    e_cut_MeV: float = 0.010


@torch.no_grad()
def run_atomic_relaxation(
    vac_q: Dict[str, torch.Tensor],
    tables,
    edep_flat: torch.Tensor,
    *,
    Z: int, Y: int, X: int,
    voxel_size_cm: Tuple[float, float, float],
    cfg: AtomicRelaxationConfig,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    Nv = int(vac_q["mat"].numel())
    device = vac_q["pos_cm"].device
    if Nv == 0:
        empty = lambda shape, dtype: torch.empty(shape, device=device, dtype=dtype)
        ph0 = {
            "pos_cm": empty((0, 3), torch.float32),
            "dir": empty((0, 3), torch.float32),
            "E_MeV": empty((0,), torch.float32),
            "w": empty((0,), torch.float32),
            "ebin": empty((0,), torch.int32),  # filled later in engine
            "rng_key0": empty((0,), torch.int32),
            "rng_key1": empty((0,), torch.int32),
            "rng_ctr0": empty((0,), torch.int32),
            "rng_ctr1": empty((0,), torch.int32),
            "rng_ctr2": empty((0,), torch.int32),
            "rng_ctr3": empty((0,), torch.int32),
            "has": empty((0,), torch.int8),
        }
        e0 = {k: v.clone() for k, v in ph0.items()}
        return ph0, e0, {k: v[:0] for k, v in vac_q.items()}

    out_ph = {
        "pos_cm": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "dir": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "E_MeV": torch.empty((Nv,), device=device, dtype=torch.float32),
        "w": torch.empty((Nv,), device=device, dtype=torch.float32),
        "ebin": torch.empty((Nv,), device=device, dtype=torch.int32),  # computed later
        "rng_key0": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_key1": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr0": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr1": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr2": torch.empty((Nv,), device=device, dtype=torch.int32),
        "rng_ctr3": torch.empty((Nv,), device=device, dtype=torch.int32),
        "has": torch.empty((Nv,), device=device, dtype=torch.int8),
    }
    out_e = {
        "pos_cm": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "dir": torch.empty((Nv, 3), device=device, dtype=torch.float32),
        "E_MeV": torch.empty((Nv,), device=device, dtype=torch.float32),
        "w": torch.empty((Nv,), device=device, dtype=torch.float32),
        "ebin": torch.empty((Nv,), device=device, dtype=torch.int32),  # computed later
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
    vx, vy, vz = voxel_size_cm

    grid = (triton.cdiv(Nv, int(cfg.block)),)
    atomic_relaxation_kernel[grid](
        vac_q["pos_cm"], vac_q["mat"], vac_q["shell"], vac_q["w"],
        vac_q["rng_key0"], vac_q["rng_key1"],
        vac_q["rng_ctr0"], vac_q["rng_ctr1"], vac_q["rng_ctr2"], vac_q["rng_ctr3"],
        tables.relax_fluor_yield, tables.relax_E_xray_MeV, tables.relax_E_auger_MeV,
        out_ph["pos_cm"], out_ph["dir"], out_ph["E_MeV"], out_ph["w"],
        out_ph["rng_key0"], out_ph["rng_key1"],
        out_ph["rng_ctr0"], out_ph["rng_ctr1"], out_ph["rng_ctr2"], out_ph["rng_ctr3"],
        out_ph["has"],
        out_e["pos_cm"], out_e["dir"], out_e["E_MeV"], out_e["w"],
        out_e["rng_key0"], out_e["rng_key1"],
        out_e["rng_ctr0"], out_e["rng_ctr1"], out_e["rng_ctr2"], out_e["rng_ctr3"],
        out_e["has"],
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

    return out_ph, out_e, {k: v[:0] for k, v in vac_q.items()}