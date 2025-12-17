from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CPUOracleConfig:
    photon_cut_MeV: float = 0.005
    electron_cut_MeV: float = 0.010
    boundary_mode: str = "absorbing"


def run_cpu_oracle_transport(
    *,
    primaries,
    alpha_local_edep_MeV: torch.Tensor,
    mats,
    tables,
    voxel_size_cm: tuple[float, float, float],
    cfg: CPUOracleConfig,
    seed: int = 0,
) -> torch.Tensor:
    """MVP CPU transport: deposit all primary energy locally.

    This is a correctness/bring-up helper and is NOT a physical transport.
    """
    Z, Y, X = mats.material_id.shape
    edep = torch.zeros((Z, Y, X), device="cpu", dtype=torch.float32)
    edep += alpha_local_edep_MeV.to(edep.dtype)

    def _deposit(queue):
        if queue is None:
            return
        if queue["E_MeV"].numel() == 0:
            return
        pos_cm = queue["pos_cm"].to("cpu", dtype=torch.float32)
        E = queue["E_MeV"].to("cpu", dtype=torch.float32)
        w = queue["w"].to("cpu", dtype=torch.float32)

        vx, vy, vz = voxel_size_cm
        z = torch.floor(pos_cm[:, 0] / float(vz)).to(torch.int64)
        y = torch.floor(pos_cm[:, 1] / float(vy)).to(torch.int64)
        x = torch.floor(pos_cm[:, 2] / float(vx)).to(torch.int64)

        m = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X) & (E > 0)
        if not torch.any(m):
            return

        lin = (z[m] * (Y * X) + y[m] * X + x[m]).to(torch.int64)
        edep.view(-1).index_add_(0, lin, (E[m] * w[m]).to(torch.float32))

    _deposit(primaries.photons)
    _deposit(primaries.electrons)
    _deposit(primaries.positrons)

    return edep
