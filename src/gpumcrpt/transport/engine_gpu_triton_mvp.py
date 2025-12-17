from __future__ import annotations

import torch


class TritonTransportEngine:
    """MVP transport engine (runnable end-to-end).

    Current behavior: deposit all particle kinetic energy locally in the voxel where it was born.

    This is intentionally simple to unblock Milestone 1 (runnable MVP). It should be
    replaced by the real wavefront/Triton transport engine in later milestones.
    """

    def __init__(
        self,
        *,
        mats,
        tables,
        sim_config: dict,
        voxel_size_cm: tuple[float, float, float],
        device: str = "cuda",
    ) -> None:
        self.mats = mats
        self.tables = tables
        self.sim_config = sim_config
        self.voxel_size_cm = voxel_size_cm
        self.device = device

    @torch.no_grad()
    def run_one_batch(self, primaries, alpha_local_edep: torch.Tensor) -> torch.Tensor:
        Z, Y, X = self.mats.material_id.shape
        edep = torch.zeros((Z, Y, X), device=self.device, dtype=torch.float32)
        if alpha_local_edep is not None:
            edep += alpha_local_edep.to(device=self.device, dtype=torch.float32)

        def _deposit(queue):
            if queue is None:
                return
            if queue["E_MeV"].numel() == 0:
                return

            pos_cm = queue["pos_cm"].to(device=self.device, dtype=torch.float32)
            E = queue["E_MeV"].to(device=self.device, dtype=torch.float32)
            w = queue["w"].to(device=self.device, dtype=torch.float32)

            vx, vy, vz = self.voxel_size_cm
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
