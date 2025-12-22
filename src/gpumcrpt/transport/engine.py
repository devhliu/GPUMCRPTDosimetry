from __future__ import annotations

from dataclasses import dataclass

import torch

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics_tables.tables import PhysicsTables
from gpumcrpt.transport.cpu_oracle import CPUOracleConfig, run_cpu_oracle_transport
from gpumcrpt.transport.engine_gpu_triton_localdepositonly import LocalDepositOnlyTransportEngine
from gpumcrpt.transport.engine_gpu_triton_photon_only import PhotonOnlyTransportEngine
from gpumcrpt.transport.engine_gpu_triton_photon_em_condensedhistory import TritonPhotonEMCondensedHistoryEngine


@dataclass
class TransportEngine:
    mats: MaterialsVolume
    tables: PhysicsTables
    sim_config: dict
    voxel_size_cm: tuple[float, float, float]
    device: str = "cuda"

    def run_batches(self, primaries, alpha_local_edep: torch.Tensor, n_batches: int) -> torch.Tensor:
        Z, Y, X = self.mats.material_id.shape

        if self.device == "cpu":
            edep_batches = torch.zeros((n_batches, Z, Y, X), device="cpu", dtype=torch.float32)
            n_ph = int(primaries.photons["E_MeV"].shape[0])
            n_el = int(primaries.electrons["E_MeV"].shape[0])
            n_po = int(primaries.positrons["E_MeV"].shape[0])
            N = max(n_ph, n_el, n_po)
            chunk = (N + n_batches - 1) // n_batches if N > 0 else 0
            photon_cut = float(self.sim_config["cutoffs"]["photon_keV"]) * 1e-3
            electron_cut = float(self.sim_config["cutoffs"]["electron_keV"]) * 1e-3
            cfg = CPUOracleConfig(photon_cut_MeV=photon_cut, electron_cut_MeV=electron_cut, boundary_mode="absorbing")

            mats_cpu = MaterialsVolume(material_id=self.mats.material_id.detach().cpu(), rho=self.mats.rho.detach().cpu())
            tables_cpu = self.tables.as_cpu() if hasattr(self.tables, "as_cpu") else self.tables

            for b in range(n_batches):
                if chunk == 0:
                    break
                s = b * chunk
                t = min((b + 1) * chunk, N)
                if s >= t:
                    continue
                prim_b = type(primaries)(
                    photons={k: v[s : min(t, n_ph)] for k, v in primaries.photons.items()},
                    electrons={k: v[s : min(t, n_el)] for k, v in primaries.electrons.items()},
                    positrons={k: v[s : min(t, n_po)] for k, v in primaries.positrons.items()},
                )
                edep_batches[b] = run_cpu_oracle_transport(
                    primaries=prim_b,
                    alpha_local_edep_MeV=alpha_local_edep.detach().cpu(),
                    mats=mats_cpu,
                    tables=tables_cpu,
                    voxel_size_cm=self.voxel_size_cm,
                    cfg=cfg,
                    seed=int(self.sim_config.get("seed", 0)) + b,
                )
            return edep_batches

        edep_batches = torch.zeros((n_batches, Z, Y, X), device=self.device, dtype=torch.float32)

        triton_cfg = self.sim_config.get("monte_carlo", {}).get("triton", {})
        triton_engine = str(triton_cfg.get("engine", "mvp")).lower()

        if triton_engine == "em_condensed" or triton_engine == "photon-em-condensedhistorymultiparticle":
            tr_engine = TritonPhotonEMCondensedHistoryEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        elif triton_engine == "photon_only" or triton_engine == "photononly":
            tr_engine = PhotonOnlyTransportEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        elif triton_engine == "mvp" or triton_engine == "localdepositonly":
            # LocalDepositOnly engine: local-deposition transport (runnable end-to-end).
            tr_engine = LocalDepositOnlyTransportEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown monte_carlo.triton.engine={triton_engine!r} (expected 'mvp'/'localdepositonly', 'photon_only'/'photononly', or 'em_condensed'/'photon-em-condensedhistorymultiparticle')"
            )

        n_ph = int(primaries.photons["E_MeV"].shape[0])
        n_el = int(primaries.electrons["E_MeV"].shape[0])
        n_po = int(primaries.positrons["E_MeV"].shape[0])
        N = max(n_ph, n_el, n_po)
        chunk = (N + n_batches - 1) // n_batches if N > 0 else 0

        for b in range(n_batches):
            if chunk == 0:
                break
            s = b * chunk
            t = min((b + 1) * chunk, N)
            if s >= t:
                continue
            prim_b = type(primaries)(
                photons={k: v[s : min(t, n_ph)] for k, v in primaries.photons.items()},
                electrons={k: v[s : min(t, n_el)] for k, v in primaries.electrons.items()},
                positrons={k: v[s : min(t, n_po)] for k, v in primaries.positrons.items()},
            )
            edep_batches[b] = tr_engine.run_one_batch(prim_b, alpha_local_edep)

        return edep_batches