from __future__ import annotations

from dataclasses import dataclass

import torch

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics_tables.tables import PhysicsTables
from gpumcrpt.transport.triton_kernels.perf.optimization import GPUConfigOptimizer, create_soa_layout
from gpumcrpt.transport.triton_kernels.perf.monitor import PerformanceMonitor


@dataclass
class TransportEngine:
    mats: MaterialsVolume
    tables: PhysicsTables
    sim_config: dict
    voxel_size_cm: tuple[float, float, float]
    device: str = "cuda"
    
    def __post_init__(self):
        """Initialize optimization components."""
        if self.device == "cpu":
            raise NotImplementedError(
                "CPU transport is not supported. Use 'cuda' device with GPU transport engines."
            )

        device_obj = torch.device(self.device)
        self.performance_monitor = PerformanceMonitor(device_obj)
        self.config_optimizer = GPUConfigOptimizer(device_obj)

    def run_batches(self, primaries, alpha_local_edep: torch.Tensor, n_batches: int) -> torch.Tensor:
        Z, Y, X = self.mats.material_id.shape

        if self.device == "cpu":
            raise NotImplementedError(
                "CPU transport is not supported. Use 'cuda' device with GPU transport engines."
            )

        edep_batches = torch.zeros((n_batches, Z, Y, X), device=self.device, dtype=torch.float32)

        triton_cfg = self.sim_config.get("monte_carlo", {}).get("triton", {})
        triton_engine = str(triton_cfg.get("engine", "local_deposit")).lower()

        # Import locally to avoid circular imports
        from gpumcrpt.transport.engine_gpu_triton_localdeposit_only import LocalDepositOnlyTransportEngine
        from gpumcrpt.transport.engine_gpu_triton_photon_electron_local import PhotonElectronLocalTransportEngine
        from gpumcrpt.transport.engine_gpu_triton_photon_electron_condensed import TritonPhotonElectronCondensedEngine
        from gpumcrpt.transport.engine_gpu_triton_photon_em_energybucketed import TritonPhotonEMEnergyBucketedGraphsEngine

        if triton_engine in {"photon_electron_condensed"}:
            tr_engine = TritonPhotonElectronCondensedEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        elif triton_engine in {
            "em_energybucketed",
            "photon-em-energybucketed",
            "photon_em_energybucketed",
            "energy_bucketed",
            "energybucketed",
        }:
            tr_engine = TritonPhotonEMEnergyBucketedGraphsEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        elif triton_engine in {"photon_electron_local"}:
            tr_engine = PhotonElectronLocalTransportEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        elif triton_engine in {"localdepositonly", "local_deposit", "local-deposit"}:
            tr_engine = LocalDepositOnlyTransportEngine(
                mats=self.mats,
                tables=self.tables,
                sim_config=self.sim_config,
                voxel_size_cm=self.voxel_size_cm,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown monte_carlo.triton.engine={triton_engine!r} (expected "
                "'localdepositonly'/'local_deposit', "
                "'photon_electron_local', "
                "'photon_electron_condensed', "
                "or 'em_energybucketed'/'photon-em-energybucketed')"
            )

        n_ph = int(primaries.photons["E_MeV"].shape[0])
        n_el = int(primaries.electrons["E_MeV"].shape[0])
        n_po = int(primaries.positrons["E_MeV"].shape[0])
        N = max(n_ph, n_el, n_po)
        chunk = (N + n_batches - 1) // n_batches if N > 0 else 0

        def _slice_queue(queue, start, end):
            if queue is None or queue["E_MeV"].shape[0] == 0:
                return queue
            return {k: v[start:end] for k, v in queue.items()}

        for b in range(n_batches):
            if chunk == 0:
                break
            s = b * chunk
            t = min((b + 1) * chunk, N)
            if s >= t:
                continue
            # Prepare batch
            prim_b = type(primaries)(
                photons=self._prepare_primaries(_slice_queue(primaries.photons, s, min(t, n_ph))),
                electrons=self._prepare_primaries(_slice_queue(primaries.electrons, s, min(t, n_el))),
                positrons=self._prepare_primaries(_slice_queue(primaries.positrons, s, min(t, n_po))),
            )
            edep_batches[b] = tr_engine.run_one_batch(prim_b, alpha_local_edep)

        # Print performance report if monitoring is enabled
        if self.performance_monitor:
            self.performance_monitor.print_performance_report()

        return edep_batches

    def _prepare_primaries(self, primary_queue: dict) -> dict:
        """
        Prepare primary particles with SoA memory layout optimization.
        
        Philox RNG initialization is handled internally by the kernels,
        so we only need to apply memory layout optimizations here.
        """
        # Apply memory layout optimizations
        primary_queue = create_soa_layout(primary_queue)

        return primary_queue