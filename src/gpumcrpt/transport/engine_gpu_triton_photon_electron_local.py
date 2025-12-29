from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from gpumcrpt.transport.engine_base import BaseTransportEngine
from gpumcrpt.transport.triton_kernels.photon.flight import photon_woodcock_flight_kernel_philox
from gpumcrpt.transport.triton_kernels.photon.interactions import photon_interaction_kernel

from gpumcrpt.utils.constants import ELECTRON_REST_MASS_MEV


@dataclass
class PhotonElectronLocalStats:
    escaped_energy_MeV: float


class PhotonElectronLocalTransportEngine(BaseTransportEngine):
    """Photon-electron-local backend: photon transport + local charged-particle deposit (GPU/Triton).

    - Woodcock flight using tables.sigma_total + sigma_max
    - Interaction classification via per-material xs (PE/Compton/Rayleigh/Pair)
    - Photoelectric: deposit full photon energy locally and kill photon
    - Compton: update photon kinematics, deposit recoil electron energy locally
    - Rayleigh: update direction, keep energy
    - Pair: deposit full photon energy locally and kill (MVP)

    Notes:
      - This engine currently deposits charged secondaries locally; electron/positron transport is Milestone 3+.
      - Requires CUDA.
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
        # Initialize base class
        super().__init__(
            mats=mats,
            tables=tables,
            sim_config=sim_config,
            voxel_size_cm=voxel_size_cm,
            device=device,
        )
        
        self._last_stats = PhotonElectronLocalStats(escaped_energy_MeV=0.0)

    @property
    def last_stats(self) -> PhotonElectronLocalStats:
        return self._last_stats

    @torch.no_grad()
    def run_one_batch(self, primaries, alpha_local_edep: torch.Tensor) -> torch.Tensor:
        Z, Y, X = self.mats.material_id.shape
        edep = torch.zeros((Z, Y, X), device=self.device, dtype=torch.float32)
        if alpha_local_edep is not None:
            edep += alpha_local_edep.to(device=self.device, dtype=torch.float32)

        # Deposit non-photon primaries locally (Milestone 2 scope)
        for q in (primaries.electrons, primaries.positrons):
            if q is None or q["E_MeV"].numel() == 0:
                continue
            self._deposit_local(
                pos=q["pos_cm"].to(self.device, dtype=torch.float32),
                E=q["E_MeV"].to(self.device, dtype=torch.float32),
                w=q["w"].to(self.device, dtype=torch.float32),
                edep_flat=edep.view(-1),
                Z=Z, Y=Y, X=X,
                voxel_size_cm=self.voxel_size_cm,
            )

        # Photon queue
        pos = primaries.photons["pos_cm"].to(self.device, dtype=torch.float32).contiguous()
        direction = primaries.photons["dir"].to(self.device, dtype=torch.float32).contiguous()
        E = primaries.photons["E_MeV"].to(self.device, dtype=torch.float32).contiguous()
        w = primaries.photons["w"].to(self.device, dtype=torch.float32).contiguous()

        N = int(E.numel())
        if N == 0:
            self._last_stats = PhotonElectronLocalStats(escaped_energy_MeV=0.0)
            return edep

        # RNG seed for Triton kernels (Philox - stateless)
        seed = int(self.sim_config.get("seed", 0))

        # Preallocate ping-pong buffers
        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)

        ebin = torch.empty((N,), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((N,), device=self.device, dtype=torch.int8)
        real = torch.empty_like(alive)

        # Interaction outputs
        out_ph_pos = torch.empty_like(pos)
        out_ph_dir = torch.empty_like(direction)
        out_ph_E = torch.empty_like(E)
        out_ph_w = torch.empty_like(w)
        out_ph_ebin = torch.empty_like(ebin)
        
        out_e_pos = torch.empty_like(pos)
        out_e_dir = torch.empty_like(direction)
        out_e_E = torch.empty_like(E)
        out_e_w = torch.empty_like(w)
        
        out_po_pos = torch.empty_like(pos)
        out_po_dir = torch.empty_like(direction)
        out_po_E = torch.empty_like(E)
        out_po_w = torch.empty_like(w)

        escaped_energy = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        photon_cut_MeV = float(self.sim_config.get("cutoffs", {}).get("photon_keV", 3.0)) * 1e-3
        max_steps = int(self.sim_config.get("monte_carlo", {}).get("max_wavefront_iters", 512))

        # Accuracy: use precomputed Compton inverse-CDF (u -> cos(theta)).
        # The loader enforces the convention (cos_theta).
        allow_placeholders = bool(
            self.sim_config.get("monte_carlo", {}).get("triton", {}).get("allow_placeholder_samplers", False)
        )
        if self.tables.compton_inv_cdf is None:
            if not allow_placeholders:
                raise ValueError(
                    "Missing tables.compton_inv_cdf. For accurate Compton sampling, provide "
                    "'/samplers/photon/compton/inv_cdf' in the physics .h5 (convention='cos_theta'). "
                    "To run with the old isotropic placeholder, set monte_carlo.triton.allow_placeholder_samplers=true."
                )
            K = 256
            cos_grid = torch.linspace(-1.0, 1.0, K, device=self.device, dtype=torch.float32)
            compton_inv_cdf = cos_grid.repeat(int(self.tables.e_centers_MeV.numel()), 1).contiguous()
        else:
            compton_inv_cdf = self.tables.compton_inv_cdf.to(self.device, dtype=torch.float32).contiguous()
            if compton_inv_cdf.ndim != 2 or int(compton_inv_cdf.shape[0]) != int(self.tables.e_centers_MeV.numel()):
                raise ValueError(f"compton_inv_cdf must have shape [ECOUNT,K]; got {tuple(compton_inv_cdf.shape)}")
            K = int(compton_inv_cdf.shape[1])

        # Flattened geometry fields
        material_id_flat = self.mats.material_id.to(self.device, dtype=torch.int32).contiguous().view(-1)
        rho_flat = self.mats.rho.to(self.device, dtype=torch.float32).contiguous().view(-1)

        ECOUNT = int(self.tables.e_centers_MeV.numel())
        M = int(self.tables.ref_density_g_cm3.numel())
        vx, vy, vz = self.voxel_size_cm

        # Main loop
        for _ in range(max_steps):
            # Kill/Deposit below-cutoff photons
            below = (E > 0) & (E < photon_cut_MeV) & (w > 0)
            if torch.any(below):
                self._deposit_local(
                    pos=pos,
                    E=torch.where(below, E, torch.zeros_like(E)),
                    w=w,
                    edep_flat=edep.view(-1),
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )
                E = torch.where(below, torch.zeros_like(E), E)
                w = torch.where(below, torch.zeros_like(w), w)

            # Early exit
            if int((E > 0).sum().item()) == 0:
                break

            # Compute ebin from edges (bucketize)
            edges = self.tables.e_edges_MeV
            ebin = torch.bucketize(E, edges) - 1
            ebin = torch.clamp(ebin, 0, ECOUNT - 1).to(torch.int32)

            # Woodcock flight
            grid = (triton.cdiv(N, 256),)
            photon_woodcock_flight_kernel_philox[grid](
                pos, direction, E, w,
                seed,
                ebin,
                pos2, dir2, E2, w2,
                ebin2,
                alive, real,
                material_id_flat,
                rho_flat,
                self.tables.sigma_total,
                self.tables.sigma_max,
                self.tables.ref_density_g_cm3,
                Z=Z, Y=Y, X=X,
                M=M, ECOUNT=ECOUNT,
                N=N,
                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            )

            # Escaped photons: alive==0 after flight
            escaped_mask = (alive == 0) & (E2 > 0) & (w2 > 0)
            if torch.any(escaped_mask):
                escaped_energy = escaped_energy + (E2[escaped_mask] * w2[escaped_mask]).sum(dtype=torch.float32)
                E2 = torch.where(escaped_mask, torch.zeros_like(E2), E2)
                w2 = torch.where(escaped_mask, torch.zeros_like(w2), w2)

            # Use fused photon interaction kernel to combine classification and interaction
            photon_interaction_kernel[grid](
                real,
                pos2, dir2, E2, w2, ebin2,
                material_id_flat, rho_flat, self.tables.ref_density_g_cm3,
                self.tables.sigma_photo, self.tables.sigma_compton, self.tables.sigma_rayleigh,
                self.tables.sigma_pair,
                compton_inv_cdf, K,
                seed,
                # outputs:
                out_ph_pos, out_ph_dir, out_ph_E, out_ph_w, out_ph_ebin,
                out_e_pos, out_e_dir, out_e_E, out_e_w,
                out_po_pos, out_po_dir, out_po_E, out_po_w,
                edep.view(-1),
                Z=Z, Y=Y, X=X,
                M=M, ECOUNT=ECOUNT,
                N=N,
                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            )

            # Photon-electron-local mode: deposit charged-secondary energy locally.
            # The fused kernel outputs secondaries but does not transport them here.
            pair_mask = (out_ph_E == 0) & (out_e_E > 0) & (out_po_E > 0) & (out_ph_w > 0)
            if torch.any(pair_mask):
                pair_edep = out_e_E + out_po_E + (2.0 * ELECTRON_REST_MASS_MEV)
                self._deposit_local(
                    pos=out_ph_pos,
                    E=torch.where(pair_mask, pair_edep, torch.zeros_like(pair_edep)),
                    w=out_ph_w,
                    edep_flat=edep.view(-1),
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )

            e_mask = (out_e_E > 0) & (out_ph_w > 0) & (~pair_mask)
            if torch.any(e_mask):
                self._deposit_local(
                    pos=out_ph_pos,
                    E=torch.where(e_mask, out_e_E, torch.zeros_like(out_e_E)),
                    w=out_ph_w,
                    edep_flat=edep.view(-1),
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )

            # Update photon state after interaction
            pos2 = out_ph_pos
            dir2 = out_ph_dir
            E2 = out_ph_E
            w2 = out_ph_w
            ebin2 = out_ph_ebin

            # Ping-pong
            pos, direction, E, w, ebin = (
                pos2, dir2, E2, w2, ebin2
            )

        self._last_stats = PhotonElectronLocalStats(escaped_energy_MeV=float(escaped_energy.detach().cpu().item()))
        return edep



