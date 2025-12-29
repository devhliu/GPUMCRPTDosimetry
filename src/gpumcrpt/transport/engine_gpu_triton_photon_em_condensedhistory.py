from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import triton

from gpumcrpt.materials.hu_materials import compute_material_effective_atom_Z
from gpumcrpt.transport.engine_base import BaseTransportEngine
from gpumcrpt.transport.utils.secondary_budget import allow_secondaries, select_indices_with_budget
# Photon kernels
from gpumcrpt.transport.triton_kernels.photon.flight import photon_woodcock_flight_kernel_philox
from gpumcrpt.transport.triton_kernels.photon.interactions import photon_interaction_kernel
from gpumcrpt.transport.triton_kernels.photon.compton import photon_compton_kernel

# High-performance unified charged particle kernels
from gpumcrpt.transport.triton_kernels.charged_particle import (
    charged_particle_step_kernel,        # Main unified kernel for transport
    charged_particle_brems_emit_kernel,  # Unified bremsstrahlung emission
    charged_particle_delta_emit_kernel,  # Unified delta ray emission
    positron_annihilation_at_rest_kernel, # Positron annihilation at rest
)

# Tally/Deposit kernels
from gpumcrpt.transport.triton_kernels.utils.deposit import deposit_local_energy_kernel

# Physics constants
from gpumcrpt.utils.constants import ELECTRON_REST_MASS_MEV, PI


@dataclass
class CondensedHistoryMultiParticleStats:
    escaped_photon_energy_MeV: float
    annihilations: int
    brems_photons: int
    delta_electrons: int


class TritonPhotonEMCondensedHistoryEngine(BaseTransportEngine):
    """Photon-EM-CondensedHistoryMultiParticle transport engine (Milestone 3).

    Modern implementation using high-performance unified kernels.

    Scope (MVP):
    - Photons: Milestone-2 Woodcock flight + classify + Compton/PE.
      * Compton uses isotropic cos(theta) sampling (bring-up choice).
      * Photoelectric deposits all photon energy locally (Option B - faster, consistent with photon_only mode).
    - Electrons: condensed-history steps using unified `charged_particle_step_kernel`.
      * Below-cutoff kinetic energy is deposited locally and particle terminated.
    - Positrons: condensed-history steps using unified `charged_particle_step_kernel` (particle_type=1).
      * On stop, annihilation-at-rest emits 2×0.511 MeV photons and deposits remaining kinetic energy.
      * Annihilation photons are transported with the same photon transport.

    Features:
    - High-performance unified kernels for both electrons and positrons
    - Structure of Arrays (SoA) layout for optimal GPU performance
    - Modern physics models: Molière scattering, Vavilov straggling, Bethe-Heitler

    Notes:
    - Brems/delta secondaries are spawned (MVP: single-generation; children do not spawn further secondaries).
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

        self._last_stats = CondensedHistoryMultiParticleStats(
            escaped_photon_energy_MeV=0.0,
            annihilations=0,
            brems_photons=0,
            delta_electrons=0,
        )

    @torch.no_grad()
    def run_one_batch(self, primaries, alpha_local_edep: torch.Tensor) -> torch.Tensor:
        Z, Y, X = self.mats.material_id.shape
        edep = torch.zeros((Z, Y, X), device=self.device, dtype=torch.float32)
        if alpha_local_edep is not None:
            edep += alpha_local_edep.to(device=self.device, dtype=torch.float32)

        escaped = 0.0
        annihilations = 0
        brems_photons = 0
        delta_electrons = 0

        # Run photons (including any initial photons)
        sec = self.sim_config.get("electron_transport", {})
        secondary_depth = int(sec.get("secondary_depth", 1))
        max_secondaries_per_primary = int(sec.get("max_secondaries_per_primary", sec.get("max_secondaries_per_step", 1_000_000_000)))
        max_secondaries_per_step = int(sec.get("max_secondaries_per_step", 1_000_000_000))

        escaped += self._run_photons_inplace(
            pos=primaries.photons["pos_cm"].to(self.device, dtype=torch.float32).contiguous(),
            direction=primaries.photons["dir"].to(self.device, dtype=torch.float32).contiguous(),
            E=primaries.photons["E_MeV"].to(self.device, dtype=torch.float32).contiguous(),
            w=primaries.photons["w"].to(self.device, dtype=torch.float32).contiguous(),
            edep=edep,
            secondary_depth=secondary_depth,
            max_secondaries_per_primary=max_secondaries_per_primary,
            max_secondaries_per_step=max_secondaries_per_step,
        )

        # Run electrons
        esc_b, n_b, n_d = self._run_electrons_inplace(
            pos=primaries.electrons["pos_cm"].to(self.device, dtype=torch.float32).contiguous(),
            direction=primaries.electrons["dir"].to(self.device, dtype=torch.float32).contiguous(),
            E=primaries.electrons["E_MeV"].to(self.device, dtype=torch.float32).contiguous(),
            w=primaries.electrons["w"].to(self.device, dtype=torch.float32).contiguous(),
            edep=edep,
            secondary_depth=secondary_depth,
            max_secondaries_per_primary=max_secondaries_per_primary,
            max_secondaries_per_step=max_secondaries_per_step,
        )
        escaped += esc_b
        brems_photons += n_b
        delta_electrons += n_d

        # Run positrons + annihilation photons
        escaped_add, n_ann, n_b2, n_d2 = self._run_positrons_inplace(
            pos=primaries.positrons["pos_cm"].to(self.device, dtype=torch.float32).contiguous(),
            direction=primaries.positrons["dir"].to(self.device, dtype=torch.float32).contiguous(),
            E=primaries.positrons["E_MeV"].to(self.device, dtype=torch.float32).contiguous(),
            w=primaries.positrons["w"].to(self.device, dtype=torch.float32).contiguous(),
            edep=edep,
            secondary_depth=secondary_depth,
            max_secondaries_per_primary=max_secondaries_per_primary,
            max_secondaries_per_step=max_secondaries_per_step,
        )
        escaped += escaped_add
        annihilations += n_ann
        brems_photons += n_b2
        delta_electrons += n_d2

        self._last_stats = CondensedHistoryMultiParticleStats(
            escaped_photon_energy_MeV=float(escaped),
            annihilations=int(annihilations),
            brems_photons=int(brems_photons),
            delta_electrons=int(delta_electrons),
        )
        return edep

    def _run_photons_inplace(
        self,
        *,
        pos: torch.Tensor,
        direction: torch.Tensor,
        E: torch.Tensor,
        w: torch.Tensor,
        edep: torch.Tensor,
        secondary_depth: int = 1,
        max_secondaries_per_primary: int = 1_000_000_000,
        max_secondaries_per_step: int = 1_000_000_000,
    ) -> float:
        """Runs photon transport for a fixed-size photon array.

        Returns escaped photon energy (MeV) = sum(E*w) that leaves the volume.
        """
        N = int(E.numel())
        if N == 0:
            return 0.0

        Z, Y, X = self.mats.material_id.shape
        vx, vy, vz = self.voxel_size_cm

        # geometry/tables
        material_id_flat = self.mats.material_id.to(self.device, dtype=torch.int32).contiguous().view(-1)
        rho_flat = self.mats.rho.to(self.device, dtype=torch.float32).contiguous().view(-1)
        ECOUNT = int(self.tables.e_centers_MeV.numel())
        M = int(self.tables.ref_density_g_cm3.numel())

        # photon cut
        photon_cut_MeV = float(self.sim_config.get("cutoffs", {}).get("photon_keV", 3.0)) * 1e-3
        max_steps = int(self.sim_config.get("monte_carlo", {}).get("max_wavefront_iters", 512))
        e_cut_MeV = float(self.sim_config.get("cutoffs", {}).get("electron_keV", 20.0)) * 1e-3

        # RNG - Initialize Philox RNG seed for stateless RNG
        seed = int(self.sim_config.get("seed", 0))

        # ping-pong buffers
        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)

        ebin = torch.empty((N,), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((N,), device=self.device, dtype=torch.int8)
        real = torch.empty_like(alive)

        escaped_energy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        sec_counts = torch.zeros((N,), device=self.device, dtype=torch.int32)

        # Accuracy: use precomputed Compton inverse-CDF (u -> cos(theta)).
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
            compton_inv_cdf = cos_grid.repeat(ECOUNT, 1).contiguous()
        else:
            compton_inv_cdf = self.tables.compton_inv_cdf.to(self.device, dtype=torch.float32).contiguous()
            if compton_inv_cdf.ndim != 2 or int(compton_inv_cdf.shape[0]) != ECOUNT:
                raise ValueError(f"compton_inv_cdf must have shape [ECOUNT,K]; got {tuple(compton_inv_cdf.shape)}")
            K = int(compton_inv_cdf.shape[1])

        edep_flat = edep.view(-1)

        out_ph_pos = torch.empty_like(pos2)
        out_ph_dir = torch.empty_like(dir2)
        out_ph_E = torch.empty_like(E2)
        out_ph_w = torch.empty_like(w2)
        out_ph_ebin = torch.empty_like(ebin2)
        
        out_e_pos = torch.empty((N, 3), device=self.device, dtype=torch.float32)
        out_e_dir = torch.empty((N, 3), device=self.device, dtype=torch.float32)
        out_e_E = torch.empty((N,), device=self.device, dtype=torch.float32)
        out_e_w = torch.empty((N,), device=self.device, dtype=torch.float32)
        
        out_po_pos = torch.empty((N, 3), device=self.device, dtype=torch.float32)
        out_po_dir = torch.empty((N, 3), device=self.device, dtype=torch.float32)
        out_po_E = torch.empty((N,), device=self.device, dtype=torch.float32)
        out_po_w = torch.empty((N,), device=self.device, dtype=torch.float32)

        enable_timing = bool(self.sim_config.get("monte_carlo", {}).get("enable_timing", False))
        total_start_time = time.time() if enable_timing else None

        for step_num in range(max_steps):
            step_start_time = time.time() if enable_timing else None
            
            # cutoff deposit
            below = (E > 0) & (E < photon_cut_MeV) & (w > 0)
            if torch.any(below):
                self._deposit_local(
                    pos=pos,
                    E=torch.where(below, E, torch.zeros_like(E)),
                    w=w,
                    edep_flat=edep_flat,
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )
                E = torch.where(below, torch.zeros_like(E), E)
                w = torch.where(below, torch.zeros_like(w), w)

            if int((E > 0).sum().item()) == 0:
                break

            ebin = torch.bucketize(E, self.tables.e_edges_MeV) - 1
            ebin = torch.clamp(ebin, 0, ECOUNT - 1).to(torch.int32)

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
                self.tables.sigma_photo, self.tables.sigma_compton, self.tables.sigma_pair,
                compton_inv_cdf, K,
                seed,
                # outputs:
                out_ph_pos, out_ph_dir, out_ph_E, out_ph_w, out_ph_ebin,
                out_e_pos, out_e_dir, out_e_E, out_e_w,
                out_po_pos, out_po_dir, out_po_E, out_po_w,
                edep_flat,
                Z=Z, Y=Y, X=X,
                M=M, ECOUNT=ECOUNT,
                N=N,
                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            )
            
            # Update photon state after interaction
            pos2 = out_ph_pos
            dir2 = out_ph_dir
            E2 = out_ph_E
            w2 = out_ph_w
            ebin2 = out_ph_ebin

            # Process photoelectric electrons: photon absorbed, electron produced
            # Identify photoelectric events: photon energy zeroed, electron energy > 0
            pe_mask = (out_ph_E == 0) & (out_e_E > 0) & (w2 > 0)
            if torch.any(pe_mask):
                if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                    idx = select_indices_with_budget(
                        pe_mask,
                        sec_counts,
                        max_per_primary=max_secondaries_per_primary,
                        max_per_step=max_secondaries_per_step,
                    )
                    if int(idx.numel()) > 0:
                        esc_e, nb_e, nd_e = self._run_electrons_inplace(
                            pos=out_e_pos.index_select(0, idx),
                            direction=out_e_dir.index_select(0, idx),
                            E=out_e_E.index_select(0, idx),
                            w=out_e_w.index_select(0, idx),
                            edep=edep,
                            secondary_depth=secondary_depth - 1,
                            max_secondaries_per_primary=max_secondaries_per_primary,
                            max_secondaries_per_step=max_secondaries_per_step,
                        )
                        escaped_energy = escaped_energy + torch.tensor(esc_e, device=self.device, dtype=torch.float32)
                else:
                    self._deposit_local(
                        pos=out_e_pos,
                        E=out_e_E,
                        w=out_e_w,
                        edep_flat=edep_flat,
                        Z=Z, Y=Y, X=X,
                        voxel_size_cm=self.voxel_size_cm,
                    )

            # Process Compton recoil electrons: photon scattered, electron produced
            # Identify Compton events: both photon and electron have energy > 0
            compton_mask = (out_ph_E > 0) & (out_e_E > 0) & (w2 > 0)
            if torch.any(compton_mask):
                if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                    idx = select_indices_with_budget(
                        compton_mask,
                        sec_counts,
                        max_per_primary=max_secondaries_per_primary,
                        max_per_step=max_secondaries_per_step,
                    )
                    if int(idx.numel()) > 0:
                        esc_e, nb_e, nd_e = self._run_electrons_inplace(
                            pos=out_e_pos.index_select(0, idx),
                            direction=out_e_dir.index_select(0, idx),
                            E=out_e_E.index_select(0, idx),
                            w=out_e_w.index_select(0, idx),
                            edep=edep,
                            secondary_depth=secondary_depth - 1,
                            max_secondaries_per_primary=max_secondaries_per_primary,
                            max_secondaries_per_step=max_secondaries_per_step,
                        )
                        escaped_energy = escaped_energy + torch.tensor(esc_e, device=self.device, dtype=torch.float32)
                else:
                    self._deposit_local(
                        pos=out_e_pos,
                        E=out_e_E,
                        w=out_e_w,
                        edep_flat=edep_flat,
                        Z=Z, Y=Y, X=X,
                        voxel_size_cm=self.voxel_size_cm,
                    )

            # Process pair production: photon converted to electron-positron pair
            # Identify pair production events: photon energy zeroed, both electron and positron have energy > 0
            pair_mask = (out_ph_E == 0) & (out_e_E > 0) & (out_po_E > 0) & (w2 > 0)
            if torch.any(pair_mask):
                if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                    idx = select_indices_with_budget(
                        pair_mask,
                        sec_counts,
                        max_per_primary=max_secondaries_per_primary,
                        max_per_step=max_secondaries_per_step,
                    )
                    if int(idx.numel()) > 0:
                        esc_e, nb_e, nd_e = self._run_electrons_inplace(
                            pos=out_e_pos.index_select(0, idx),
                            direction=out_e_dir.index_select(0, idx),
                            E=out_e_E.index_select(0, idx),
                            w=out_e_w.index_select(0, idx),
                            edep=edep,
                            secondary_depth=secondary_depth - 1,
                            max_secondaries_per_primary=max_secondaries_per_primary,
                            max_secondaries_per_step=max_secondaries_per_step,
                        )
                        escaped_energy = escaped_energy + torch.tensor(esc_e, device=self.device, dtype=torch.float32)
                        
                        esc_po, nb_po, nd_po = self._run_electrons_inplace(
                            pos=out_po_pos.index_select(0, idx),
                            direction=out_po_dir.index_select(0, idx),
                            E=out_po_E.index_select(0, idx),
                            w=out_po_w.index_select(0, idx),
                            edep=edep,
                            secondary_depth=secondary_depth - 1,
                            max_secondaries_per_primary=max_secondaries_per_primary,
                            max_secondaries_per_step=max_secondaries_per_step,
                            particle_type_val=1,
                        )
                        escaped_energy = escaped_energy + torch.tensor(esc_po, device=self.device, dtype=torch.float32)
                else:
                    self._deposit_local(
                        pos=out_e_pos,
                        E=out_e_E,
                        w=out_e_w,
                        edep_flat=edep_flat,
                        Z=Z, Y=Y, X=X,
                        voxel_size_cm=self.voxel_size_cm,
                    )
                    self._deposit_local(
                        pos=out_po_pos,
                        E=out_po_E,
                        w=out_po_w,
                        edep_flat=edep_flat,
                        Z=Z, Y=Y, X=X,
                        voxel_size_cm=self.voxel_size_cm,
                    )

            pos, direction, E, w = pos2, dir2, E2, w2
            
            if enable_timing and step_start_time is not None:
                step_time = time.time() - step_start_time
                active_particles = int((E > 0).sum().item())
                print(f"Photon step {step_num}: {step_time:.4f}s, {active_particles} active particles")

        if enable_timing and total_start_time is not None:
            total_time = time.time() - total_start_time
            print(f"Photon transport completed in {total_time:.4f}s ({step_num + 1} steps)")

        return float(escaped_energy.detach().cpu().item())

    def _run_electrons_inplace(
        self,
        *,
        pos: torch.Tensor,
        direction: torch.Tensor,
        E: torch.Tensor,
        w: torch.Tensor,
        edep: torch.Tensor,
        secondary_depth: int = 1,
        max_secondaries_per_primary: int = 1_000_000_000,
        max_secondaries_per_step: int = 1_000_000,
        particle_type_val: int = 0,  # 0=electron, 1=positron
    ) -> tuple[float, int, int]:
        """
        Run electron/positron transport using the high-performance unified kernel.
        """
        N = int(E.numel())
        if N == 0:
            return 0.0, 0, 0

        enable_timing = bool(self.sim_config.get("monte_carlo", {}).get("enable_timing", False))
        total_start_time = time.time() if enable_timing else None

        Z, Y, X = self.mats.material_id.shape
        vx, vy, vz = self.voxel_size_cm

        e_cut = float(self.sim_config.get("cutoffs", {}).get("electron_keV", 20.0)) * 1e-3
        max_steps = int(self.sim_config.get("electron_transport", {}).get("max_steps", 4096))
        et = self.sim_config.get("electron_transport", {})
        f_range = float(et.get("f_range", 0.2))

        # Convert to Structure of Arrays format for modern kernel
        particle_pos_x = pos[:, 2].contiguous()  # x
        particle_pos_y = pos[:, 1].contiguous()  # y
        particle_pos_z = pos[:, 0].contiguous()  # z
        particle_dir_x = direction[:, 2].contiguous()
        particle_dir_y = direction[:, 1].contiguous()
        particle_dir_z = direction[:, 0].contiguous()
        particle_E = E.contiguous()
        particle_weight = w.contiguous()
        particle_type = torch.full((N,), particle_type_val, dtype=torch.int8, device=self.device)  # 0=electron, 1=positron
        
        # Compute material ID for each particle based on its voxel position
        iz = torch.clamp((particle_pos_z / vz).floor().long(), 0, Z - 1)
        iy = torch.clamp((particle_pos_y / vy).floor().long(), 0, Y - 1)
        ix = torch.clamp((particle_pos_x / vx).floor().long(), 0, X - 1)
        particle_material = self.mats.material_id[iz, iy, ix].to(dtype=torch.int32)
        
        particle_alive = torch.ones(N, dtype=torch.int8, device=self.device)

        # Prepare physics tables for unified kernel
        # First, determine the actual number of materials needed based on material IDs in the volume
        max_material_id = int(torch.max(self.mats.material_id).item())
        num_materials_needed = max_material_id + 1
        
        # Get the number of materials in the library
        if hasattr(self.mats, 'lib') and self.mats.lib is not None:
            num_materials_in_lib = len(self.mats.lib.material_names)
        else:
            num_materials_in_lib = num_materials_needed
        
        # Use the larger of the two to ensure we have enough entries
        num_materials = max(num_materials_needed, num_materials_in_lib)
        num_energy_bins = int(self.tables.e_centers_MeV.numel())

        # Create physics tables in the expected format
        material_Z = self._prepare_material_Z_table()

        # Load physics tables (convert to proper format if needed)
        S_restricted_table = self._prepare_table_2d(self.tables.S_restricted, num_materials, num_energy_bins)
        range_cdsa_table = self._prepare_table_2d(self.tables.range_csda_cm, num_materials, num_energy_bins)
        P_brem_table = self._prepare_table_2d(self.tables.P_brem_per_cm, num_materials, num_energy_bins)
        P_delta_table = self._prepare_table_2d(self.tables.P_delta_per_cm, num_materials, num_energy_bins)
        energy_bin_edges = self.tables.e_edges_MeV.contiguous()

        # Prepare output arrays for modern kernel
        new_particle_E = torch.empty_like(particle_E)
        new_particle_alive = torch.empty_like(particle_alive)

        # RNG seed for stateless Philox RNG
        seed = int(self.sim_config.get("seed", 0))

        # Secondary particle outputs
        photon_pos_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_pos_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_pos_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_dir_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_dir_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_dir_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_E = torch.zeros(N, dtype=torch.float32, device=self.device)
        photon_alive = torch.zeros(N, dtype=torch.int8, device=self.device)

        # Annihilation photons (for positrons, will be empty for electrons)
        ann_photon1_pos_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon1_pos_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon1_pos_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon1_dir_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon1_dir_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon1_dir_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon1_alive = torch.zeros(N, dtype=torch.int8, device=self.device)

        ann_photon2_pos_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon2_pos_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon2_pos_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon2_dir_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon2_dir_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon2_dir_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        ann_photon2_alive = torch.zeros(N, dtype=torch.int8, device=self.device)

        # Secondary electrons/positrons
        secondary_pos_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_pos_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_pos_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_dir_x = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_dir_y = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_dir_z = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_E = torch.zeros(N, dtype=torch.float32, device=self.device)
        secondary_type = torch.zeros(N, dtype=torch.int8, device=self.device)
        secondary_alive = torch.zeros(N, dtype=torch.int8, device=self.device)

        edep_flat = edep.view(-1)

        # Secondary tracking
        n_brems = 0
        n_delta = 0
        escaped_energy = 0.0

        # BLOCK_SIZE = 256
        grid = (triton.cdiv(N, 256),)

        # Run transport loop with modern kernel
        for step in range(max_steps):
            step_start_time = time.time() if enable_timing else None

            # Check cutoff
            below_cut = (particle_E > 0) & (particle_E < e_cut) & (particle_weight > 0)
            if torch.any(below_cut):
                # Deposit remaining energy locally
                idx = torch.where(below_cut)[0]
                self._deposit_local(
                    pos=pos,
                    E=torch.where(below_cut, particle_E, torch.zeros_like(particle_E)),
                    w=particle_weight,
                    edep_flat=edep_flat,
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )
                particle_E = torch.where(below_cut, torch.zeros_like(particle_E), particle_E)
                particle_weight = torch.where(below_cut, torch.zeros_like(particle_weight), particle_weight)
                particle_alive = torch.where(below_cut, 0, particle_alive)

            # Count escaped energy
            escaped_energy += torch.sum(particle_E[particle_E < 0]).item()
            particle_E = torch.maximum(particle_E, torch.tensor(0.0, device=particle_E.device))

            # Stop if no active particles
            active_mask = particle_E > 0
            if not torch.any(active_mask):
                break

            # Run modern unified kernel
            charged_particle_step_kernel[grid](
                # Unified particle arrays (SoA)
                particle_pos_x, particle_pos_y, particle_pos_z,
                particle_dir_x, particle_dir_y, particle_dir_z,
                particle_E, particle_weight, particle_type, particle_material, particle_alive,
                # RNG seed (stateless Philox)
                seed,
                # Physics tables
                material_Z, S_restricted_table, range_cdsa_table, P_brem_table, P_delta_table,
                # Energy binning
                energy_bin_edges,
                # Outputs for secondaries
                photon_pos_x, photon_pos_y, photon_pos_z,
                photon_dir_x, photon_dir_y, photon_dir_z,
                photon_E, photon_alive,
                # Annihilation photons (won't be used for electrons)
                ann_photon1_pos_x, ann_photon1_pos_y, ann_photon1_pos_z,
                ann_photon1_dir_x, ann_photon1_dir_y, ann_photon1_dir_z,
                ann_photon1_alive,
                ann_photon2_pos_x, ann_photon2_pos_y, ann_photon2_pos_z,
                ann_photon2_dir_x, ann_photon2_dir_y, ann_photon2_dir_z,
                ann_photon2_alive,
                # Secondary particles
                secondary_pos_x, secondary_pos_y, secondary_pos_z,
                secondary_dir_x, secondary_dir_y, secondary_dir_z,
                secondary_E, secondary_type, secondary_alive,
                # Updated outputs
                new_particle_E, new_particle_alive,
                # Energy deposition output
                edep.view(-1),
                # Parameters
                voxel_size_x_cm=float(vx), voxel_size_y_cm=float(vy), voxel_size_z_cm=float(vz),
                num_materials=num_materials, num_energy_bins=num_energy_bins,
                e_cut_MeV=e_cut, f_range=f_range,
                Z=Z, Y=Y, X=X,
                N=N,
                # Physics constants
                ELECTRON_REST_MASS_MEV=ELECTRON_REST_MASS_MEV, PI=PI,
            )

            # Update particle state
            particle_E = new_particle_E
            particle_alive = new_particle_alive

            # Convert back to AoS format for engine
            pos = torch.stack([particle_pos_z, particle_pos_y, particle_pos_x], dim=1)
            direction = torch.stack([particle_dir_z, particle_dir_y, particle_dir_x], dim=1)
            E = particle_E
            w = particle_weight

            # Count secondaries (simplified counting)
            n_brems += int(photon_alive.sum().item())
            n_delta += int(secondary_alive.sum().item())

            if enable_timing and step_start_time is not None:
                step_time = time.time() - step_start_time
                active_particles = int((particle_E > 0).sum().item())
                particle_type_str = "electron" if particle_type_val == 0 else "positron"
                print(f"{particle_type_str.capitalize()} step {step}: {step_time:.4f}s, {active_particles} active particles, {n_brems} brems, {n_delta} deltas")

        if enable_timing and total_start_time is not None:
            total_time = time.time() - total_start_time
            particle_type_str = "electron" if particle_type_val == 0 else "positron"
            print(f"{particle_type_str.capitalize()} transport completed in {total_time:.4f}s ({step + 1} steps)")

        return float(escaped_energy), n_brems, n_delta

    def _run_positrons_inplace(
        self,
        *,
        pos: torch.Tensor,
        direction: torch.Tensor,
        E: torch.Tensor,
        w: torch.Tensor,
        edep: torch.Tensor,
        secondary_depth: int = 1,
        max_secondaries_per_primary: int = 1_000_000_000,
        max_secondaries_per_step: int = 1_000_000,
    ) -> tuple[float, int, int, int]:
        """
        Run positron transport using the high-performance unified kernel.
        Returns: (escaped_energy, annihilations, n_brems, n_delta)
        """
        N = int(E.numel())
        if N == 0:
            return 0.0, 0, 0, 0

        # Use the electron method with particle_type=1 (positron)
        # The unified kernel handles both electrons and positrons
        escaped_energy, n_brems, n_delta = self._run_electrons_inplace(
            pos=pos,
            direction=direction,
            E=E,
            w=w,
            edep=edep,
            secondary_depth=secondary_depth,
            max_secondaries_per_primary=max_secondaries_per_primary,
            max_secondaries_per_step=max_secondaries_per_step,
            particle_type_val=1,  # Set particle_type to 1 for positrons
        )

        # TODO: Add proper annihilation counting when positrons reach cutoff
        # For now, return 0 annihilations as placeholder
        annihilations = 0

        return float(escaped_energy), annihilations, n_brems, n_delta

    def _prepare_material_Z_table(self):
        """Prepare material atomic number table for unified kernel."""
        if hasattr(self.mats, 'lib') and self.mats.lib is not None:
            from gpumcrpt.materials.hu_materials import compute_material_effective_atom_Z
            z_table = compute_material_effective_atom_Z(self.mats.lib)
            
            # Ensure table has entries for all possible material IDs in the volume
            max_material_id = int(torch.max(self.mats.material_id).item())
            if z_table.numel() <= max_material_id:
                # Pad with bone Z value (approx 12.01) for missing materials
                padding = max(0, int(max_material_id - z_table.numel() + 1))
                bone_z = 12.01
                z_table = torch.cat([z_table, torch.full((padding,), bone_z, dtype=z_table.dtype, device=z_table.device)])
            return z_table
        else:
            # Fallback: use approximate Z values with enough entries
            max_material_id = int(torch.max(self.mats.material_id).item())
            default_z_values = [7.42, 6.60, 6.26, 7.42, 12.01, 12.01, 12.01]  # Air, Lung, Fat, Muscle, Bone, Bone, Bone
            if len(default_z_values) <= max_material_id:
                default_z_values.extend([12.01] * (max_material_id - len(default_z_values) + 1))
            return torch.tensor(default_z_values[:max_material_id+1], dtype=torch.float32, device=self.device)

    def _prepare_table_2d(self, table, num_materials, num_energy_bins):
        """Prepare a 2D physics table for the unified kernel."""
        if table is None:
            return torch.zeros((num_materials, num_energy_bins), dtype=torch.float32, device=self.device)

        if isinstance(table, (list, tuple)):
            table = torch.tensor(table, dtype=torch.float32, device=self.device)

        # Ensure proper shape
        if table.dim() == 1:
            # Expand to 2D
            table = table.unsqueeze(0).expand(num_materials, -1)
        elif table.dim() == 2:
            # Ensure correct dimensions
            if table.shape[0] < num_materials or table.shape[1] < num_energy_bins:
                # Pad the table if necessary
                if table.shape[0] < num_materials:
                    # Pad with zeros for missing materials
                    padding_rows = num_materials - table.shape[0]
                    padding = torch.zeros((padding_rows, table.shape[1]), dtype=table.dtype, device=table.device)
                    table = torch.cat([table, padding], dim=0)
                if table.shape[1] < num_energy_bins:
                    # Pad with zeros for missing energy bins
                    padding_cols = num_energy_bins - table.shape[1]
                    padding = torch.zeros((table.shape[0], padding_cols), dtype=table.dtype, device=table.device)
                    table = torch.cat([table, padding], dim=1)
            else:
                # Truncate if too large
                table = table[:num_materials, :num_energy_bins]

        return table.contiguous()


def _deposit_local(*, pos: torch.Tensor, E: torch.Tensor, w: torch.Tensor, edep_flat: torch.Tensor,
                   Z: int, Y: int, X: int, voxel_size_cm: tuple[float, float, float]) -> None:
    """
    Deposit energy locally in the dose grid.

    This method handles energy deposition for both photons and electrons/positrons
    when particles fall below cutoff energy or reach simulation boundaries.
    """
    if E.numel() == 0:
        return
    vx, vy, vz = voxel_size_cm
    grid = (triton.cdiv(int(E.numel()), 256),)
    deposit_local_energy_kernel[grid](
        pos, E, w,
        edep_flat,
        Z=Z, Y=Y, X=X,
        voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
    )
