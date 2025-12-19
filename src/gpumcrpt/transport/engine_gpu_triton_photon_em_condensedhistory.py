from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from gpumcrpt.materials.hu_materials import compute_material_effective_atom_Z
from gpumcrpt.transport.secondary_budget import allow_secondaries, select_indices_with_budget
from gpumcrpt.transport.triton.brems_delta import electron_brems_emit_kernel, electron_delta_emit_kernel
from gpumcrpt.transport.triton.compton import photon_compton_kernel
from gpumcrpt.transport.triton.edep_deposit import deposit_local_energy_kernel
from gpumcrpt.transport.triton.electron_step import electron_condensed_step_kernel
from gpumcrpt.transport.triton.pair import photon_pair_kernel
from gpumcrpt.transport.triton.photon_flight import photon_woodcock_flight_kernel
from gpumcrpt.transport.triton.photon_interactions import photon_classify_kernel
from gpumcrpt.transport.triton.photoelectric_with_vacancy import photon_photoelectric_with_vacancy_kernel
from gpumcrpt.transport.triton.positron import positron_annihilation_at_rest_kernel
from gpumcrpt.transport.triton.positron_step import positron_condensed_step_kernel
from gpumcrpt.transport.triton.rayleigh import photon_rayleigh_kernel
from gpumcrpt.physics.relaxation_tables import RelaxationTables


@dataclass
class CondensedHistoryMultiParticleStats:
    escaped_photon_energy_MeV: float
    annihilations: int
    brems_photons: int
    delta_electrons: int


class TritonPhotonEMCondensedHistoryEngine:
    """Photon-EM-CondensedHistoryMultiParticle transport engine (Milestone 3).

    Scope (MVP):
    - Photons: Milestone-2 Woodcock flight + classify + Compton/Rayleigh/PE.
      * Compton uses isotropic cos(theta) sampling (bring-up choice).
      * Photoelectric deposits photon energy locally (charged secondary transport from PE is future work).
    - Electrons: condensed-history steps using `electron_condensed_step_kernel`.
      * Below-cutoff kinetic energy is deposited locally and particle terminated.
    - Positrons: condensed-history steps using `positron_condensed_step_kernel`.
      * On stop, annihilation-at-rest emits 2×0.511 MeV photons and deposits remaining kinetic energy.
      * Annihilation photons are transported with the same photon transport.

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
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TritonPhotonEMCondensedHistoryEngine requires CUDA")

        self.mats = mats
        self.tables = tables
        self.sim_config = sim_config
        self.voxel_size_cm = voxel_size_cm
        self.device = device

        self._last_stats = CondensedHistoryMultiParticleStats(
            escaped_photon_energy_MeV=0.0,
            annihilations=0,
            brems_photons=0,
            delta_electrons=0,
        )

        self._relax_tables: RelaxationTables | None = None

        if hasattr(self.tables, "relax_shell_cdf") and hasattr(self.tables, "relax_E_bind_MeV"):
            if hasattr(self.tables, "relax_fluor_yield") and hasattr(self.tables, "relax_E_xray_MeV") and hasattr(self.tables, "relax_E_auger_MeV"):
                self._relax_tables = RelaxationTables(
                    relax_shell_cdf=self.tables.relax_shell_cdf,
                    relax_E_bind_MeV=self.tables.relax_E_bind_MeV,
                    relax_fluor_yield=self.tables.relax_fluor_yield,
                    relax_E_xray_MeV=self.tables.relax_E_xray_MeV,
                    relax_E_auger_MeV=self.tables.relax_E_auger_MeV,
                )
            else:
                M = int(self.tables.ref_density_g_cm3.numel())
                S = int(getattr(self.tables, "relax_shell_cdf").shape[1])
                base = RelaxationTables.dummy(self.device, M=M, S=S)
                self._relax_tables = RelaxationTables(
                    relax_shell_cdf=self.tables.relax_shell_cdf,
                    relax_E_bind_MeV=self.tables.relax_E_bind_MeV,
                    relax_fluor_yield=base.relax_fluor_yield,
                    relax_E_xray_MeV=base.relax_E_xray_MeV,
                    relax_E_auger_MeV=base.relax_E_auger_MeV,
                )
        else:
            pe = self.sim_config.get("photon_transport", {}).get("photoelectric", {})
            use_dummy = pe.get("use_dummy_relaxation_tables", True)
            if use_dummy:
                M = int(self.tables.ref_density_g_cm3.numel())
                S = int(pe.get("shells", 4))
                self._relax_tables = RelaxationTables.dummy(self.device, M=M, S=S)

    @property
    def last_stats(self) -> CondensedHistoryMultiParticleStats:
        return self._last_stats

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

    def _inv_cdf_or_uniform(self, inv_cdf: torch.Tensor | None, *, ECOUNT: int, max_efrac: float) -> tuple[torch.Tensor, int]:
        # Returns (inv_cdf, K).
        # For accuracy, missing samplers are treated as an error unless
        # monte_carlo.triton.allow_placeholder_samplers=true.
        if inv_cdf is not None:
            inv = inv_cdf.to(device=self.device, dtype=torch.float32).contiguous()
            if inv.ndim != 2 or int(inv.shape[0]) != ECOUNT:
                raise ValueError(f"inv_cdf must have shape [ECOUNT,K]; got {tuple(inv.shape)} (ECOUNT={ECOUNT})")
            return inv, int(inv.shape[1])

        allow_placeholders = bool(
            self.sim_config.get("monte_carlo", {}).get("triton", {}).get("allow_placeholder_samplers", False)
        )
        if not allow_placeholders:
            raise ValueError(
                "Missing required inverse-CDF sampler table in physics .h5. "
                "Provide the appropriate /samplers/.../inv_cdf_* dataset, or set "
                "monte_carlo.triton.allow_placeholder_samplers=true to use a uniform placeholder (not physically accurate)."
            )

        K = 256
        grid = torch.linspace(0.0, float(max_efrac), K, device=self.device, dtype=torch.float32)
        inv = grid.repeat(ECOUNT, 1).contiguous()
        return inv, K

    def _rng_i32(self, n: int) -> torch.Tensor:

        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.sim_config.get("seed", 0)))
        return torch.randint(1, 2**31 - 1, (n,), generator=g, device=self.device, dtype=torch.int32)

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

        # RNG
        rng = self._rng_i32(N)
        rng2 = torch.empty_like(rng)

        # ping-pong buffers
        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)

        ebin = torch.empty((N,), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((N,), device=self.device, dtype=torch.int8)
        real = torch.empty_like(alive)
        typ = torch.empty((N,), device=self.device, dtype=torch.int8)

        # scatter outputs
        scat_pos = torch.empty_like(pos)
        scat_dir = torch.empty_like(direction)
        scat_E = torch.empty_like(E)
        scat_w = torch.empty_like(w)
        scat_rng = torch.empty_like(rng)
        scat_ebin = torch.empty_like(ebin)

        e_pos = torch.empty_like(pos)
        e_dir = torch.empty_like(direction)
        e_E = torch.empty_like(E)
        e_w = torch.empty_like(w)

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

        for _ in range(max_steps):
            # cutoff deposit
            below = (E > 0) & (E < photon_cut_MeV) & (w > 0)
            if torch.any(below):
                _deposit_local(
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
            photon_woodcock_flight_kernel[grid](
                pos, direction, E, w, rng,
                ebin,
                pos2, dir2, E2, w2, rng2,
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

            # classify
            photon_classify_kernel[grid](
                real,
                pos2,
                E2,
                ebin2,
                rng2,
                material_id_flat,
                rho_flat,
                self.tables.ref_density_g_cm3,
                self.tables.sigma_photo,
                self.tables.sigma_compton,
                self.tables.sigma_rayleigh,
                self.tables.sigma_pair,
                typ,
                rng,
                Z=Z, Y=Y, X=X,
                M=M, ECOUNT=ECOUNT,
                BLOCK=256,
                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            )

            # photoelectric: deposit all photon energy locally
            pe = (typ == 1) & (E2 > 0) & (w2 > 0)
            if torch.any(pe):
                idx = select_indices_with_budget(
                    pe,
                    sec_counts,
                    max_per_primary=max_secondaries_per_primary,
                    max_per_step=max_secondaries_per_step,
                )

                if int(idx.numel()) > 0:
                    ns = int(idx.numel())
                    s_pos = pos2.index_select(0, idx)
                    s_dir = dir2.index_select(0, idx)
                    s_E = E2.index_select(0, idx)
                    s_w = w2.index_select(0, idx)
                    s_rng = rng2.index_select(0, idx)
                    s_ebin = ebin2.index_select(0, idx)

                    if self._relax_tables is None:
                        raise RuntimeError("Photoelectric was selected but relaxation tables were not available")

                    shell_cdf = self._relax_tables.relax_shell_cdf.to(self.device, dtype=torch.float32).contiguous()
                    E_bind = self._relax_tables.relax_E_bind_MeV.to(self.device, dtype=torch.float32).contiguous()
                    S = int(shell_cdf.shape[1])

                    e_pos_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    e_dir_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    e_E_out = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    e_w_out = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    out_rng = torch.empty((ns,), device=self.device, dtype=torch.int32)

                    v_pos_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    v_mat_out = torch.empty((ns,), device=self.device, dtype=torch.int32)
                    v_shell_out = torch.empty((ns,), device=self.device, dtype=torch.int8)
                    v_w_out = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    v_has = torch.empty((ns,), device=self.device, dtype=torch.int8)

                    g2 = (triton.cdiv(ns, 256),)
                    photon_photoelectric_with_vacancy_kernel[g2](
                        s_pos, s_dir, s_E, s_w, s_rng, s_ebin,
                        material_id_flat,
                        shell_cdf, E_bind,
                        M=M, S=S,
                        out_e_pos_ptr=e_pos_out,
                        out_e_dir_ptr=e_dir_out,
                        out_e_E_ptr=e_E_out,
                        out_e_w_ptr=e_w_out,
                        out_e_rng_ptr=out_rng,
                        out_vac_pos_ptr=v_pos_out,
                        out_vac_mat_ptr=v_mat_out,
                        out_vac_shell_ptr=v_shell_out,
                        out_vac_w_ptr=v_w_out,
                        out_has_vac_ptr=v_has,
                        edep_ptr=edep_flat,
                        N=ns,
                        Z=Z, Y=Y, X=X,
                        BLOCK=256,
                        voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
                        num_warps=4,
                    )

                    selected = torch.zeros_like(pe)
                    selected.index_fill_(0, idx, True)
                    pe_not = pe & (~selected)
                    if torch.any(pe_not):
                        _deposit_local(
                            pos=pos2,
                            E=torch.where(pe_not, E2, torch.zeros_like(E2)),
                            w=w2,
                            edep_flat=edep_flat,
                            Z=Z, Y=Y, X=X,
                            voxel_size_cm=self.voxel_size_cm,
                        )
                        E2 = torch.where(pe_not, torch.zeros_like(E2), E2)
                        w2 = torch.where(pe_not, torch.zeros_like(w2), w2)

                    E2.index_fill_(0, idx, 0.0)
                    w2.index_fill_(0, idx, 0.0)
                    rng2.index_copy_(0, idx, out_rng)

                    e_mask = (e_E_out > 0) & (e_w_out > 0)
                    if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                        if torch.any(e_mask):
                            esc_e, nb_e, nd_e = self._run_electrons_inplace(
                                pos=e_pos_out[e_mask],
                                direction=e_dir_out[e_mask],
                                E=e_E_out[e_mask],
                                w=e_w_out[e_mask],
                                edep=edep,
                                secondary_depth=secondary_depth - 1,
                                max_secondaries_per_primary=max_secondaries_per_primary,
                                max_secondaries_per_step=max_secondaries_per_step,
                            )
                            escaped_energy = escaped_energy + torch.tensor(esc_e, device=self.device, dtype=torch.float32)
                        if self._relax_tables is not None and torch.any(v_has.to(torch.bool)):
                            vmask = v_has.to(torch.bool)
                            v_pos = v_pos_out[vmask]
                            v_mat = v_mat_out[vmask].to(torch.int32)
                            v_shell = v_shell_out[vmask].to(torch.int8)
                            v_w = v_w_out[vmask]
                            v_rng = out_rng[vmask].contiguous()

                            from gpumcrpt.transport.triton.atomic_relaxation import atomic_relaxation_kernel

                            Srel = int(self._relax_tables.relax_fluor_yield.shape[1])
                            ph_pos_out = torch.empty((int(v_pos.shape[0]), 3), device=self.device, dtype=torch.float32)
                            ph_dir_out = torch.empty((int(v_pos.shape[0]), 3), device=self.device, dtype=torch.float32)
                            ph_E_out = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.float32)
                            ph_w_out = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.float32)
                            ph_rng_out = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.int32)
                            ph_has = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.int8)

                            a_pos_out = torch.empty((int(v_pos.shape[0]), 3), device=self.device, dtype=torch.float32)
                            a_dir_out = torch.empty((int(v_pos.shape[0]), 3), device=self.device, dtype=torch.float32)
                            a_E_out = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.float32)
                            a_w_out = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.float32)
                            a_rng_out = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.int32)
                            a_has = torch.empty((int(v_pos.shape[0]),), device=self.device, dtype=torch.int8)

                            g3 = (triton.cdiv(int(v_pos.shape[0]), 256),)
                            atomic_relaxation_kernel[g3](
                                v_pos, v_mat, v_shell, v_w, v_rng,
                                self._relax_tables.relax_fluor_yield.to(self.device, dtype=torch.float32).contiguous(),
                                self._relax_tables.relax_E_xray_MeV.to(self.device, dtype=torch.float32).contiguous(),
                                self._relax_tables.relax_E_auger_MeV.to(self.device, dtype=torch.float32).contiguous(),
                                M=M,
                                S=Srel,
                                photon_cut_MeV=float(photon_cut_MeV),
                                e_cut_MeV=float(e_cut_MeV),
                                out_ph_pos_ptr=ph_pos_out,
                                out_ph_dir_ptr=ph_dir_out,
                                out_ph_E_ptr=ph_E_out,
                                out_ph_w_ptr=ph_w_out,
                                out_ph_rng_ptr=ph_rng_out,
                                out_has_ph_ptr=ph_has,
                                out_e_pos_ptr=a_pos_out,
                                out_e_dir_ptr=a_dir_out,
                                out_e_E_ptr=a_E_out,
                                out_e_w_ptr=a_w_out,
                                out_e_rng_ptr=a_rng_out,
                                out_has_e_ptr=a_has,
                                edep_ptr=edep_flat,
                                material_id_ptr=material_id_flat,
                                Z=Z, Y=Y, X=X,
                                BLOCK=256,
                                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
                                num_warps=4,
                            )

                            if allow_secondaries(secondary_depth=secondary_depth - 1, max_per_primary=max_secondaries_per_primary):
                                if torch.any(ph_has.to(torch.bool)):
                                    escaped_energy = escaped_energy + torch.tensor(
                                        self._run_photons_inplace(
                                            pos=ph_pos_out[ph_has.to(torch.bool)],
                                            direction=ph_dir_out[ph_has.to(torch.bool)],
                                            E=ph_E_out[ph_has.to(torch.bool)],
                                            w=ph_w_out[ph_has.to(torch.bool)],
                                            edep=edep,
                                            secondary_depth=secondary_depth - 1,
                                            max_secondaries_per_primary=max_secondaries_per_primary,
                                            max_secondaries_per_step=max_secondaries_per_step,
                                        ),
                                        device=self.device,
                                        dtype=torch.float32,
                                    )
                                if torch.any(a_has.to(torch.bool)):
                                    esc2, nb2, nd2 = self._run_electrons_inplace(
                                        pos=a_pos_out[a_has.to(torch.bool)],
                                        direction=a_dir_out[a_has.to(torch.bool)],
                                        E=a_E_out[a_has.to(torch.bool)],
                                        w=a_w_out[a_has.to(torch.bool)],
                                        edep=edep,
                                        secondary_depth=secondary_depth - 1,
                                        max_secondaries_per_primary=max_secondaries_per_primary,
                                        max_secondaries_per_step=max_secondaries_per_step,
                                    )
                                    escaped_energy = escaped_energy + torch.tensor(esc2, device=self.device, dtype=torch.float32)
                    else:
                        if torch.any(e_mask):
                            _deposit_local(
                                pos=e_pos_out[e_mask],
                                E=e_E_out[e_mask],
                                w=e_w_out[e_mask],
                                edep_flat=edep_flat,
                                Z=Z, Y=Y, X=X,
                                voxel_size_cm=self.voxel_size_cm,
                            )
                else:
                    _deposit_local(
                        pos=pos2,
                        E=torch.where(pe, E2, torch.zeros_like(E2)),
                        w=w2,
                        edep_flat=edep_flat,
                        Z=Z, Y=Y, X=X,
                        voxel_size_cm=self.voxel_size_cm,
                    )
                    E2 = torch.where(pe, torch.zeros_like(E2), E2)
                    w2 = torch.where(pe, torch.zeros_like(w2), w2)

            pa = (typ == 4) & (E2 > 0) & (w2 > 0)
            if torch.any(pa) and allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                idx = select_indices_with_budget(
                    pa,
                    sec_counts,
                    max_per_primary=max_secondaries_per_primary,
                    max_per_step=max_secondaries_per_step,
                )
                if int(idx.numel()) > 0:
                    ns = int(idx.numel())
                    s_pos = pos2.index_select(0, idx)
                    s_dir = dir2.index_select(0, idx)
                    s_E = E2.index_select(0, idx)
                    s_w = w2.index_select(0, idx)
                    s_rng = rng2.index_select(0, idx)
                    s_ebin = ebin2.index_select(0, idx)

                    e_pos_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    e_dir_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    e_E_out = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    e_w_out = torch.empty((ns,), device=self.device, dtype=torch.float32)

                    p_pos_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    p_dir_out = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    p_E_out = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    p_w_out = torch.empty((ns,), device=self.device, dtype=torch.float32)

                    out_rng = torch.empty((ns,), device=self.device, dtype=torch.int32)

                    g2 = (triton.cdiv(ns, 256),)
                    photon_pair_kernel[g2](
                        s_pos, s_dir, s_E, s_w, s_rng, s_ebin,
                        e_pos_out, e_dir_out, e_E_out, e_w_out,
                        p_pos_out, p_dir_out, p_E_out, p_w_out,
                        out_rng,
                        ECOUNT=ECOUNT,
                        BLOCK=256,
                    )

                    E2.index_fill_(0, idx, 0.0)
                    w2.index_fill_(0, idx, 0.0)
                    rng2.index_copy_(0, idx, out_rng)

                    esc_e, nb_e, nd_e = self._run_electrons_inplace(
                        pos=e_pos_out,
                        direction=e_dir_out,
                        E=e_E_out,
                        w=e_w_out,
                        edep=edep,
                        secondary_depth=secondary_depth - 1,
                        max_secondaries_per_primary=max_secondaries_per_primary,
                        max_secondaries_per_step=max_secondaries_per_step,
                    )
                    escaped_energy = escaped_energy + torch.tensor(esc_e, device=self.device, dtype=torch.float32)

                    esc_p, ann_p, nb_p, nd_p = self._run_positrons_inplace(
                        pos=p_pos_out,
                        direction=p_dir_out,
                        E=p_E_out,
                        w=p_w_out,
                        edep=edep,
                        secondary_depth=secondary_depth - 1,
                        max_secondaries_per_primary=max_secondaries_per_primary,
                        max_secondaries_per_step=max_secondaries_per_step,
                    )
                    escaped_energy = escaped_energy + torch.tensor(esc_p, device=self.device, dtype=torch.float32)

            # Compton
            co = (typ == 2) & (E2 > 0) & (w2 > 0)
            if torch.any(co):
                E_in = torch.where(co, E2, torch.zeros_like(E2))
                w_in = torch.where(co, w2, torch.zeros_like(w2))

                photon_compton_kernel[grid](
                    pos2, dir2, E_in, w_in, rng, ebin2,
                    compton_inv_cdf, K,
                    scat_pos, scat_dir, scat_E, scat_w, scat_rng, scat_ebin,
                    e_pos, e_dir, e_E, e_w,
                    ECOUNT=ECOUNT,
                    BLOCK=256,
                )

                if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                    idx = select_indices_with_budget(
                        co,
                        sec_counts,
                        max_per_primary=max_secondaries_per_primary,
                        max_per_step=max_secondaries_per_step,
                    )
                    if int(idx.numel()) > 0:
                        esc_e, nb_e, nd_e = self._run_electrons_inplace(
                            pos=e_pos.index_select(0, idx),
                            direction=e_dir.index_select(0, idx),
                            E=e_E.index_select(0, idx),
                            w=e_w.index_select(0, idx),
                            edep=edep,
                            secondary_depth=secondary_depth - 1,
                            max_secondaries_per_primary=max_secondaries_per_primary,
                            max_secondaries_per_step=max_secondaries_per_step,
                        )
                        escaped_energy = escaped_energy + torch.tensor(esc_e, device=self.device, dtype=torch.float32)
                else:
                    _deposit_local(
                        pos=pos2,
                        E=e_E,
                        w=e_w,
                        edep_flat=edep_flat,
                        Z=Z, Y=Y, X=X,
                        voxel_size_cm=self.voxel_size_cm,
                    )

                co3 = co[:, None]
                pos2 = torch.where(co3, scat_pos, pos2)
                dir2 = torch.where(co3, scat_dir, dir2)
                E2 = torch.where(co, scat_E, E2)
                w2 = torch.where(co, scat_w, w2)
                rng = torch.where(co, scat_rng, rng)
                ebin2 = torch.where(co, scat_ebin, ebin2)

            # Rayleigh
            ra = (typ == 3) & (E2 > 0) & (w2 > 0)
            if torch.any(ra):
                E_in = torch.where(ra, E2, torch.zeros_like(E2))
                w_in = torch.where(ra, w2, torch.zeros_like(w2))

                photon_rayleigh_kernel[grid](
                    pos2, dir2, E_in, w_in, rng, ebin2,
                    scat_pos, scat_dir, scat_E, scat_w, scat_rng, scat_ebin,
                    ECOUNT=ECOUNT,
                    BLOCK=256,
                )

                ra3 = ra[:, None]
                pos2 = torch.where(ra3, scat_pos, pos2)
                dir2 = torch.where(ra3, scat_dir, dir2)
                E2 = torch.where(ra, scat_E, E2)
                w2 = torch.where(ra, scat_w, w2)
                rng = torch.where(ra, scat_rng, rng)
                ebin2 = torch.where(ra, scat_ebin, ebin2)

            pos, direction, E, w, rng = pos2, dir2, E2, w2, rng

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
        max_secondaries_per_step: int = 1_000_000_000,
    ) -> tuple[float, int, int]:
        N = int(E.numel())
        if N == 0:
            return 0.0, 0, 0

        Z, Y, X = self.mats.material_id.shape
        vx, vy, vz = self.voxel_size_cm

        e_cut = float(self.sim_config.get("cutoffs", {}).get("electron_keV", 20.0)) * 1e-3
        max_steps = int(self.sim_config.get("electron_transport", {}).get("max_steps", 4096))
        et = self.sim_config.get("electron_transport", {})
        f_vox = float(et.get("f_voxel", 0.3))
        f_range = float(et.get("f_range", 0.2))
        max_dE_frac = float(et.get("max_dE_frac", 0.2))

        material_id = self.mats.material_id.to(self.device, dtype=torch.int32).contiguous()
        rho = self.mats.rho.to(self.device, dtype=torch.float32).contiguous()

        M = int(self.tables.ref_density_g_cm3.numel())
        ECOUNT = int(self.tables.e_centers_MeV.numel())

        P_brem = self.tables.P_brem_per_cm if self.tables.P_brem_per_cm is not None else torch.zeros_like(self.tables.S_restricted)
        P_delta = self.tables.P_delta_per_cm if self.tables.P_delta_per_cm is not None else torch.zeros_like(self.tables.S_restricted)

        brem_inv_cdf, K_brem = self._inv_cdf_or_uniform(self.tables.brem_inv_cdf_Efrac, ECOUNT=ECOUNT, max_efrac=0.3)
        delta_inv_cdf, K_delta = self._inv_cdf_or_uniform(self.tables.delta_inv_cdf_Efrac, ECOUNT=ECOUNT, max_efrac=0.5)

        rng = self._rng_i32(N)
        rng2 = torch.empty_like(rng)

        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)

        ebin = torch.empty((N,), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((N,), device=self.device, dtype=torch.int8)
        emit_b = torch.empty_like(alive)
        emit_d = torch.empty_like(alive)

        edep_flat = edep.view(-1)
        grid = (triton.cdiv(N, 256),)

        escaped_brems = 0.0
        n_brems = 0
        n_delta = 0

        sec_counts = torch.zeros((N,), device=self.device, dtype=torch.int32)

        for _ in range(max_steps):
            below = (E > 0) & (E < e_cut) & (w > 0)
            if torch.any(below):
                _deposit_local(
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

        # Calculate effective atomic numbers for materials
        if self.mats.lib is not None:
            Z_material = compute_material_effective_atom_Z(self.mats.lib)
        else:
            # Fallback: use material ID as atomic number (for testing)
            Z_material = torch.arange(M, device=self.device, dtype=torch.int32) + 1

        electron_condensed_step_kernel[grid](
            pos, direction, E, w, rng, ebin,
            material_id, rho, self.tables.ref_density_g_cm3,
            self.tables.S_restricted, self.tables.range_csda_cm,
            P_brem, P_delta,
            Z_material,  # Atomic numbers for materials
            edep_flat,
            pos2, dir2, E2, w2, rng2, ebin2,
            alive, emit_b, emit_d,
            Z=Z, Y=Y, X=X,
            M=M, ECOUNT=ECOUNT,
            N=N,  # number of particles
            voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            f_vox=f_vox, f_range=f_range, max_dE_frac=max_dE_frac,
            BLOCK_SIZE_KERNEL=256,
        )

        if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
            # Brems photons
            bmask = emit_b.to(torch.bool) & (E2 > 0) & (w2 > 0)
            idx = select_indices_with_budget(
                bmask,
                sec_counts,
                max_per_primary=max_secondaries_per_primary,
                max_per_step=max_secondaries_per_step,
            )
            if int(idx.numel()) > 0:
                ns = int(idx.numel())
                n_brems += ns

                s_pos = pos2.index_select(0, idx)
                s_dir = dir2.index_select(0, idx)
                s_E = E2.index_select(0, idx)
                s_w = w2.index_select(0, idx)
                s_rng = rng2.index_select(0, idx)
                s_ebin = ebin2.index_select(0, idx)

                out_parent_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                out_rng = torch.empty((ns,), device=self.device, dtype=torch.int32)
                ph_pos = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                ph_dir = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                ph_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                ph_w = torch.empty((ns,), device=self.device, dtype=torch.float32)

                g2 = (triton.cdiv(ns, 256),)
                electron_brems_emit_kernel[g2](
                    s_pos, s_dir, s_E, s_w, s_rng, s_ebin,
                    brem_inv_cdf, K_brem,
                    out_parent_E, out_rng,
                    ph_pos, ph_dir, ph_E, ph_w,
                    ECOUNT=ECOUNT,
                    BLOCK=256,
                    num_warps=4,
                )

                E2.index_copy_(0, idx, out_parent_E)
                rng2.index_copy_(0, idx, out_rng)

                escaped_brems += self._run_photons_inplace(
                    pos=ph_pos,
                    direction=ph_dir,
                    E=ph_E,
                    w=ph_w,
                    edep=edep,
                    secondary_depth=secondary_depth - 1,
                    max_secondaries_per_primary=max_secondaries_per_primary,
                    max_secondaries_per_step=max_secondaries_per_step,
                )

            # Delta electrons (transport as condensed electrons; bounded recursion)
            dmask = emit_d.to(torch.bool) & (E2 > 0) & (w2 > 0)
            idx = select_indices_with_budget(
                dmask,
                sec_counts,
                max_per_primary=max_secondaries_per_primary,
                max_per_step=max_secondaries_per_step,
            )
            if int(idx.numel()) > 0:
                ns = int(idx.numel())
                n_delta += ns

                s_pos = pos2.index_select(0, idx)
                s_dir = dir2.index_select(0, idx)
                s_E = E2.index_select(0, idx)
                s_w = w2.index_select(0, idx)
                s_rng = rng2.index_select(0, idx)
                s_ebin = ebin2.index_select(0, idx)

                out_parent_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                out_rng = torch.empty((ns,), device=self.device, dtype=torch.int32)
                de_pos = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                de_dir = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                de_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                de_w = torch.empty((ns,), device=self.device, dtype=torch.float32)

                g2 = (triton.cdiv(ns, 256),)
                electron_delta_emit_kernel[g2](
                    s_pos, s_dir, s_E, s_w, s_rng, s_ebin,
                    delta_inv_cdf, K_delta,
                    out_parent_E, out_rng,
                    de_pos, de_dir, de_E, de_w,
                    ECOUNT=ECOUNT,
                    BLOCK=256,
                    num_warps=4,
                )

                E2.index_copy_(0, idx, out_parent_E)
                rng2.index_copy_(0, idx, out_rng)

                # Transport delta electrons (bounded recursion)
                esc2, nb2, nd2 = self._run_electrons_inplace(
                    pos=de_pos,
                    direction=de_dir,
                    E=de_E,
                    w=de_w,
                    edep=edep,
                    secondary_depth=secondary_depth - 1,
                    max_secondaries_per_primary=max_secondaries_per_primary,
                    max_secondaries_per_step=max_secondaries_per_step,
                )
                escaped_brems += esc2
                n_brems += nb2
                n_delta += nd2

        pos, direction, E, w, rng = pos2, dir2, E2, w2, rng2

        return float(escaped_brems), int(n_brems), int(n_delta)

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
        max_secondaries_per_step: int = 1_000_000_000,
    ) -> tuple[float, int, int, int]:
        N = int(E.numel())
        if N == 0:
            return 0.0, 0, 0, 0

        Z, Y, X = self.mats.material_id.shape
        vx, vy, vz = self.voxel_size_cm

        e_cut = float(self.sim_config.get("cutoffs", {}).get("electron_keV", 20.0)) * 1e-3
        max_steps = int(self.sim_config.get("electron_transport", {}).get("max_steps", 4096))
        et = self.sim_config.get("electron_transport", {})
        f_vox = float(et.get("f_voxel", 0.3))
        f_range = float(et.get("f_range", 0.2))
        max_dE_frac = float(et.get("max_dE_frac", 0.2))

        material_id = self.mats.material_id.to(self.device, dtype=torch.int32).contiguous()
        rho = self.mats.rho.to(self.device, dtype=torch.float32).contiguous()

        M = int(self.tables.ref_density_g_cm3.numel())
        ECOUNT = int(self.tables.e_centers_MeV.numel())

        P_brem = self.tables.P_brem_per_cm if self.tables.P_brem_per_cm is not None else torch.zeros_like(self.tables.S_restricted)
        P_delta = self.tables.P_delta_per_cm if self.tables.P_delta_per_cm is not None else torch.zeros_like(self.tables.S_restricted)

        brem_inv_cdf, K_brem = self._inv_cdf_or_uniform(self.tables.brem_inv_cdf_Efrac, ECOUNT=ECOUNT, max_efrac=0.3)
        delta_inv_cdf, K_delta = self._inv_cdf_or_uniform(self.tables.delta_inv_cdf_Efrac, ECOUNT=ECOUNT, max_efrac=0.5)

        rng = self._rng_i32(N)
        rng2 = torch.empty_like(rng)

        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)

        ebin = torch.empty((N,), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((N,), device=self.device, dtype=torch.int8)
        emit_b = torch.empty_like(alive)
        emit_d = torch.empty_like(alive)
        stop = torch.empty_like(alive)

        edep_flat = edep.view(-1)
        grid = (triton.cdiv(N, 256),)

        escaped = 0.0
        annihilations = 0
        n_brems = 0
        n_delta = 0

        sec_counts = torch.zeros((N,), device=self.device, dtype=torch.int32)

        for _ in range(max_steps):
            if int((E > 0).sum().item()) == 0:
                break

            ebin = torch.bucketize(E, self.tables.e_edges_MeV) - 1
            ebin = torch.clamp(ebin, 0, ECOUNT - 1).to(torch.int32)

            positron_condensed_step_kernel[grid](
                pos, direction, E, w, rng, ebin,
                material_id, rho, self.tables.ref_density_g_cm3,
                self.tables.S_restricted, self.tables.range_csda_cm,
                P_brem, P_delta,
                edep_flat,
                pos2, dir2, E2, w2, rng2, ebin2,
                alive, emit_b, emit_d, stop,
                Z=Z, Y=Y, X=X,
                M=M, ECOUNT=ECOUNT,
                BLOCK=256,
                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
                f_vox=f_vox, f_range=f_range, max_dE_frac=max_dE_frac,
                e_cut_MeV=e_cut,
                num_warps=4,
            )

            # Brems/delta secondaries (bounded recursion + per-primary cap)
            if allow_secondaries(secondary_depth=secondary_depth, max_per_primary=max_secondaries_per_primary):
                bmask = emit_b.to(torch.bool) & (E2 > 0) & (w2 > 0)
                idx = select_indices_with_budget(
                    bmask,
                    sec_counts,
                    max_per_primary=max_secondaries_per_primary,
                    max_per_step=max_secondaries_per_step,
                )
                if int(idx.numel()) > 0:
                    ns = int(idx.numel())
                    n_brems += ns

                    s_pos = pos2.index_select(0, idx)
                    s_dir = dir2.index_select(0, idx)
                    s_E = E2.index_select(0, idx)
                    s_w = w2.index_select(0, idx)
                    s_rng = rng2.index_select(0, idx)
                    s_ebin = ebin2.index_select(0, idx)

                    out_parent_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    out_rng = torch.empty((ns,), device=self.device, dtype=torch.int32)
                    ph_pos = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    ph_dir = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    ph_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    ph_w = torch.empty((ns,), device=self.device, dtype=torch.float32)

                    g2 = (triton.cdiv(ns, 256),)
                    electron_brems_emit_kernel[g2](
                        s_pos, s_dir, s_E, s_w, s_rng, s_ebin,
                        brem_inv_cdf, K_brem,
                        out_parent_E, out_rng,
                        ph_pos, ph_dir, ph_E, ph_w,
                        ECOUNT=ECOUNT,
                        BLOCK=256,
                        num_warps=4,
                    )
                    E2.index_copy_(0, idx, out_parent_E)
                    rng2.index_copy_(0, idx, out_rng)
                    escaped += self._run_photons_inplace(
                        pos=ph_pos,
                        direction=ph_dir,
                        E=ph_E,
                        w=ph_w,
                        edep=edep,
                        secondary_depth=secondary_depth - 1,
                        max_secondaries_per_primary=max_secondaries_per_primary,
                        max_secondaries_per_step=max_secondaries_per_step,
                    )

                dmask = emit_d.to(torch.bool) & (E2 > 0) & (w2 > 0)
                idx = select_indices_with_budget(
                    dmask,
                    sec_counts,
                    max_per_primary=max_secondaries_per_primary,
                    max_per_step=max_secondaries_per_step,
                )
                if int(idx.numel()) > 0:
                    ns = int(idx.numel())
                    n_delta += ns

                    s_pos = pos2.index_select(0, idx)
                    s_dir = dir2.index_select(0, idx)
                    s_E = E2.index_select(0, idx)
                    s_w = w2.index_select(0, idx)
                    s_rng = rng2.index_select(0, idx)
                    s_ebin = ebin2.index_select(0, idx)

                    out_parent_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    out_rng = torch.empty((ns,), device=self.device, dtype=torch.int32)
                    de_pos = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    de_dir = torch.empty((ns, 3), device=self.device, dtype=torch.float32)
                    de_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                    de_w = torch.empty((ns,), device=self.device, dtype=torch.float32)

                    g2 = (triton.cdiv(ns, 256),)
                    electron_delta_emit_kernel[g2](
                        s_pos, s_dir, s_E, s_w, s_rng, s_ebin,
                        delta_inv_cdf, K_delta,
                        out_parent_E, out_rng,
                        de_pos, de_dir, de_E, de_w,
                        ECOUNT=ECOUNT,
                        BLOCK=256,
                        num_warps=4,
                    )
                    E2.index_copy_(0, idx, out_parent_E)
                    rng2.index_copy_(0, idx, out_rng)

                    # Transport delta electrons (bounded recursion)
                    esc2, nb2, nd2 = self._run_electrons_inplace(
                        pos=de_pos,
                        direction=de_dir,
                        E=de_E,
                        w=de_w,
                        edep=edep,
                        secondary_depth=secondary_depth - 1,
                        max_secondaries_per_primary=max_secondaries_per_primary,
                        max_secondaries_per_step=max_secondaries_per_step,
                    )
                    escaped += esc2
                    n_brems += nb2
                    n_delta += nd2

            # Handle stops (annihilation at rest)
            stop_mask = stop.to(torch.bool) & (w2 > 0)
            if torch.any(stop_mask):
                idx = torch.nonzero(stop_mask, as_tuple=False).flatten()
                ns = int(idx.numel())
                annihilations += ns

                s_pos = pos2.index_select(0, idx)
                s_dir = dir2.index_select(0, idx)
                s_E = E2.index_select(0, idx)
                s_w = w2.index_select(0, idx)
                s_rng = rng2.index_select(0, idx)

                ph1_pos = torch.empty_like(s_pos)
                ph1_dir = torch.empty_like(s_dir)
                ph1_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                ph1_w = torch.empty_like(s_w)

                ph2_pos = torch.empty_like(s_pos)
                ph2_dir = torch.empty_like(s_dir)
                ph2_E = torch.empty((ns,), device=self.device, dtype=torch.float32)
                ph2_w = torch.empty_like(s_w)

                out_rng = torch.empty_like(s_rng)

                g2 = (triton.cdiv(ns, 256),)
                positron_annihilation_at_rest_kernel[g2](
                    s_pos, s_dir, s_E, s_w, s_rng,
                    edep_flat,
                    ph1_pos, ph1_dir, ph1_E, ph1_w,
                    ph2_pos, ph2_dir, ph2_E, ph2_w,
                    out_rng,
                    material_id.view(-1),
                    Z=Z, Y=Y, X=X,
                    BLOCK=256,
                    voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
                )

                # Transport annihilation photons (2 per stop)
                ph_pos = torch.cat([ph1_pos, ph2_pos], dim=0)
                ph_dir = torch.cat([ph1_dir, ph2_dir], dim=0)
                ph_E = torch.cat([ph1_E, ph2_E], dim=0)
                ph_w = torch.cat([ph1_w, ph2_w], dim=0)

                escaped += self._run_photons_inplace(
                    pos=ph_pos,
                    direction=ph_dir,
                    E=ph_E,
                    w=ph_w,
                    edep=edep,
                    secondary_depth=secondary_depth - 1,
                    max_secondaries_per_primary=max_secondaries_per_primary,
                    max_secondaries_per_step=max_secondaries_per_step,
                )

                # Kill stopped positrons in the main arrays
                E2.index_fill_(0, idx, 0.0)
                w2.index_fill_(0, idx, 0.0)

            pos, direction, E, w, rng = pos2, dir2, E2, w2, rng2

        return escaped, annihilations, n_brems, n_delta



def _deposit_local(*, pos: torch.Tensor, E: torch.Tensor, w: torch.Tensor, edep_flat: torch.Tensor,
                   Z: int, Y: int, X: int, voxel_size_cm: tuple[float, float, float]) -> None:
    if E.numel() == 0:
        return
    vx, vy, vz = voxel_size_cm
    grid = (triton.cdiv(int(E.numel()), 256),)
    deposit_local_energy_kernel[grid](
        pos, E, w,
        edep_flat,
        Z=Z, Y=Y, X=X,
        BLOCK=256,
        voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
    )
