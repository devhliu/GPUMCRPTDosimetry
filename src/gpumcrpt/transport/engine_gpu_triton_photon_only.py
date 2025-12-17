from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from gpumcrpt.transport.triton.photon_flight import photon_woodcock_flight_kernel
from gpumcrpt.transport.triton.photon_interactions import photon_classify_kernel
from gpumcrpt.transport.triton.compton import photon_compton_kernel
from gpumcrpt.transport.triton.rayleigh import photon_rayleigh_kernel
from gpumcrpt.transport.triton.edep_deposit import deposit_local_energy_kernel


@dataclass
class PhotonOnlyStats:
    escaped_energy_MeV: float


class TritonPhotonOnlyTransportEngine:
    """Milestone 2 engine: photon-only transport (GPU/Triton).

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
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TritonPhotonOnlyTransportEngine requires CUDA")

        self.mats = mats
        self.tables = tables
        self.sim_config = sim_config
        self.voxel_size_cm = voxel_size_cm
        self.device = device

        self._last_stats = PhotonOnlyStats(escaped_energy_MeV=0.0)

    @property
    def last_stats(self) -> PhotonOnlyStats:
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
            _deposit_local(
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
            self._last_stats = PhotonOnlyStats(escaped_energy_MeV=0.0)
            return edep

        # RNG state for Triton kernels (xorshift32), int32
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.sim_config.get("seed", 0)))
        rng = torch.randint(1, 2**31 - 1, (N,), generator=g, device=self.device, dtype=torch.int32)

        # Preallocate ping-pong buffers
        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)
        rng2 = torch.empty_like(rng)

        ebin = torch.empty((N,), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((N,), device=self.device, dtype=torch.int8)
        real = torch.empty_like(alive)
        typ = torch.empty((N,), device=self.device, dtype=torch.int8)

        # Interaction outputs
        scat_pos = torch.empty_like(pos)
        scat_dir = torch.empty_like(direction)
        scat_E = torch.empty_like(E)
        scat_w = torch.empty_like(w)
        scat_rng = torch.empty_like(rng)
        scat_ebin = torch.empty_like(ebin)

        # Dummy electron outputs for Compton kernel (we deposit locally using E_e and pos)
        e_pos = torch.empty_like(pos)
        e_dir = torch.empty_like(direction)
        e_E = torch.empty_like(E)
        e_w = torch.empty_like(w)

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
                _deposit_local(
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
                BLOCK=256,
                voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            )

            # Escaped photons: alive==0 after flight
            escaped_mask = (alive == 0) & (E2 > 0) & (w2 > 0)
            if torch.any(escaped_mask):
                escaped_energy = escaped_energy + (E2[escaped_mask] * w2[escaped_mask]).sum(dtype=torch.float32)
                E2 = torch.where(escaped_mask, torch.zeros_like(E2), E2)
                w2 = torch.where(escaped_mask, torch.zeros_like(w2), w2)

            # Classify only real interactions
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

            # Interaction handling
            # Photoelectric -> deposit all E2
            pe = (typ == 1) & (E2 > 0) & (w2 > 0)
            if torch.any(pe):
                _deposit_local(
                    pos=pos2,
                    E=torch.where(pe, E2, torch.zeros_like(E2)),
                    w=w2,
                    edep_flat=edep.view(-1),
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )
                E2 = torch.where(pe, torch.zeros_like(E2), E2)
                w2 = torch.where(pe, torch.zeros_like(w2), w2)

            # Pair -> deposit all E2 (MVP)
            pa = (typ == 4) & (E2 > 0) & (w2 > 0)
            if torch.any(pa):
                _deposit_local(
                    pos=pos2,
                    E=torch.where(pa, E2, torch.zeros_like(E2)),
                    w=w2,
                    edep_flat=edep.view(-1),
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )
                E2 = torch.where(pa, torch.zeros_like(E2), E2)
                w2 = torch.where(pa, torch.zeros_like(w2), w2)

            # Compton -> call kernel for all photons but only meaningful where typ==2
            # To avoid additional gather/compaction in Milestone 2, we run it over full N and gate using E==0/w==0.
            co = (typ == 2) & (E2 > 0) & (w2 > 0)
            if torch.any(co):
                # Use E==0 for non-co to no-op
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

                # Deposit recoil electron energy locally
                _deposit_local(
                    pos=pos2,
                    E=e_E,
                    w=e_w,
                    edep_flat=edep.view(-1),
                    Z=Z, Y=Y, X=X,
                    voxel_size_cm=self.voxel_size_cm,
                )

                # Update photon state only for Compton events; keep others unchanged.
                co3 = co[:, None]
                pos2 = torch.where(co3, scat_pos, pos2)
                dir2 = torch.where(co3, scat_dir, dir2)
                E2 = torch.where(co, scat_E, E2)
                w2 = torch.where(co, scat_w, w2)
                rng = torch.where(co, scat_rng, rng)
                ebin2 = torch.where(co, scat_ebin, ebin2)

            # Rayleigh -> direction change (energy conserved)
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

            # Ping-pong
            pos, direction, E, w, rng = pos2, dir2, E2, w2, rng

        self._last_stats = PhotonOnlyStats(escaped_energy_MeV=float(escaped_energy.detach().cpu().item()))
        return edep


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
