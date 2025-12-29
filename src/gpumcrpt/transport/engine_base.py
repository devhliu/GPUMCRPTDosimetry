"""
Base engine class to eliminate redundancies across different transport engines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import torch
import triton

from gpumcrpt.transport.triton_kernels.utils.deposit import deposit_local_energy_kernel
from gpumcrpt.physics_tables.relaxation_tables import RelaxationTables


@dataclass
class BaseStats:
    escaped_energy_MeV: float = 0.0


class BaseTransportEngine:
    """Base transport engine with common functionality to eliminate redundancies."""

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
            raise RuntimeError("Transport engine requires CUDA")

        self.mats = mats
        self.tables = tables
        self.sim_config = sim_config
        self.voxel_size_cm = voxel_size_cm
        self.device = device

        # Initialize relaxation tables if available
        self._relax_tables: RelaxationTables | None = None
        if hasattr(self.tables, "relax_shell_cdf") and hasattr(self.tables, "relax_E_bind_MeV"):
            if (hasattr(self.tables, "relax_fluor_yield") and
                hasattr(self.tables, "relax_E_xray_MeV") and
                hasattr(self.tables, "relax_E_auger_MeV")):
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

    def _deposit_local(
        self,
        *,
        pos: torch.Tensor,
        E: torch.Tensor,
        w: torch.Tensor,
        edep_flat: torch.Tensor,
        Z: int, Y: int, X: int,
        voxel_size_cm: tuple[float, float, float]
    ) -> None:
        """Deposit energy locally in voxels."""
        if E.numel() == 0:
            return
        vx, vy, vz = voxel_size_cm
        N = int(E.numel())
        grid = (triton.cdiv(N, 256),)
        deposit_local_energy_kernel[grid](
            pos, E, w,
            edep_flat,
            Z=Z, Y=Y, X=X,
            voxel_z_cm=float(vz), voxel_y_cm=float(vy), voxel_x_cm=float(vx),
            N=N,
        )

    def _ensure_rng(self, q: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Ensure RNG exists in particle queue."""
        if "rng" not in q:
            n = q["E_MeV"].shape[0]
            q["rng"] = torch.randint(1, 2**31 - 1, (n,), device=device, dtype=torch.int32)
        return q

    def _precompute_ebin(self, E: torch.Tensor, e_edges: torch.Tensor) -> torch.Tensor:
        """Precompute energy bins."""
        return (torch.bucketize(E, e_edges) - 1).clamp_(0, e_edges.numel() - 2).to(torch.int32)

    def _get_inverse_cdf_or_uniform(self, inv_cdf: torch.Tensor | None, *, ECOUNT: int, max_efrac: float) -> tuple[torch.Tensor, int]:
        """Get inverse CDF or uniform placeholder."""
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

    def _get_material_properties(self) -> tuple[int, int, int, int, int]:
        """Get material properties: Z, Y, X, M, ECOUNT."""
        Z, Y, X = self.mats.material_id.shape
        M = int(self.tables.ref_density_g_cm3.numel())
        ECOUNT = int(self.tables.e_edges_MeV.numel() - 1)
        return Z, Y, X, M, ECOUNT

    def _create_rng_state(self, N: int) -> torch.Tensor:
        """Create initial RNG state."""
        g = torch.Generator(device=self.device)
        g.manual_seed(int(self.sim_config.get("seed", 0)))
        return torch.randint(1, 2**31 - 1, (N,), generator=g, device=self.device, dtype=torch.int32)

    def _get_cutoff_energy(self, particle_type: str) -> float:
        """Get cutoff energy for a particle type."""
        if particle_type == "photon":
            return float(self.sim_config.get("cutoffs", {}).get("photon_keV", 3.0)) * 1e-3
        elif particle_type == "electron":
            return float(self.sim_config.get("cutoffs", {}).get("electron_keV", 20.0)) * 1e-3
        else:
            return 0.0  # Default for other particles

    def _get_max_steps(self) -> int:
        """Get maximum number of wavefront iterations."""
        return int(self.sim_config.get("monte_carlo", {}).get("max_wavefront_iters", 512))

    def _create_ping_pong_buffers(self, pos: torch.Tensor, direction: torch.Tensor,
                                  E: torch.Tensor, w: torch.Tensor,
                                  rng: torch.Tensor | None = None) -> tuple:
        """Create ping-pong buffers for particle transport."""
        pos2 = torch.empty_like(pos)
        dir2 = torch.empty_like(direction)
        E2 = torch.empty_like(E)
        w2 = torch.empty_like(w)
        rng2 = torch.empty_like(rng) if rng is not None else None

        ebin = torch.empty((E.numel(),), device=self.device, dtype=torch.int32)
        ebin2 = torch.empty_like(ebin)

        alive = torch.empty((E.numel(),), device=self.device, dtype=torch.int8)
        real = torch.empty_like(alive)

        return pos2, dir2, E2, w2, rng2, ebin, ebin2, alive, real