from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class RelaxationTables:
    # shapes: [M,S]
    relax_shell_cdf: torch.Tensor        # float32
    relax_E_bind_MeV: torch.Tensor       # float32
    relax_fluor_yield: torch.Tensor      # float32
    relax_E_xray_MeV: torch.Tensor       # float32
    relax_E_auger_MeV: torch.Tensor      # float32

    @staticmethod
    def dummy(device: torch.device | str, M: int, S: int = 4) -> "RelaxationTables":
        """
        Debug-only placeholder tables (not physically accurate).
        Lets you integrate Phase 9 plumbing before real EADL-derived data is loaded.
        """
        device = torch.device(device)
        # shell cdf: uniform over shells
        shell_cdf = torch.linspace(1.0 / S, 1.0, S, device=device, dtype=torch.float32).view(1, S).repeat(M, 1).contiguous()

        # crude binding energies in MeV (K ~ 0.01 MeV, L ~ 0.002 MeV) just for wiring
        Ebind = torch.tensor([0.010, 0.002, 0.001, 0.0005], device=device, dtype=torch.float32).view(1, S).repeat(M, 1).contiguous()
        fy = torch.tensor([0.8, 0.3, 0.2, 0.1], device=device, dtype=torch.float32).view(1, S).repeat(M, 1).contiguous()

        Ex = 0.8 * Ebind
        Ea = Ebind - Ex
        return RelaxationTables(shell_cdf, Ebind, fy, Ex, Ea)


@dataclass
class MaterialElementCDF:
    """Mixture mapping for materials composed of multiple elements.

    material_element_cdf[m, :] is a cumulative distribution over elements for material m,
    derived from its elemental mass fractions.
    """

    element_Z: torch.Tensor            # int32 [E]
    material_element_cdf: torch.Tensor # float32 [M,E]


def build_material_element_cdf(*, element_Z: torch.Tensor, material_wfrac: torch.Tensor) -> MaterialElementCDF:
    """Build a per-material element selection CDF from mass fractions.

    element_Z: int32 [E]
    material_wfrac: float32 [M,E], rows should sum to 1 (will be normalized defensively)
    """
    if material_wfrac.ndim != 2:
        raise ValueError(f"material_wfrac must be [M,E], got shape={tuple(material_wfrac.shape)}")
    if element_Z.ndim != 1:
        raise ValueError(f"element_Z must be [E], got shape={tuple(element_Z.shape)}")
    if material_wfrac.shape[1] != element_Z.numel():
        raise ValueError("material_wfrac second dim must match element_Z length")

    w = material_wfrac.to(dtype=torch.float32)
    w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
    cdf = torch.cumsum(w, dim=1)
    cdf = torch.clamp(cdf, min=0.0, max=1.0)
    # ensure last entry exactly 1.0
    cdf[:, -1] = 1.0

    return MaterialElementCDF(
        element_Z=element_Z.to(dtype=torch.int32),
        material_element_cdf=cdf.contiguous(),
    )