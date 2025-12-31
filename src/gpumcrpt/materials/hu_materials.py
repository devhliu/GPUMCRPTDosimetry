from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass
class MaterialsLibrary:
    """Per-material metadata for clinical-material realism.

    element_symbol and element_Z define the column ordering of material_wfrac.
    material_wfrac is mass fraction by element.
    """

    element_symbol: list[str]
    element_Z: torch.Tensor            # int32 [E]

    material_names: list[str]
    ref_density_g_cm3: torch.Tensor    # float32 [M]
    material_wfrac: torch.Tensor       # float32 [M,E]

    def to(self, device: str | torch.device) -> "MaterialsLibrary":
        def mv(x):
            return x.to(device=device) if isinstance(x, torch.Tensor) else x

        return MaterialsLibrary(
            element_symbol=list(self.element_symbol),
            element_Z=mv(self.element_Z),
            material_names=list(self.material_names),
            ref_density_g_cm3=mv(self.ref_density_g_cm3),
            material_wfrac=mv(self.material_wfrac),
        )


@dataclass
class MaterialsVolume:
    material_id: torch.Tensor  # int32 [Z,Y,X]
    rho: torch.Tensor          # float32 [Z,Y,X] in g/cm^3
    lib: MaterialsLibrary | None = None
    material_atom_Z: torch.Tensor | None = None  # int32 [M] (effective rounded Z)


def _interp1d_piecewise_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Torch-only 1D piecewise-linear interpolation."""
    if xp.numel() == 1:
        return torch.full_like(x, fp[0])

    x_flat = x.reshape(-1)
    idx = torch.searchsorted(xp, x_flat, right=True)
    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    t = (x_flat - x0) / torch.clamp(x1 - x0, min=1e-6)
    y = y0 + t * (y1 - y0)
    return y.reshape(x.shape)


def build_materials_library_from_config(
    cfg: Mapping,
    *,
    device: str | torch.device = "cuda",
    default_element_symbol: Sequence[str] = ("H", "C", "N", "O", "P", "Ca"),
) -> MaterialsLibrary:
    device = torch.device(device)
    lib_cfg = cfg.get("material_library", None)
    if lib_cfg is None:
        raise KeyError("materials.material_library not found")

    element_symbol = list(lib_cfg.get("elements", list(default_element_symbol)))
    z_map = {
        "H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "Ca": 20, "Ar": 18,
        "Na": 11, "S": 16, "Cl": 17, "K": 19
    }
    try:
        element_Z = torch.tensor([z_map[s] for s in element_symbol], device=device, dtype=torch.int32)
    except KeyError as e:
        raise ValueError(f"Unknown element symbol in material_library.elements: {e}") from e

    mats = list(lib_cfg.get("materials", []))
    if not mats:
        raise ValueError("materials.material_library.materials is empty")

    material_names: list[str] = []
    ref_density: list[float] = []
    wfrac_rows: list[list[float]] = []

    for m in mats:
        material_names.append(str(m["name"]))
        ref_density.append(float(m["ref_density_g_cm3"]))
        wfrac: Mapping[str, float] = m.get("wfrac", {})
        row = [float(wfrac.get(sym, 0.0)) for sym in element_symbol]
        s = sum(row)
        if s <= 0.0:
            raise ValueError(f"material_library entry {m['name']} has zero composition")
        wfrac_rows.append([v / s for v in row])

    return MaterialsLibrary(
        element_symbol=element_symbol,
        element_Z=element_Z,
        material_names=material_names,
        ref_density_g_cm3=torch.tensor(ref_density, device=device, dtype=torch.float32),
        material_wfrac=torch.tensor(wfrac_rows, device=device, dtype=torch.float32),
    )


def compute_material_effective_atom_Z(lib: MaterialsLibrary, *, power: float = 1.0) -> torch.Tensor:
    z = lib.element_Z.to(dtype=torch.float32)
    w = lib.material_wfrac.to(dtype=torch.float32)
    w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
    zeff = torch.sum(w * (z[None, :] ** float(power)), dim=1)
    zeff = torch.clamp(zeff, min=1.0)
    return torch.round(zeff).to(dtype=torch.int32)


def build_materials_from_hu(
    *,
    hu: torch.Tensor,
    hu_to_density: Sequence[Sequence[float]] | None,
    hu_to_class: Sequence[Sequence[float]] | None,
    material_library: MaterialsLibrary | None = None,
    device: str = "cuda",
) -> MaterialsVolume:
    """Build per-voxel material_id and density from HU."""
    hu = hu.to(device=device, dtype=torch.float32)

    if not hu_to_density:
        rho = torch.ones_like(hu, dtype=torch.float32)
    else:
        anchors = torch.tensor(list(hu_to_density), device=device, dtype=torch.float32)
        xp = anchors[:, 0]
        fp = anchors[:, 1]
        order = torch.argsort(xp)
        xp = xp[order]
        fp = fp[order]
        rho = _interp1d_piecewise_linear(hu, xp, fp)
        rho = torch.clamp(rho, min=0.0)

    material_id = torch.zeros_like(hu, dtype=torch.int32)
    if hu_to_class:
        for lo, hi, mid in hu_to_class:
            mask = (hu >= float(lo)) & (hu < float(hi))
            material_id = torch.where(mask, torch.as_tensor(int(mid), device=device, dtype=torch.int32), material_id)
    else:
        material_id.fill_(2)

    lib = material_library.to(device) if material_library is not None else None
    material_atom_Z = compute_material_effective_atom_Z(lib) if lib is not None else None
    return MaterialsVolume(material_id=material_id, rho=rho, lib=lib, material_atom_Z=material_atom_Z)


def validate_materials_library(lib: MaterialsLibrary) -> Dict[str, bool]:
    """Validate a materials library for consistency."""
    validation_results = {}
    
    # Check that weight fractions sum to 1.0
    row_sums = lib.material_wfrac.sum(dim=1)
    validation_results['weight_fractions_sum_to_one'] = torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0.0
    )
    
    # Check that densities are positive
    validation_results['positive_densities'] = torch.all(lib.ref_density_g_cm3 > 0)
    
    # Check that atomic numbers are valid
    validation_results['valid_atomic_numbers'] = torch.all((lib.element_Z >= 1) & (lib.element_Z <= 118))
    
    # Check that material names are unique
    validation_results['unique_material_names'] = len(lib.material_names) == len(set(lib.material_names))
    
    return validation_results


def get_material_composition_summary(lib: MaterialsLibrary) -> Dict[str, Dict]:
    """Get a summary of material compositions."""
    summary = {}
    
    for i, name in enumerate(lib.material_names):
        composition = {}
        for j, element in enumerate(lib.element_symbol):
            fraction = lib.material_wfrac[i, j].item()
            if fraction > 1e-6:  # Only include significant fractions
                composition[element] = fraction
        
        summary[name] = {
            'density_g_cm3': lib.ref_density_g_cm3[i].item(),
            'composition': composition,
            'effective_Z': compute_material_effective_atom_Z(lib)[i].item()
        }
    
    return summary


def compute_electron_density_relative_to_water(lib: MaterialsLibrary) -> torch.Tensor:
    """Compute electron density relative to water for each material."""
    
    # Atomic numbers and approximate atomic masses
    Z = lib.element_Z.to(dtype=torch.float32)
    # Approximate atomic masses (g/mol)
    A_approx = torch.tensor([
        1.008,   # H
        12.011,  # C
        14.007,  # N
        15.999,  # O
        30.974,  # P
        40.078,  # Ca
        39.948   # Ar
    ], device=Z.device, dtype=torch.float32)
    
    # Electron density per element (electrons per gram)
    electrons_per_gram = Z / A_approx
    
    # Water composition (H2O)
    water_composition = torch.tensor([[0.1119, 0.0, 0.0, 0.8881, 0.0, 0.0, 0.0]], 
                                    device=Z.device, dtype=torch.float32)
    water_electron_density = torch.sum(water_composition * electrons_per_gram)
    
    # Material electron densities
    material_electron_densities = torch.sum(
        lib.material_wfrac * electrons_per_gram, 
        dim=1
    )
    
    # Relative to water
    relative_electron_densities = material_electron_densities / water_electron_density
    
    return relative_electron_densities


def create_custom_materials_library(
    material_names: List[str],
    element_symbols: List[str],
    compositions: List[Dict[str, float]],
    densities: List[float],
    device: str = "cuda"
) -> MaterialsLibrary:
    """Create a custom materials library from user-provided data."""
    
    device = torch.device(device)
    
    # Map element symbols to atomic numbers
    z_map = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "Ca": 20, "Ar": 18,
             "Na": 11, "S": 16, "Cl": 17, "K": 19}
    
    element_Z = torch.tensor([z_map[s] for s in element_symbols], 
                            device=device, dtype=torch.int32)
    
    # Build weight fraction matrix
    wfrac_rows = []
    for comp in compositions:
        row = [comp.get(sym, 0.0) for sym in element_symbols]
        total = sum(row)
        if total <= 0.0:
            raise ValueError(f"Material composition sums to zero: {comp}")
        wfrac_rows.append([v / total for v in row])
    
    material_wfrac = torch.tensor(wfrac_rows, device=device, dtype=torch.float32)
    ref_density_g_cm3 = torch.tensor(densities, device=device, dtype=torch.float32)
    
    return MaterialsLibrary(
        element_symbol=element_symbols,
        element_Z=element_Z,
        material_names=material_names,
        ref_density_g_cm3=ref_density_g_cm3,
        material_wfrac=material_wfrac
    )

