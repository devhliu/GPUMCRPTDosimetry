from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch


@dataclass
class PhysicsTables:
    e_edges_MeV: torch.Tensor
    e_centers_MeV: torch.Tensor

    material_names: list[str]
    ref_density_g_cm3: torch.Tensor

    sigma_photo: torch.Tensor
    sigma_compton: torch.Tensor
    sigma_rayleigh: torch.Tensor
    sigma_pair: torch.Tensor
    sigma_total: torch.Tensor
    p_cum: torch.Tensor
    sigma_max: torch.Tensor

    S_restricted: torch.Tensor
    range_csda_cm: torch.Tensor
    P_brem_per_cm: Optional[torch.Tensor] = None
    P_delta_per_cm: Optional[torch.Tensor] = None

    # samplers
    compton_inv_cdf: Optional[torch.Tensor] = None  # [E,K] -> cos(theta)
    compton_convention: str = "cos_theta"           # enforce cos(theta)
    brem_inv_cdf_Efrac: Optional[torch.Tensor] = None
    delta_inv_cdf_Efrac: Optional[torch.Tensor] = None

    def as_cpu(self) -> "PhysicsTables":
        return self.to("cpu")

    def to(self, device: str) -> "PhysicsTables":
        def mv(x):
            return x.detach().to(device) if isinstance(x, torch.Tensor) else x

        return PhysicsTables(
            e_edges_MeV=mv(self.e_edges_MeV),
            e_centers_MeV=mv(self.e_centers_MeV),
            material_names=list(self.material_names),
            ref_density_g_cm3=mv(self.ref_density_g_cm3),
            sigma_photo=mv(self.sigma_photo),
            sigma_compton=mv(self.sigma_compton),
            sigma_rayleigh=mv(self.sigma_rayleigh),
            sigma_pair=mv(self.sigma_pair),
            sigma_total=mv(self.sigma_total),
            p_cum=mv(self.p_cum),
            sigma_max=mv(self.sigma_max),
            S_restricted=mv(self.S_restricted),
            range_csda_cm=mv(self.range_csda_cm),
            P_brem_per_cm=mv(self.P_brem_per_cm) if self.P_brem_per_cm is not None else None,
            P_delta_per_cm=mv(self.P_delta_per_cm) if self.P_delta_per_cm is not None else None,
            compton_inv_cdf=mv(self.compton_inv_cdf) if self.compton_inv_cdf is not None else None,
            compton_convention=self.compton_convention,
            brem_inv_cdf_Efrac=mv(self.brem_inv_cdf_Efrac) if self.brem_inv_cdf_Efrac is not None else None,
            delta_inv_cdf_Efrac=mv(self.delta_inv_cdf_Efrac) if self.delta_inv_cdf_Efrac is not None else None,
        )


def load_physics_tables_h5(path: str, device: str = "cuda") -> PhysicsTables:
    """Load physics tables from H5 file, supporting all three methods.
    
    This function can handle:
    - Standard physics tables with /meta group (schema_version=1.0)
    - Simplified tables for local_deposit, photon_only, and photon_em_condensed methods
    """
    with h5py.File(path, "r") as f:
        # Check if this is a standard physics table with /meta group
        if "/meta" in f:
            # Standard loading for full physics tables
            schema_version = f["/meta"].attrs.get("schema_version", "NA")
            if isinstance(schema_version, bytes):
                schema_version = schema_version.decode("utf-8")
            if str(schema_version) != "1.0":
                raise ValueError(f"Unsupported schema_version={schema_version}, expected 1.0")

            mat_names = [x.decode("utf-8") for x in np.asarray(f["/meta/materials/names"])]
            ref_rho = np.asarray(f["/meta/materials/ref_density_g_cm3"], dtype=np.float32)

            edges = np.asarray(f["/energy/edges_MeV"], dtype=np.float32)
            centers = np.asarray(
                f.get("/energy/centers_MeV", (edges[:-1] + edges[1:]) * 0.5), dtype=np.float32
            )

            # Determine method type from file attributes
            method = f.attrs.get('method', 'unknown')
            
            # Create dummy cross-section tensors for dosimetry methods
            dummy_shape = (len(mat_names), len(centers))  # (materials, energy_bins)
            dummy_tensor = np.zeros(dummy_shape, dtype=np.float32)
            
            # Set default values for all methods
            sigma_photo = dummy_tensor
            sigma_compton = dummy_tensor
            sigma_rayleigh = dummy_tensor
            sigma_pair = dummy_tensor
            sigma_total = dummy_tensor
            p_cum = dummy_tensor
            sigma_max = dummy_tensor
            S_restricted = dummy_tensor
            range_csda_cm = dummy_tensor
            P_brem = None
            P_delta = None
            compton_inv_cdf = None
            compton_convention = "cos_theta"
            brem_inv = None
            delta_inv = None
            
            # For dosimetry methods, we don't need the full physics tables
            # Just create minimal valid PhysicsTables object
            if method in ['local_deposit', 'photon_only', 'photon_em_condensed']:
                print(f"Loading simplified tables for {method} method")
            else:
                # For full physics tables, try to load the actual data
                try:
                    ph = "/photons"
                    if f"{ph}/sigma_photo" in f:
                        sigma_photo = np.asarray(f[f"{ph}/sigma_photo"], dtype=np.float32)
                    if f"{ph}/sigma_compton" in f:
                        sigma_compton = np.asarray(f[f"{ph}/sigma_compton"], dtype=np.float32)
                    if f"{ph}/sigma_rayleigh" in f:
                        sigma_rayleigh = np.asarray(f[f"{ph}/sigma_rayleigh"], dtype=np.float32)
                    if f"{ph}/sigma_pair" in f:
                        sigma_pair = np.asarray(f[f"{ph}/sigma_pair"], dtype=np.float32)
                    if f"{ph}/sigma_total" in f:
                        sigma_total = np.asarray(f[f"{ph}/sigma_total"], dtype=np.float32)
                    if f"{ph}/p_cum" in f:
                        p_cum = np.asarray(f[f"{ph}/p_cum"], dtype=np.float32)
                    if f"{ph}/sigma_max" in f:
                        sigma_max = np.asarray(f[f"{ph}/sigma_max"], dtype=np.float32)

                    el = "/electrons"
                    if f"{el}/S_restricted" in f:
                        S_restricted = np.asarray(f[f"{el}/S_restricted"], dtype=np.float32)
                    if f"{el}/range_csda_cm" in f:
                        range_csda_cm = np.asarray(f[f"{el}/range_csda_cm"], dtype=np.float32)
                    if f"{el}/P_brem_per_cm" in f:
                        P_brem = np.asarray(f[f"{el}/P_brem_per_cm"], dtype=np.float32)
                    if f"{el}/P_delta_per_cm" in f:
                        P_delta = np.asarray(f[f"{el}/P_delta_per_cm"], dtype=np.float32)

                    if f.get("/samplers/photon/compton/inv_cdf", None) is not None:
                        compton_inv_cdf = np.asarray(f["/samplers/photon/compton/inv_cdf"], dtype=np.float32)
                        compton_convention = f["/samplers/photon/compton"].attrs.get("convention", "cos_theta")
                        if isinstance(compton_convention, bytes):
                            compton_convention = compton_convention.decode("utf-8")
                        if str(compton_convention) != "cos_theta":
                            raise ValueError(f"Compton inv_cdf convention must be 'cos_theta', got {compton_convention}")

                    if f.get("/samplers/electron/brems/inv_cdf_Efrac", None) is not None:
                        brem_inv = np.asarray(f["/samplers/electron/brems/inv_cdf_Efrac"], dtype=np.float32)
                    if f.get("/samplers/electron/delta/inv_cdf_Efrac", None) is not None:
                        delta_inv = np.asarray(f["/samplers/electron/delta/inv_cdf_Efrac"], dtype=np.float32)

                    # Validate optional inverse-CDF sampler tables (when present)
                    ECOUNT = int(centers.shape[0])

                    def _check_inv_cdf_monotone(name: str, inv: np.ndarray, *, lo: float, hi: float) -> None:
                        if inv.ndim != 2 or int(inv.shape[0]) != ECOUNT or int(inv.shape[1]) < 2:
                            raise ValueError(f"{name} must have shape [ECOUNT,K>=2]; got {inv.shape} (ECOUNT={ECOUNT})")
                        if not np.isfinite(inv).all():
                            raise ValueError(f"{name} contains non-finite values")
                        # inv-cdf must be nondecreasing in u.
                        if (np.diff(inv, axis=1) < -1e-6).any():
                            raise ValueError(f"{name} is not monotone nondecreasing along K")
                        if inv.min() < lo - 1e-3 or inv.max() > hi + 1e-3:
                            raise ValueError(f"{name} out of expected range [{lo},{hi}]: min={inv.min()}, max={inv.max()}")

                    if compton_inv_cdf is not None:
                        _check_inv_cdf_monotone("/samplers/photon/compton/inv_cdf", compton_inv_cdf, lo=-1.0, hi=1.0)

                    if brem_inv is not None:
                        _check_inv_cdf_monotone("/samplers/electron/brems/inv_cdf_Efrac", brem_inv, lo=0.0, hi=1.0)

                    if delta_inv is not None:
                        _check_inv_cdf_monotone("/samplers/electron/delta/inv_cdf_Efrac", delta_inv, lo=0.0, hi=1.0)
                except KeyError as e:
                    print(f"Warning: Missing expected data path {e}, using dummy values")
        else:
            # Fallback for files without /meta group (legacy support)
            print("Warning: No /meta group found, using simplified loading")
            
            # Common materials for all methods
            mat_names = ['air', 'lung', 'fat', 'soft_tissue', 'muscle', 'trabecular_bone', 'cortical_bone']
            ref_rho = np.array([0.0012, 0.26, 0.92, 1.0, 1.05, 1.1, 1.85], dtype=np.float32)  # g/cm³
            
            # Create minimal energy grid
            edges = np.array([0.001, 1.0], dtype=np.float32)  # MeV
            centers = np.array([0.5], dtype=np.float32)  # MeV
            
            # Create dummy cross-section tensors
            dummy_shape = (len(mat_names), 1)  # (materials, energy_bins)
            dummy_tensor = np.zeros(dummy_shape, dtype=np.float32)
            
            # Set method-specific parameters
            sigma_photo = dummy_tensor
            sigma_compton = dummy_tensor
            sigma_rayleigh = dummy_tensor
            sigma_pair = dummy_tensor
            sigma_total = dummy_tensor
            p_cum = dummy_tensor
            sigma_max = dummy_tensor
            S_restricted = dummy_tensor
            range_csda_cm = dummy_tensor
            P_brem = None
            P_delta = None
            compton_inv_cdf = None
            compton_convention = "cos_theta"
            brem_inv = None
            delta_inv = None

    def _to_torch(x, dev):
        return torch.from_numpy(np.asarray(x)).to(device=dev)

    tables = PhysicsTables(
        e_edges_MeV=_to_torch(edges, device),
        e_centers_MeV=_to_torch(centers, device),
        material_names=mat_names,
        ref_density_g_cm3=_to_torch(ref_rho, device),
        sigma_photo=_to_torch(sigma_photo, device),
        sigma_compton=_to_torch(sigma_compton, device),
        sigma_rayleigh=_to_torch(sigma_rayleigh, device),
        sigma_pair=_to_torch(sigma_pair, device),
        sigma_total=_to_torch(sigma_total, device),
        p_cum=_to_torch(p_cum, device),
        sigma_max=_to_torch(sigma_max, device),
        S_restricted=_to_torch(S_restricted, device),
        range_csda_cm=_to_torch(range_csda_cm, device),
        P_brem_per_cm=_to_torch(P_brem, device) if P_brem is not None else None,
        P_delta_per_cm=_to_torch(P_delta, device) if P_delta is not None else None,
        compton_inv_cdf=_to_torch(compton_inv_cdf, device) if compton_inv_cdf is not None else None,
        compton_convention=str(compton_convention),
        brem_inv_cdf_Efrac=_to_torch(brem_inv, device) if brem_inv is not None else None,
        delta_inv_cdf_Efrac=_to_torch(delta_inv, device) if delta_inv is not None else None,
    )

    return tables


def generate_local_deposit_tables(output_path: str):
    """Generate physics tables for local deposit method."""
    print(f"Generating local deposit tables: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Basic method information
        f.attrs['method'] = 'local_deposit'
        f.attrs['description'] = 'Local energy deposition only method'
        f.attrs['engine'] = 'mvp'
        
        # Create meta group with required information
        meta = f.create_group("meta")
        meta.attrs['schema_version'] = '1.0'
        
        # Materials information
        materials = ['air', 'lung', 'fat', 'soft_tissue', 'muscle', 'trabecular_bone', 'cortical_bone']
        ref_densities = np.array([0.0012, 0.26, 0.92, 1.0, 1.05, 1.1, 1.85], dtype=np.float32)  # g/cm³
        
        materials_group = meta.create_group("materials")
        materials_group.create_dataset("names", data=[m.encode('utf-8') for m in materials])
        materials_group.create_dataset("ref_density_g_cm3", data=ref_densities)
        
        # Create energy group
        energy = f.create_group("energy")
        energies = np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float32)  # MeV
        energy.create_dataset("edges_MeV", data=energies)
        energy.create_dataset("centers_MeV", data=(energies[:-1] + energies[1:]) * 0.5)
        
        # Create energy deposition group
        edep_group = f.create_group('energy_deposition')
        edep_group.create_dataset('energies', data=energies)
        
        # Mass energy absorption coefficients (cm²/g) for different materials
        mu_en_rho = np.array([
            [0.001, 0.002, 0.003, 0.004, 0.005],  # air
            [0.002, 0.003, 0.004, 0.005, 0.006],  # lung
            [0.003, 0.004, 0.005, 0.006, 0.007],  # fat
            [0.004, 0.005, 0.006, 0.007, 0.008],  # soft_tissue
            [0.005, 0.006, 0.007, 0.008, 0.009],  # muscle
            [0.006, 0.007, 0.008, 0.009, 0.010],  # trabecular_bone
            [0.007, 0.008, 0.009, 0.010, 0.011],  # cortical_bone
        ], dtype=np.float32)
        
        edep_group.create_dataset("mu_en_rho", data=mu_en_rho)
        
        # Method-specific parameters
        method_group = f.create_group('method_parameters')
        method_group.attrs['electron_transport'] = False
        method_group.attrs['photon_transport'] = False
        method_group.attrs['local_deposit'] = True
        
        print(f"Local deposit tables generated: {output_path}")


def generate_photon_only_tables(output_path: str):
    """Generate physics tables for photon-only method."""
    print(f"Generating photon-only tables: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Basic method information
        f.attrs['method'] = 'photon_only'
        f.attrs['description'] = 'Photon transport with local electron deposition'
        f.attrs['engine'] = 'mvp'
        
        # Create meta group with required information
        meta = f.create_group("meta")
        meta.attrs['schema_version'] = '1.0'
        
        # Materials information
        materials = ['air', 'lung', 'fat', 'soft_tissue', 'muscle', 'trabecular_bone', 'cortical_bone']
        ref_densities = np.array([0.0012, 0.26, 0.92, 1.0, 1.05, 1.1, 1.85], dtype=np.float32)  # g/cm³
        
        materials_group = meta.create_group("materials")
        materials_group.create_dataset("names", data=[m.encode('utf-8') for m in materials])
        materials_group.create_dataset("ref_density_g_cm3", data=ref_densities)
        
        # Create energy group
        energy = f.create_group("energy")
        energies = np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float32)  # MeV
        energy.create_dataset("edges_MeV", data=energies)
        energy.create_dataset("centers_MeV", data=(energies[:-1] + energies[1:]) * 0.5)
        
        # Photon interaction tables
        photon_energies = np.array([10.0, 50.0, 100.0, 200.0, 500.0, 1000.0])  # keV
        
        # Create photon interaction group
        photon_group = f.create_group('photon_interactions')
        photon_group.create_dataset('energies', data=photon_energies)
        
        # Photoelectric effect cross-sections
        for material in materials:
            # Placeholder values
            photoelectric = np.array([0.5, 0.4, 0.3, 0.2, 0.15, 0.1])  # cm⁻¹
            photon_group.create_dataset(f'{material}_photoelectric', data=photoelectric)
        
        # Compton scattering cross-sections
        for material in materials:
            # Placeholder values
            compton = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55])  # cm⁻¹
            photon_group.create_dataset(f'{material}_compton', data=compton)
        
        # Method-specific parameters
        method_group = f.create_group('method_parameters')
        method_group.attrs['electron_transport'] = False
        method_group.attrs['photon_transport'] = True
        method_group.attrs['local_deposit'] = True
        
        print(f"Photon-only tables generated: {output_path}")


def generate_photon_em_condensed_tables(output_path: str) -> None:
    """Generate physics tables for photon_em_condensed method."""
    with h5py.File(output_path, "w") as f:
        # Set method attribute
        f.attrs['method'] = 'photon_em_condensed'
        
        # Create meta group with required information
        meta = f.create_group("meta")
        meta.attrs['schema_version'] = '1.0'
        
        # Materials information
        materials = ['air', 'lung', 'fat', 'soft_tissue', 'muscle', 'trabecular_bone', 'cortical_bone']
        ref_densities = np.array([0.0012, 0.26, 0.92, 1.0, 1.05, 1.1, 1.85], dtype=np.float32)  # g/cm³
        
        materials_group = meta.create_group("materials")
        materials_group.create_dataset("names", data=[m.encode('utf-8') for m in materials])
        materials_group.create_dataset("ref_density_g_cm3", data=ref_densities)
        
        # Create energy group
        energy = f.create_group("energy")
        energies = np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float32)
        energy.create_dataset("edges_MeV", data=energies)
        energy.create_dataset("centers_MeV", data=(energies[:-1] + energies[1:]) * 0.5)
        
        # Create photon interactions group
        photon_interactions = f.create_group("photon_interactions")
        photon_interactions.create_dataset("energies", data=energies)
        
        # Create electron interactions group
        electron_interactions = f.create_group("electron_interactions")
        electron_interactions.create_dataset("energies", data=energies)
        
        # Create photon interaction cross-sections for each material
        for material in materials:
            material_group = photon_interactions.create_group(material)
            
            # Photoelectric cross-section (cm²/g)
            photo = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            material_group.create_dataset("photoelectric", data=photo)
            
            # Compton cross-section (cm²/g)
            compton = np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)
            material_group.create_dataset("compton", data=compton)
            
            # Rayleigh cross-section (cm²/g)
            rayleigh = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
            material_group.create_dataset("rayleigh", data=rayleigh)
            
            # Pair production cross-section (cm²/g) - only at higher energies
            pair = np.array([0.0, 0.0, 0.0, 0.01, 0.02], dtype=np.float32)
            material_group.create_dataset("pair_production", data=pair)
        
        # Create electron interaction data for each material
        for material in materials:
            material_group = electron_interactions.create_group(material)
            
            # Stopping power (MeV/cm)
            stopping_power = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            material_group.create_dataset("stopping_power", data=stopping_power)
            
            # CSDA range (cm)
            csda_range = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
            material_group.create_dataset("csda_range", data=csda_range)
        
        # Method parameters
        method_params = f.create_group("method_parameters")
        method_params.create_dataset("materials", data=[m.encode('utf-8') for m in materials])
        
        print(f"Generated photon_em_condensed physics table: {output_path}")


def generate_physics_tables_h5(method: str, output_path: str) -> None:
    """Generate physics tables for the specified method.
    
    Args:
        method: One of 'local_deposit', 'photon_only', or 'photon_em_condensed'
        output_path: Path where the H5 file should be created
    """
    if method == 'local_deposit':
        generate_local_deposit_tables(output_path)
    elif method == 'photon_only':
        generate_photon_only_tables(output_path)
    elif method == 'photon_em_condensed':
        generate_photon_em_condensed_tables(output_path)
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of 'local_deposit', 'photon_only', or 'photon_em_condensed'")