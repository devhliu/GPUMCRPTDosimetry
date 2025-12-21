"""
Pre-computed data generation for GPU-accelerated Monte Carlo simulations.

This module handles the generation and management of pre-computed physics data
that can be used to accelerate Monte Carlo simulations on GPUs.
"""

import os
import h5py
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from ..materials.hu_materials import MaterialsLibrary, compute_material_effective_atom_Z
from ..materials.materials_registry import MaterialsTableConfig


@dataclass
class PrecomputedMaterialsData:
    """Container for pre-computed materials data."""
    material_names: List[str]
    effective_Z: torch.Tensor  # [M] effective atomic numbers
    electron_density_relative_to_water: torch.Tensor  # [M] relative electron density
    mass_attenuation_coefficients: Optional[torch.Tensor] = None  # [M, E] for energy bins
    stopping_powers: Optional[torch.Tensor] = None  # [M, E] for energy bins
    energy_bins: Optional[torch.Tensor] = None  # [E] energy values in MeV
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'material_names': self.material_names,
            'effective_Z': self.effective_Z.cpu().numpy(),
            'electron_density_relative_to_water': self.electron_density_relative_to_water.cpu().numpy(),
            'mass_attenuation_coefficients': (
                self.mass_attenuation_coefficients.cpu().numpy() 
                if self.mass_attenuation_coefficients is not None else None
            ),
            'stopping_powers': (
                self.stopping_powers.cpu().numpy() 
                if self.stopping_powers is not None else None
            ),
            'energy_bins': (
                self.energy_bins.cpu().numpy() 
                if self.energy_bins is not None else None
            ),
        }
    
    @classmethod
    def from_dict(cls, data_dict: Dict, device: str = "cuda") -> 'PrecomputedMaterialsData':
        """Create from dictionary."""
        device = torch.device(device)
        
        return cls(
            material_names=data_dict['material_names'],
            effective_Z=torch.tensor(data_dict['effective_Z'], device=device, dtype=torch.float32),
            electron_density_relative_to_water=torch.tensor(
                data_dict['electron_density_relative_to_water'], 
                device=device, 
                dtype=torch.float32
            ),
            mass_attenuation_coefficients=(
                torch.tensor(data_dict['mass_attenuation_coefficients'], device=device, dtype=torch.float32)
                if data_dict.get('mass_attenuation_coefficients') is not None else None
            ),
            stopping_powers=(
                torch.tensor(data_dict['stopping_powers'], device=device, dtype=torch.float32)
                if data_dict.get('stopping_powers') is not None else None
            ),
            energy_bins=(
                torch.tensor(data_dict['energy_bins'], device=device, dtype=torch.float32)
                if data_dict.get('energy_bins') is not None else None
            ),
        )


class PrecomputedDataGenerator:
    """Generator for pre-computed physics data."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._get_default_data_dir()
        self._ensure_data_dir()
    
    def _get_default_data_dir(self) -> str:
        """Get the default data directory."""
        package_dir = Path(__file__).parent.parent
        return str(package_dir / "data" / "precomputed")
    
    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_basic_materials_data(
        self, 
        materials_library: MaterialsLibrary,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Generate basic materials data (effective Z, electron density)."""
        
        device = torch.device(device)
        
        # Compute effective atomic numbers
        effective_Z = compute_material_effective_atom_Z(materials_library)
        
        # Compute electron density relative to water
        electron_density_relative_to_water = self._compute_relative_electron_density(
            materials_library
        )
        
        return PrecomputedMaterialsData(
            material_names=materials_library.material_names,
            effective_Z=effective_Z.to(device),
            electron_density_relative_to_water=electron_density_relative_to_water.to(device)
        )
    
    def generate_physics_tables(
        self,
        materials_library: MaterialsLibrary,
        energy_range: Tuple[float, float] = (0.01, 20.0),  # MeV
        num_energy_bins: int = 1000,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Generate comprehensive physics tables including attenuation and stopping powers."""
        
        device = torch.device(device)
        
        # Generate basic materials data
        basic_data = self.generate_basic_materials_data(materials_library, device)
        
        # Generate energy bins
        energy_bins = torch.logspace(
            np.log10(energy_range[0]), 
            np.log10(energy_range[1]), 
            num_energy_bins,
            device=device,
            dtype=torch.float32
        )
        
        # Generate mass attenuation coefficients (simplified model)
        mass_attenuation_coefficients = self._generate_mass_attenuation_coefficients(
            materials_library, energy_bins
        )
        
        # Generate stopping powers (simplified model)
        stopping_powers = self._generate_stopping_powers(materials_library, energy_bins)
        
        # Create new PrecomputedMaterialsData with all fields properly set
        physics_data = PrecomputedMaterialsData(
            material_names=basic_data.material_names,
            effective_Z=basic_data.effective_Z,
            electron_density_relative_to_water=basic_data.electron_density_relative_to_water,
            mass_attenuation_coefficients=mass_attenuation_coefficients,
            stopping_powers=stopping_powers,
            energy_bins=energy_bins
        )
        
        return physics_data
    
    def _compute_relative_electron_density(self, lib: MaterialsLibrary) -> torch.Tensor:
        """Compute electron density relative to water."""
        
        # Atomic numbers and approximate atomic masses
        Z = lib.element_Z.to(dtype=torch.float32)
        
        # Approximate atomic masses (g/mol)
        atomic_masses = torch.tensor([
            1.008,   # H
            12.011,  # C
            14.007,  # N
            15.999,  # O
            30.974,  # P
            40.078,  # Ca
            39.948   # Ar
        ], device=Z.device, dtype=torch.float32)
        
        # Electron density per element (electrons per gram)
        electrons_per_gram = Z / atomic_masses
        
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
    
    def _generate_mass_attenuation_coefficients(
        self, 
        lib: MaterialsLibrary, 
        energy_bins: torch.Tensor
    ) -> torch.Tensor:
        """Generate simplified mass attenuation coefficients."""
        
        # Simplified model: attenuation ~ Z^4 / E^3 (approximate for photoelectric effect)
        # This is a placeholder - real implementations would use NIST data or similar
        
        num_materials = len(lib.material_names)
        num_energies = len(energy_bins)
        
        coefficients = torch.zeros((num_materials, num_energies), 
                                 device=energy_bins.device, dtype=torch.float32)
        
        for i in range(num_materials):
            # Get effective Z for this material
            Z_eff = compute_material_effective_atom_Z(lib)[i].float()
            
            # Simplified photoelectric attenuation model
            # μ/ρ ~ Z^4 / E^3 (cm²/g)
            attenuation = (Z_eff ** 4) / (energy_bins ** 3)
            
            # Add Compton scattering component (approximately constant with Z)
            compton_component = 0.05 / (energy_bins + 0.1)
            
            coefficients[i] = attenuation + compton_component
        
        # Normalize and clamp to reasonable values
        coefficients = torch.clamp(coefficients, min=1e-6, max=100.0)
        
        return coefficients
    
    def _generate_stopping_powers(
        self, 
        lib: MaterialsLibrary, 
        energy_bins: torch.Tensor
    ) -> torch.Tensor:
        """Generate simplified stopping powers."""
        
        # Simplified model: stopping power ~ Z * log(E) / E
        # This is a placeholder - real implementations would use Bethe-Bloch formula
        
        num_materials = len(lib.material_names)
        num_energies = len(energy_bins)
        
        stopping_powers = torch.zeros((num_materials, num_energies), 
                                    device=energy_bins.device, dtype=torch.float32)
        
        for i in range(num_materials):
            # Get effective Z for this material
            Z_eff = compute_material_effective_atom_Z(lib)[i].float()
            
            # Simplified stopping power model (MeV/cm)
            # Based on approximate Bethe-Bloch behavior
            log_energy = torch.log(energy_bins + 1e-6)
            stopping_power = Z_eff * log_energy / (energy_bins + 0.1)
            
            # Scale to reasonable values (MeV/cm)
            stopping_powers[i] = stopping_power * 2.0
        
        # Clamp to reasonable values
        stopping_powers = torch.clamp(stopping_powers, min=1e-3, max=100.0)
        
        return stopping_powers
    
    def save_to_hdf5(
        self, 
        data: PrecomputedMaterialsData, 
        file_path: str,
        table_name: str = "materials_data"
    ) -> None:
        """Save pre-computed data to HDF5 file."""
        
        data_dict = data.to_dict()
        
        with h5py.File(file_path, 'w') as f:
            # Create group for this table
            group = f.create_group(table_name)
            
            # Save basic data
            group.create_dataset('material_names', data=np.array(data_dict['material_names'], dtype='S'))
            group.create_dataset('effective_Z', data=data_dict['effective_Z'])
            group.create_dataset('electron_density_relative_to_water', 
                               data=data_dict['electron_density_relative_to_water'])
            
            # Save physics data if available
            if data_dict['mass_attenuation_coefficients'] is not None:
                group.create_dataset('mass_attenuation_coefficients', 
                                   data=data_dict['mass_attenuation_coefficients'])
            
            if data_dict['stopping_powers'] is not None:
                group.create_dataset('stopping_powers', data=data_dict['stopping_powers'])
            
            if data_dict['energy_bins'] is not None:
                group.create_dataset('energy_bins', data=data_dict['energy_bins'])
            
            # Save metadata
            group.attrs['num_materials'] = len(data_dict['material_names'])
            group.attrs['has_physics_data'] = data_dict['energy_bins'] is not None
            
            if data_dict['energy_bins'] is not None:
                group.attrs['num_energy_bins'] = len(data_dict['energy_bins'])
                group.attrs['energy_range_min'] = data_dict['energy_bins'][0]
                group.attrs['energy_range_max'] = data_dict['energy_bins'][-1]
    
    def load_from_hdf5(
        self, 
        file_path: str, 
        table_name: str = "materials_data",
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Load pre-computed data from HDF5 file."""
        
        with h5py.File(file_path, 'r') as f:
            group = f[table_name]
            
            # Load basic data
            material_names = [name.decode('utf-8') for name in group['material_names'][:]]
            effective_Z = group['effective_Z'][:]
            electron_density_relative_to_water = group['electron_density_relative_to_water'][:]
            
            # Load physics data if available
            mass_attenuation_coefficients = (
                group['mass_attenuation_coefficients'][:] 
                if 'mass_attenuation_coefficients' in group else None
            )
            
            stopping_powers = (
                group['stopping_powers'][:] 
                if 'stopping_powers' in group else None
            )
            
            energy_bins = group['energy_bins'][:] if 'energy_bins' in group else None
            
            data_dict = {
                'material_names': material_names,
                'effective_Z': effective_Z,
                'electron_density_relative_to_water': electron_density_relative_to_water,
                'mass_attenuation_coefficients': mass_attenuation_coefficients,
                'stopping_powers': stopping_powers,
                'energy_bins': energy_bins
            }
        
        return PrecomputedMaterialsData.from_dict(data_dict, device)
    
    def get_precomputed_file_path(
        self, 
        table_name: str, 
        include_physics: bool = False
    ) -> str:
        """Get the file path for pre-computed data."""
        
        suffix = "_physics.h5" if include_physics else "_basic.h5"
        filename = f"{table_name}{suffix}"
        return os.path.join(self.data_dir, filename)
    
    def precompute_and_save(
        self,
        materials_config: MaterialsTableConfig,
        include_physics: bool = False,
        device: str = "cuda"
    ) -> str:
        """Pre-compute and save data for a materials table."""
        
        # Import the function to build materials library
        from .hu_materials import build_materials_library_from_config
        
        # Build materials library from config
        materials_library = build_materials_library_from_config(
            cfg={"material_library": materials_config.material_library},
            device=device
        )
        
        # Generate data
        if include_physics:
            data = self.generate_physics_tables(materials_library, device=device)
        else:
            data = self.generate_basic_materials_data(materials_library, device=device)
        
        # Save to file
        file_path = self.get_precomputed_file_path(
            materials_config.name, 
            include_physics=include_physics
        )
        
        self.save_to_hdf5(data, file_path, table_name=materials_config.name)
        
        return file_path