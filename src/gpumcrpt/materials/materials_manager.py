"""
Materials management system for GPU-accelerated Monte Carlo dosimetry.

This module provides a unified interface for managing HU materials tables
and generating pre-computed data for GPU acceleration.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import h5py
import numpy as np

from .hu_materials import (
    MaterialsLibrary, 
    MaterialsVolume, 
    build_materials_from_hu,
    build_materials_library_from_config,
    compute_material_effective_atom_Z
)
from .materials_registry import MaterialsRegistry, MaterialsTableConfig, get_default_registry
from .precomputed_data import PrecomputedMaterialsData, PrecomputedDataGenerator


class MaterialsManager:
    """Main class for managing HU materials and pre-computed data."""
    
    def __init__(self, registry: Optional[MaterialsRegistry] = None):
        self.registry = registry or get_default_registry()
        self._precomputed_data: Dict[str, PrecomputedMaterialsData] = {}
        self._data_generator = PrecomputedDataGenerator()
    
    def get_materials_volume(
        self, 
        hu_volume: torch.Tensor, 
        table_name: Optional[str] = None,
        device: str = "cuda"
    ) -> MaterialsVolume:
        """Convert HU volume to materials volume using the specified table."""
        
        table_config = (self.registry.get_table(table_name) 
                       if table_name else self.registry.get_default_table())
        
        # Build materials library
        materials_lib = build_materials_library_from_config(
            cfg={"material_library": table_config.material_library},
            device=device
        )
        
        # Convert HU to materials
        materials_volume = build_materials_from_hu(
            hu=hu_volume,
            hu_to_density=table_config.hu_to_density,
            hu_to_class=table_config.hu_to_class,
            material_library=materials_lib,
            device=device
        )
        
        return materials_volume
    
    def precompute_basic_materials_data(
        self, 
        table_name: Optional[str] = None,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Pre-compute basic materials data (effective Z, electron density)."""
        
        if table_name in self._precomputed_data:
            return self._precomputed_data[table_name]
        
        table_config = (self.registry.get_table(table_name) 
                       if table_name else self.registry.get_default_table())
        
        # Build materials library
        materials_lib = build_materials_library_from_config(
            cfg={"material_library": table_config.material_library},
            device=device
        )
        
        # Generate basic materials data
        precomputed_data = self._data_generator.generate_basic_materials_data(
            materials_lib, device=device
        )
        
        # Cache the result
        self._precomputed_data[table_name or "default"] = precomputed_data
        
        return precomputed_data
    
    def precompute_physics_tables(
        self,
        table_name: Optional[str] = None,
        energy_range: Tuple[float, float] = (0.01, 20.0),
        num_energy_bins: int = 1000,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Pre-compute comprehensive physics tables including attenuation and stopping powers."""
        
        if table_name in self._precomputed_data:
            return self._precomputed_data[table_name]
        
        table_config = (self.registry.get_table(table_name) 
                       if table_name else self.registry.get_default_table())
        
        # Build materials library
        materials_lib = build_materials_library_from_config(
            cfg={"material_library": table_config.material_library},
            device=device
        )
        
        # Generate physics tables
        precomputed_data = self._data_generator.generate_physics_tables(
            materials_lib, energy_range, num_energy_bins, device=device
        )
        
        # Cache the result
        self._precomputed_data[table_name or "default"] = precomputed_data
        
        return precomputed_data
    
    def save_precomputed_data(
        self, 
        table_name: str, 
        file_path: str,
        include_physics: bool = False,
        device: str = "cuda"
    ) -> str:
        """Save pre-computed data to HDF5 file."""
        
        config = self.registry.get_table(table_name)
        
        # Pre-compute and save data
        return self._data_generator.precompute_and_save(
            config, include_physics=include_physics, device=device
        )
    
    def load_precomputed_data(
        self, 
        file_path: str,
        table_name: str,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Load pre-computed data from HDF5 file."""
        
        # Load data from HDF5
        precomputed_data = self._data_generator.load_from_hdf5(
            file_path, table_name=table_name, device=device
        )
        
        # Cache the result
        self._precomputed_data[table_name] = precomputed_data
        
        return precomputed_data
    
    def get_available_tables(self) -> List[str]:
        """Get list of available materials tables."""
        return self.registry.list_tables()
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get information about a specific materials table."""
        config = self.registry.get_table(table_name)
        return {
            'name': config.name,
            'description': config.description,
            'source': config.source,
            'is_default': config.is_default,
            'num_materials': len(config.material_library['materials']),
            'num_elements': len(config.material_library['elements'])
        }


def get_default_materials_manager() -> MaterialsManager:
    """Get the default materials manager with built-in tables."""
    from .materials_registry import get_default_registry
    return MaterialsManager(get_default_registry())