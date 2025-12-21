"""
Physics tables management system for GPU-accelerated Monte Carlo dosimetry.

This module provides a unified interface for managing pre-computed physics data
including attenuation coefficients, stopping powers, and other physics tables
for GPU-accelerated Monte Carlo simulations.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import h5py
import numpy as np

from .precomputed_data import PrecomputedMaterialsData, PrecomputedDataGenerator
from ..materials.materials_registry import MaterialsTableConfig
from ..materials.hu_materials import MaterialsLibrary, build_materials_library_from_config


class PhysicsTablesManager:
    """Main class for managing pre-computed physics data for Monte Carlo simulations."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._get_default_data_dir()
        self._precomputed_data: Dict[str, PrecomputedMaterialsData] = {}
        self._data_generator = PrecomputedDataGenerator(data_dir=self.data_dir)
    
    def _get_default_data_dir(self) -> str:
        """Get the default data directory for pre-computed physics data."""
        return str(Path(__file__).parent / "precomputed_tables")
    
    def precompute_basic_materials_data(
        self, 
        materials_library: MaterialsLibrary,
        table_name: Optional[str] = None,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Pre-compute basic materials data (effective Z, electron density)."""
        
        cache_key = table_name or "default"
        if cache_key in self._precomputed_data:
            return self._precomputed_data[cache_key]
        
        # Generate basic materials data
        precomputed_data = self._data_generator.generate_basic_materials_data(
            materials_library, device=device
        )
        
        # Cache the result
        self._precomputed_data[cache_key] = precomputed_data
        
        return precomputed_data
    
    def precompute_physics_tables(
        self,
        materials_library: MaterialsLibrary,
        table_name: Optional[str] = None,
        energy_range: Tuple[float, float] = (0.01, 20.0),
        num_energy_bins: int = 1000,
        device: str = "cuda"
    ) -> PrecomputedMaterialsData:
        """Pre-compute comprehensive physics tables including attenuation and stopping powers."""
        
        cache_key = table_name or "default"
        if cache_key in self._precomputed_data:
            return self._precomputed_data[cache_key]
        
        # Generate physics tables
        precomputed_data = self._data_generator.generate_physics_tables(
            materials_library, energy_range, num_energy_bins, device=device
        )
        
        # Cache the result
        self._precomputed_data[cache_key] = precomputed_data
        
        return precomputed_data
    
    def save_precomputed_data(
        self, 
        materials_library: MaterialsLibrary,
        table_name: str,
        file_path: Optional[str] = None,
        include_physics: bool = False,
        device: str = "cuda"
    ) -> str:
        """Save pre-computed data to HDF5 file."""
        
        if file_path is None:
            file_path = str(Path(self.data_dir) / f"{table_name}_precomputed.h5")
        
        # Pre-compute and save data
        return self._data_generator.precompute_and_save(
            materials_library, 
            file_path=file_path,
            include_physics=include_physics, 
            device=device
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
    
    def get_precomputed_data(self, table_name: str) -> Optional[PrecomputedMaterialsData]:
        """Get cached pre-computed data for a table."""
        return self._precomputed_data.get(table_name)
    
    def clear_cache(self, table_name: Optional[str] = None) -> None:
        """Clear cached pre-computed data."""
        if table_name:
            self._precomputed_data.pop(table_name, None)
        else:
            self._precomputed_data.clear()


def get_default_physics_tables_manager() -> PhysicsTablesManager:
    """Get the default physics tables manager."""
    return PhysicsTablesManager()