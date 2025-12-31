"""
Materials management system for GPU-accelerated Monte Carlo dosimetry.

This module provides a unified interface for managing HU materials tables
and converting HU volumes to materials volumes for GPU acceleration.
"""

from typing import Dict, List, Optional

import torch

from gpumcrpt.materials.hu_materials import (
    MaterialsVolume, 
    build_materials_from_hu,
    build_materials_library_from_config
)
from gpumcrpt.materials.materials_registry import MaterialsRegistry, get_default_registry


class MaterialsManager:
    """Main class for managing HU materials tables and volume conversion."""
    
    def __init__(self, registry: Optional[MaterialsRegistry] = None):
        self.registry = registry or get_default_registry()
    
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
    
    def get_table_config(self, table_name: Optional[str] = None):
        """Get the configuration for a specific materials table."""
        return (self.registry.get_table(table_name) 
                if table_name else self.registry.get_default_table())


def get_default_materials_manager() -> MaterialsManager:
    """Get the default materials manager with built-in tables."""
    from .materials_registry import get_default_registry
    return MaterialsManager(get_default_registry())