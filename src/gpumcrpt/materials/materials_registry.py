from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .hu_materials import MaterialsLibrary, build_materials_library_from_config


@dataclass
class MaterialsTableConfig:
    """Configuration for a single HU materials table."""
    name: str
    description: str
    source: str  # e.g., "ICRU-44", "Schneider", "Custom"
    file_path: str
    
    # HU to density mapping
    hu_to_density: List[List[float]]
    
    # HU to material class mapping
    hu_to_class: List[List[float]]
    
    # Material library configuration
    material_library: Dict
    
    is_default: bool = False


class MaterialsRegistry:
    """Registry for managing multiple HU materials tables."""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._get_default_data_dir()
        self.tables: Dict[str, MaterialsTableConfig] = {}
        self.default_table: Optional[str] = None
        self._load_builtin_tables()
    
    def _get_default_data_dir(self) -> str:
        """Get the default data directory for built-in materials tables."""
        return str(Path(__file__).parent / "data")
    
    def _load_builtin_tables(self) -> None:
        """Load built-in materials tables from the data directory."""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            return
            
        for yaml_file in data_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                table_config = MaterialsTableConfig(
                    name=config_data['name'],
                    description=config_data.get('description', ''),
                    source=config_data.get('source', 'Custom'),
                    file_path=str(yaml_file),
                    is_default=config_data.get('is_default', False),
                    hu_to_density=config_data['hu_to_density'],
                    hu_to_class=config_data['hu_to_class'],
                    material_library=config_data['material_library']
                )
                
                self.tables[table_config.name] = table_config
                
                if table_config.is_default:
                    self.default_table = table_config.name
                    
            except Exception as e:
                print(f"Warning: Failed to load materials table from {yaml_file}: {e}")
    
    def register_table(self, config: MaterialsTableConfig) -> None:
        """Register a new materials table."""
        self.tables[config.name] = config
        if config.is_default:
            self.default_table = config.name
    
    def load_table_from_file(self, file_path: str, name: Optional[str] = None) -> str:
        """Load a materials table from a YAML file."""
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        table_name = name or config_data.get('name', Path(file_path).stem)
        
        table_config = MaterialsTableConfig(
            name=table_name,
            description=config_data.get('description', ''),
            source=config_data.get('source', 'Custom'),
            file_path=file_path,
            is_default=config_data.get('is_default', False),
            hu_to_density=config_data['hu_to_density'],
            hu_to_class=config_data['hu_to_class'],
            material_library=config_data['material_library']
        )
        
        self.register_table(table_config)
        return table_name
    
    def get_table(self, name: str) -> MaterialsTableConfig:
        """Get a materials table configuration by name."""
        if name not in self.tables:
            raise KeyError(f"Materials table '{name}' not found. "
                          f"Available tables: {list(self.tables.keys())}")
        return self.tables[name]
    
    def get_default_table(self) -> MaterialsTableConfig:
        """Get the default materials table."""
        if not self.default_table:
            raise ValueError("No default materials table configured")
        return self.tables[self.default_table]
    
    def list_tables(self) -> List[str]:
        """List all available materials tables."""
        return list(self.tables.keys())
    
    def get_table_info(self, name: str) -> Dict:
        """Get information about a materials table."""
        config = self.get_table(name)
        return {
            'name': config.name,
            'description': config.description,
            'source': config.source,
            'file_path': config.file_path,
            'is_default': config.is_default,
            'material_count': len(config.material_library.get('materials', [])),
            'element_count': len(config.material_library.get('elements', [])),
            'hu_ranges': len(config.hu_to_class)
        }


def get_default_registry() -> MaterialsRegistry:
    """Get the default materials registry with built-in tables."""
    return MaterialsRegistry()