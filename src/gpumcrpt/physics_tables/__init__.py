"""
Physics tables module for GPU-accelerated Monte Carlo dosimetry.

This module provides functionality for managing pre-computed physics data
including attenuation coefficients, stopping powers, and other physics tables
for GPU-accelerated Monte Carlo simulations.
"""

from .precomputed_data import PrecomputedMaterialsData, PrecomputedDataGenerator
from .physics_tables_manager import PhysicsTablesManager, get_default_physics_tables_manager

__all__ = [
    'PrecomputedMaterialsData',
    'PrecomputedDataGenerator', 
    'PhysicsTablesManager',
    'get_default_physics_tables_manager'
]