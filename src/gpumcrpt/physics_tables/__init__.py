
"""
GPUMCRPTDosimetry Physics Tables Submodule
=========================================

This submodule handles the loading and management of pre-computed physics tables
used by the transport engines.
"""

from .tables import PhysicsTables, load_physics_tables_h5
from .relaxation_tables import RelaxationTables

__all__ = [
    'PhysicsTables',
    'load_physics_tables_h5',
    'RelaxationTables'
]