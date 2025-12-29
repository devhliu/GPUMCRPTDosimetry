"""
High-Performance Unified Charged Particle Transport Module

This module provides optimized kernel implementations for both electrons
and positrons with maximum GPU performance through vectorized operations
and unified memory layout.

Features:
- Unified kernels for both electrons and positrons
- Structure of Arrays (SoA) layout for optimal GPU performance
- Vectorized physics calculations
- Minimal branch divergence
- Efficient secondary particle handling
"""

from .step import (
    charged_particle_step_kernel,
    _apply_common_charged_particle_physics,
    sample_multiple_scattering_angle,
    rotate_vector_around_axis,
)
from .emission import (
    charged_particle_brems_emit_kernel,
    charged_particle_delta_emit_kernel,
    sample_bremsstrahlung_direction,
    sample_delta_ray_direction,
    rotate_direction_to_frame,
    positron_annihilation_at_rest_kernel,
)

# Modern unified kernels for high-performance charged particle transport

__all__ = [
    # High-performance unified kernels
    "charged_particle_step_kernel",
    "charged_particle_brems_emit_kernel",
    "charged_particle_delta_emit_kernel",
    "positron_annihilation_at_rest_kernel",

    # Utility functions
    "_apply_common_charged_particle_physics",
    "sample_multiple_scattering_angle",
    "rotate_vector_around_axis",
    "sample_bremsstrahlung_direction",
    "sample_delta_ray_direction",
    "rotate_direction_to_frame",
]