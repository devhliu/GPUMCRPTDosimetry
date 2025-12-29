from __future__ import annotations

import triton
import triton.language as tl

from .gpu_math import fast_sqrt_approx, fast_sin_cos_approx


@triton.jit
def rotate_dir_kernel(
    uz: tl.tensor, uy: tl.tensor, ux: tl.tensor,
    cos_t: tl.tensor, phi: tl.tensor
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    """
    Rotate direction vector by polar angle cos_t and azimuthal angle phi.
    
    Args:
        uz, uy, ux: Original direction cosines
        cos_t: Cosine of polar scattering angle
        phi: Azimuthal scattering angle
    
    Returns:
        (nuz, nuy, nux): Rotated direction cosines
    """
    sin_t = fast_sqrt_approx(1.0 - cos_t * cos_t)
    cos_phi, sin_phi = fast_sin_cos_approx(phi)

    uz2 = uz * uz
    uy2 = uy * uy
    ux2 = ux * ux

    denom = fast_sqrt_approx(1.0 - uz2)
    denom_safe = tl.maximum(denom, 1e-12)

    nuz = uz * cos_t + denom_safe * sin_t * cos_phi
    nuy = uy * cos_t + (sin_t / denom_safe) * (uy * uz * cos_phi - ux * sin_phi)
    nux = ux * cos_t + (sin_t / denom_safe) * (ux * uz * cos_phi + uy * sin_phi)

    return nuz, nuy, nux
