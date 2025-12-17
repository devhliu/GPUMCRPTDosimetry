from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class NiftiVolume:
    data: np.ndarray
    affine: np.ndarray
    voxel_size_cm: tuple[float, float, float]
    voxel_volume_cm3: float


def load_nifti(path: str) -> NiftiVolume:
    import nibabel as nib

    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)

    zooms_mm = img.header.get_zooms()[:3]
    vx_cm, vy_cm, vz_cm = (float(zooms_mm[0]) / 10.0, float(zooms_mm[1]) / 10.0, float(zooms_mm[2]) / 10.0)
    voxel_volume_cm3 = vx_cm * vy_cm * vz_cm

    return NiftiVolume(
        data=np.asarray(data, dtype=np.float32),
        affine=affine,
        voxel_size_cm=(vx_cm, vy_cm, vz_cm),
        voxel_volume_cm3=float(voxel_volume_cm3),
    )


def save_nifti_like(reference: NiftiVolume, data: np.ndarray, out_path: str) -> None:
    import nibabel as nib

    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine=reference.affine)
    nib.save(img, out_path)


def resample_to_reference(*, moving: NiftiVolume, reference: NiftiVolume, mode: Literal["linear", "nearest"] = "linear") -> NiftiVolume:
    """Best-effort resampling using nibabel.processing if available.

    MVP behavior:
      - If shapes match, returns moving unchanged.
      - Otherwise tries nibabel.processing.resample_from_to (may require SciPy).
    """
    if moving.data.shape == reference.data.shape:
        return moving

    import nibabel as nib

    try:
        from nibabel.processing import resample_from_to
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Resampling requires nibabel.processing; please install nibabel[processing] / scipy") from e

    order = 1 if mode == "linear" else 0
    moving_img = nib.Nifti1Image(moving.data, affine=moving.affine)
    ref_img = nib.Nifti1Image(reference.data, affine=reference.affine)

    out_img = resample_from_to(moving_img, ref_img, order=order)
    out = out_img.get_fdata(dtype=np.float32)

    # Keep reference spacing
    return NiftiVolume(
        data=np.asarray(out, dtype=np.float32),
        affine=np.asarray(out_img.affine, dtype=np.float32),
        voxel_size_cm=reference.voxel_size_cm,
        voxel_volume_cm3=reference.voxel_volume_cm3,
    )
