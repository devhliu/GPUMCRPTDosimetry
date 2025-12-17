from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpumcrpt.io.nifti import NiftiVolume


@dataclass
class PhantomVolumes:
    ct_hu: NiftiVolume
    activity_bqs: NiftiVolume


@dataclass
class NemaIecBodyPhantomVolumes(PhantomVolumes):
    """NEMA IEC body phantom volumes plus an optional sphere label map.

    sphere_label: uint8 [Z,Y,X], 0=background, 1..6=spheres.
    """

    sphere_label: NiftiVolume | None = None


def _make_affine_mm(*, voxel_size_mm: tuple[float, float, float]) -> np.ndarray:
    vx, vy, vz = voxel_size_mm
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = float(vx)
    affine[1, 1] = float(vy)
    affine[2, 2] = float(vz)
    return affine


def make_water_slab_with_bone_cylinder(
    *,
    shape_zyx: tuple[int, int, int] = (64, 64, 64),
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    water_hu: float = 0.0,
    bone_hu: float = 1000.0,
    bone_radius_mm: float = 20.0,
    bone_center_yx: tuple[float, float] | None = None,
    activity_bq_per_voxel: float = 1.0,
    activity_sphere_radius_mm: float = 25.0,
    activity_center_zyx: tuple[float, float, float] | None = None,
    exclude_bone_from_activity: bool = True,
) -> PhantomVolumes:
    """Create a simple reference phantom for validation.

    Geometry
    - CT: water everywhere, with a bone cylinder along Z.
    - Activity: uniform inside a sphere (in water by default).

    Output units
    - ct_hu.data: HU
    - activity_bqs.data: Bq/s per voxel (for source sampling)

    Shapes are [Z, Y, X].
    """
    Z, Y, X = map(int, shape_zyx)
    vx_mm, vy_mm, vz_mm = map(float, voxel_size_mm)

    ct = np.full((Z, Y, X), float(water_hu), dtype=np.float32)

    if bone_center_yx is None:
        cy = 0.5 * (Y - 1)
        cx = 0.5 * (X - 1)
    else:
        cy, cx = map(float, bone_center_yx)

    yy, xx = np.meshgrid(np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32), indexing="ij")
    dy_mm = (yy - cy) * vy_mm
    dx_mm = (xx - cx) * vx_mm
    r_mm = np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)
    bone_mask_yx = r_mm <= float(bone_radius_mm)

    # broadcast along Z
    ct[:, bone_mask_yx] = float(bone_hu)

    # Activity sphere
    if activity_center_zyx is None:
        cz = 0.5 * (Z - 1)
        cy_a = 0.5 * (Y - 1)
        cx_a = 0.5 * (X - 1)
    else:
        cz, cy_a, cx_a = map(float, activity_center_zyx)

    zz, yy3, xx3 = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(Y, dtype=np.float32),
        np.arange(X, dtype=np.float32),
        indexing="ij",
    )
    dz_mm = (zz - cz) * vz_mm
    dy_mm3 = (yy3 - cy_a) * vy_mm
    dx_mm3 = (xx3 - cx_a) * vx_mm
    rr_mm = np.sqrt(dx_mm3 * dx_mm3 + dy_mm3 * dy_mm3 + dz_mm * dz_mm)
    sphere_mask = rr_mm <= float(activity_sphere_radius_mm)

    act = np.zeros((Z, Y, X), dtype=np.float32)
    act[sphere_mask] = float(activity_bq_per_voxel)

    if exclude_bone_from_activity:
        act[ct >= float(bone_hu) * 0.5] = 0.0

    affine = _make_affine_mm(voxel_size_mm=voxel_size_mm)

    # NiftiVolume expects affine in mm, and voxel_size_cm computed on load. Here we populate both.
    vx_cm, vy_cm, vz_cm = vx_mm / 10.0, vy_mm / 10.0, vz_mm / 10.0
    voxel_vol_cm3 = float(vx_cm * vy_cm * vz_cm)

    ct_vol = NiftiVolume(data=ct, affine=affine, voxel_size_cm=(vx_cm, vy_cm, vz_cm), voxel_volume_cm3=voxel_vol_cm3)
    act_vol = NiftiVolume(data=act, affine=affine, voxel_size_cm=(vx_cm, vy_cm, vz_cm), voxel_volume_cm3=voxel_vol_cm3)

    return PhantomVolumes(ct_hu=ct_vol, activity_bqs=act_vol)


def _make_nifti_volume_from_array(*, data: np.ndarray, affine: np.ndarray, voxel_size_mm: tuple[float, float, float]) -> NiftiVolume:
    vx_mm, vy_mm, vz_mm = map(float, voxel_size_mm)
    vx_cm, vy_cm, vz_cm = vx_mm / 10.0, vy_mm / 10.0, vz_mm / 10.0
    voxel_vol_cm3 = float(vx_cm * vy_cm * vz_cm)
    return NiftiVolume(
        data=np.asarray(data, dtype=np.float32),
        affine=np.asarray(affine, dtype=np.float32),
        voxel_size_cm=(vx_cm, vy_cm, vz_cm),
        voxel_volume_cm3=voxel_vol_cm3,
    )


def make_nema_iec_body_phantom(
    *,
    shape_zyx: tuple[int, int, int] = (96, 128, 128),
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    body_cylinder_radius_mm: float = 100.0,
    body_length_mm: float | None = None,
    water_hu: float = 0.0,
    air_hu: float = -1000.0,
    sphere_diameters_mm: tuple[float, float, float, float, float, float] = (10.0, 13.0, 17.0, 22.0, 28.0, 37.0),
    sphere_ring_radius_mm: float = 57.5,
    sphere_plane_offset_mm: float = 0.0,
    background_bq_per_voxel: float = 1.0,
    hot_sphere_ratio: float = 4.0,
    include_sphere_label: bool = True,
) -> NemaIecBodyPhantomVolumes:
    """Generate a simplified NEMA IEC body phantom in NIfTI format.

    This is meant for validation plumbing and cross-code comparisons.
    - CT HU: air outside body cylinder, water inside.
    - Activity: background uniform inside cylinder, with 6 hot spheres at a fixed ratio.

    Notes
    - The IEC body phantom has standardized sphere diameters; default uses the common set.
    - The exact sphere center radius varies by vendor/assembly; expose it as a parameter.
    """
    Z, Y, X = map(int, shape_zyx)
    vx_mm, vy_mm, vz_mm = map(float, voxel_size_mm)

    affine = _make_affine_mm(voxel_size_mm=voxel_size_mm)

    cy = 0.5 * (Y - 1)
    cx = 0.5 * (X - 1)
    cz = 0.5 * (Z - 1)

    yy, xx = np.meshgrid(np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32), indexing="ij")
    dy_mm = (yy - cy) * vy_mm
    dx_mm = (xx - cx) * vx_mm
    r_mm = np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm)
    body_mask_yx = r_mm <= float(body_cylinder_radius_mm)

    # Body axial extent (optional): if provided, truncate cylinder along Z.
    if body_length_mm is None:
        body_mask = np.broadcast_to(body_mask_yx[None, :, :], (Z, Y, X)).copy()
    else:
        half_len = 0.5 * float(body_length_mm)
        zz = np.arange(Z, dtype=np.float32)
        dz_mm = (zz - cz) * vz_mm
        z_mask = np.abs(dz_mm) <= half_len
        body_mask = (z_mask[:, None, None] & body_mask_yx[None, :, :]).copy()

    ct = np.full((Z, Y, X), float(air_hu), dtype=np.float32)
    ct[body_mask] = float(water_hu)

    act = np.zeros((Z, Y, X), dtype=np.float32)
    act[body_mask] = float(background_bq_per_voxel)

    if include_sphere_label:
        sphere_label = np.zeros((Z, Y, X), dtype=np.uint8)
    else:
        sphere_label = None

    # Sphere centers on a ring in a single axial plane
    z0 = cz + float(sphere_plane_offset_mm) / vz_mm
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300], dtype=np.float32))
    centers_y = cy + (float(sphere_ring_radius_mm) / vy_mm) * np.sin(angles)
    centers_x = cx + (float(sphere_ring_radius_mm) / vx_mm) * np.cos(angles)

    zz3, yy3, xx3 = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(Y, dtype=np.float32),
        np.arange(X, dtype=np.float32),
        indexing="ij",
    )

    for i, d_mm in enumerate(sphere_diameters_mm, start=1):
        rr = 0.5 * float(d_mm)
        cy_i = float(centers_y[i - 1])
        cx_i = float(centers_x[i - 1])
        dz_mm = (zz3 - float(z0)) * vz_mm
        dy_mm = (yy3 - cy_i) * vy_mm
        dx_mm = (xx3 - cx_i) * vx_mm
        dist_mm = np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm)
        sph = (dist_mm <= rr) & body_mask

        # Hot spheres override background
        act[sph] = float(background_bq_per_voxel) * float(hot_sphere_ratio)
        if sphere_label is not None:
            sphere_label[sph] = np.uint8(i)

    ct_vol = _make_nifti_volume_from_array(data=ct, affine=affine, voxel_size_mm=voxel_size_mm)
    act_vol = _make_nifti_volume_from_array(data=act, affine=affine, voxel_size_mm=voxel_size_mm)

    if sphere_label is not None:
        sphere_label_vol = _make_nifti_volume_from_array(data=sphere_label.astype(np.float32), affine=affine, voxel_size_mm=voxel_size_mm)
    else:
        sphere_label_vol = None

    return NemaIecBodyPhantomVolumes(ct_hu=ct_vol, activity_bqs=act_vol, sphere_label=sphere_label_vol)
