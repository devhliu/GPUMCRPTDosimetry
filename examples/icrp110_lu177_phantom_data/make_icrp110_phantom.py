#!/usr/bin/env python
"""
Generate a simplified ICRP110-inspired adult reference phantom.

ICRP110 is the Reference Computational Phantom for anatomical reference.
This is a simplified version with multiple tissue types:
- Lung tissue
- Bone (trabecular and cortical)
- Soft tissue (muscle, organs, etc.)
- Fat tissue
- Blood

The phantom includes:
- Torso region with lungs and heart
- Vertebral column (bone)
- Ribs (bone)
- Pelvis region
- Organs (simplified as ellipsoids)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from gpumcrpt.io.nifti import NiftiVolume, save_nifti_like


def _make_affine_mm(*, voxel_size_mm: tuple[float, float, float]) -> np.ndarray:
    vx, vy, vz = voxel_size_mm
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = float(vx)
    affine[1, 1] = float(vy)
    affine[2, 2] = float(vz)
    return affine


def _make_nifti_volume_from_array(
    *, data: np.ndarray, affine: np.ndarray, voxel_size_mm: tuple[float, float, float]
) -> NiftiVolume:
    vx_mm, vy_mm, vz_mm = map(float, voxel_size_mm)
    vx_cm, vy_cm, vz_cm = vx_mm / 10.0, vy_mm / 10.0, vz_mm / 10.0
    voxel_vol_cm3 = float(vx_cm * vy_cm * vz_cm)
    return NiftiVolume(
        data=np.asarray(data, dtype=np.float32),
        affine=np.asarray(affine, dtype=np.float32),
        voxel_size_cm=(vx_cm, vy_cm, vz_cm),
        voxel_volume_cm3=voxel_vol_cm3,
    )


def make_icrp110_phantom(
    *,
    shape_zyx: tuple[int, int, int] = (150, 160, 160),
    voxel_size_mm: tuple[float, float, float] = (2.5, 2.5, 2.5),
    background_tissue_hu: float = -950.0,  # Lung tissue HU
    soft_tissue_hu: float = 30.0,
    muscle_hu: float = 40.0,
    bone_hu: float = 500.0,  # Trabecular bone
    cortical_bone_hu: float = 1200.0,
    fat_hu: float = -100.0,
    background_activity_bq_per_voxel: float = 50.0,
    hot_organ_ratio: float = 3.0,
) -> tuple[NiftiVolume, NiftiVolume]:
    """Generate a simplified ICRP110-like reference phantom.

    This phantom includes:
    - Background lung tissue (low density)
    - Heart and mediastinum (soft tissue)
    - Vertebral column and ribs (bone)
    - Abdomen with liver, kidneys, spleen (soft tissue)
    - Pelvis region (bone + soft tissue)
    - Activity: uniform background with hot organs (liver, spleen, kidneys)

    Output units
    - CT: HU (Hounsfield Units)
    - Activity: Bq/voxel (for source sampling)

    Shapes are [Z, Y, X].
    """
    Z, Y, X = map(int, shape_zyx)
    vx_mm, vy_mm, vz_mm = map(float, voxel_size_mm)

    affine = _make_affine_mm(voxel_size_mm=voxel_size_mm)

    # Initialize CT with background (air/lung)
    ct = np.full((Z, Y, X), float(background_tissue_hu), dtype=np.float32)
    act = np.zeros((Z, Y, X), dtype=np.float32)

    cy = 0.5 * (Y - 1)
    cx = 0.5 * (X - 1)
    cz = 0.5 * (Z - 1)

    # Create body ellipsoid (outer boundary) - simplified as elliptical cylinder
    yy, xx = np.meshgrid(np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32), indexing="ij")
    dy_mm = (yy - cy) * vy_mm
    dx_mm = (xx - cx) * vx_mm

    # Body boundary: ellipse with semi-major axis ~110 mm (AP) and semi-minor axis ~95 mm (LR)
    body_ellipse_ap = 110.0  # anterior-posterior
    body_ellipse_lr = 95.0  # left-right
    body_mask_yx = (dx_mm / body_ellipse_lr) ** 2 + (dy_mm / body_ellipse_ap) ** 2 <= 1.0

    # Extend to full 3D body volume
    zz = np.arange(Z, dtype=np.float32)
    dz_mm = (zz - cz) * vz_mm

    # Body extends from z=-200mm to z=+250mm (relative to center)
    z_head_to_pelvis = 450.0  # mm
    z_max_extent = 0.5 * z_head_to_pelvis
    body_z_mask = np.abs(dz_mm) <= z_max_extent

    zz3, yy3, xx3 = np.meshgrid(
        np.arange(Z, dtype=np.float32),
        np.arange(Y, dtype=np.float32),
        np.arange(X, dtype=np.float32),
        indexing="ij",
    )
    dz_mm3 = (zz3 - cz) * vz_mm
    dy_mm3 = (yy3 - cy) * vy_mm
    dx_mm3 = (xx3 - cx) * vx_mm

    body_ellipse_3d = ((dx_mm3 / body_ellipse_lr) ** 2 + (dy_mm3 / body_ellipse_ap) ** 2) <= 1.0
    body_mask = body_ellipse_3d & (body_z_mask[:, None, None])

    # Fill body with soft tissue as default
    ct[body_mask] = float(soft_tissue_hu)
    act[body_mask] = float(background_activity_bq_per_voxel)

    # ===== LUNGS =====
    # Lungs occupy upper torso, extending from z ~ -180 to z ~ +80 mm
    lung_z_min = -180.0
    lung_z_max = 80.0
    lung_mask_z = (dz_mm3 >= lung_z_min) & (dz_mm3 <= lung_z_max)

    # Left lung: smaller ellipse on left side (x < 0)
    left_lung_x_radius = 55.0
    left_lung_y_radius = 90.0
    left_lung_center_x = -50.0
    dx_left = dx_mm3 - left_lung_x_radius * 0.3  # Shift left
    dy_left = dy_mm3
    left_lung = (
        (dx_left / left_lung_x_radius) ** 2 + (dy_left / left_lung_y_radius) ** 2 <= 1.0
    ) & lung_mask_z & body_mask

    # Right lung: larger ellipse on right side (x > 0)
    right_lung_x_radius = 60.0
    right_lung_y_radius = 90.0
    right_lung_center_x = 50.0
    dx_right = dx_mm3 - right_lung_x_radius * 0.3  # Shift right
    dy_right = dy_mm3
    right_lung = (
        (dx_right / right_lung_x_radius) ** 2 + (dy_right / right_lung_y_radius) ** 2 <= 1.0
    ) & lung_mask_z & body_mask

    lungs = left_lung | right_lung
    ct[lungs] = float(background_tissue_hu)  # Low HU for lung
    act[lungs] = 0  # No activity in lungs

    # ===== HEART & MEDIASTINUM =====
    # Heart: ellipsoid in mediastinal region (between lungs)
    heart_x_radius = 35.0
    heart_y_radius = 40.0
    heart_z_radius = 50.0
    heart_z_center = -60.0
    heart_mask = (
        (dx_mm3 / heart_x_radius) ** 2
        + (dy_mm3 / heart_y_radius) ** 2
        + ((dz_mm3 - heart_z_center) / heart_z_radius) ** 2
        <= 1.0
    ) & body_mask

    ct[heart_mask] = float(muscle_hu)
    act[heart_mask] = float(background_activity_bq_per_voxel)  # Heart perfusion

    # ===== VERTEBRAL COLUMN (Spine) =====
    # Vertebral column: cylinder along z-axis, posterior to center
    vertebra_x_radius = 18.0
    vertebra_y_radius = 25.0
    vertebra_center_dy = -80.0  # Posterior (toward back)

    dy_vert = dy_mm3 - vertebra_center_dy
    vertebra_mask = (
        (dx_mm3 / vertebra_x_radius) ** 2 + (dy_vert / vertebra_y_radius) ** 2 <= 1.0
    ) & (dz_mm3 >= lung_z_min) & (dz_mm3 <= 200.0) & body_mask

    ct[vertebra_mask] = float(bone_hu)  # Trabecular bone
    act[vertebra_mask] = 0  # No activity in bone

    # ===== RIBS (simplified as rings) =====
    rib_inner_radius = 80.0
    rib_outer_radius = 100.0
    rib_thickness = 15.0

    dx_rib = np.sqrt(dx_mm3**2 + dy_mm3**2)
    ribs_mask = (
        (dx_rib >= (rib_inner_radius - rib_thickness))
        & (dx_rib <= rib_outer_radius)
        & (dz_mm3 >= lung_z_min)
        & (dz_mm3 <= lung_z_max)
        & body_mask
    )

    ct[ribs_mask] = float(cortical_bone_hu)  # Cortical bone (harder)
    act[ribs_mask] = 0

    # ===== LIVER (right upper abdomen) =====
    liver_x_radius = 70.0
    liver_y_radius = 60.0
    liver_z_radius = 60.0
    liver_center_dy = -30.0  # Slightly posterior
    liver_center_dz = -20.0

    dy_liver = dy_mm3 - liver_center_dy
    liver_mask = (
        (dx_mm3 / liver_x_radius) ** 2
        + (dy_liver / liver_y_radius) ** 2
        + ((dz_mm3 - liver_center_dz) / liver_z_radius) ** 2
        <= 1.0
    ) & (dx_mm3 > -20)  # Mostly on right side
    liver_mask = liver_mask & body_mask

    ct[liver_mask] = float(soft_tissue_hu)
    act[liver_mask] = float(background_activity_bq_per_voxel) * float(
        hot_organ_ratio
    )  # Hot organ (liver uptake)

    # ===== SPLEEN (left upper abdomen) =====
    spleen_x_radius = 40.0
    spleen_y_radius = 35.0
    spleen_z_radius = 45.0
    spleen_center_dy = 10.0
    spleen_center_dz = -30.0

    dy_spleen = dy_mm3 - spleen_center_dy
    spleen_mask = (
        (dx_mm3 / spleen_x_radius) ** 2
        + (dy_spleen / spleen_y_radius) ** 2
        + ((dz_mm3 - spleen_center_dz) / spleen_z_radius) ** 2
        <= 1.0
    ) & (dx_mm3 < 20)  # On left side
    spleen_mask = spleen_mask & body_mask

    ct[spleen_mask] = float(soft_tissue_hu)
    act[spleen_mask] = float(background_activity_bq_per_voxel) * float(hot_organ_ratio)

    # ===== KIDNEYS (mid-abdomen, posterior) =====
    kidney_x_radius = 35.0
    kidney_y_radius = 30.0
    kidney_z_radius = 50.0
    kidney_center_dy = -60.0  # Posterior
    kidney_center_dz = 20.0

    dy_kidney = dy_mm3 - kidney_center_dy
    
    # Left kidney
    left_kidney_mask = (
        ((dx_mm3 + 50) / kidney_x_radius) ** 2
        + (dy_kidney / kidney_y_radius) ** 2
        + ((dz_mm3 - kidney_center_dz) / kidney_z_radius) ** 2
        <= 1.0
    ) & body_mask

    # Right kidney
    right_kidney_mask = (
        ((dx_mm3 - 50) / kidney_x_radius) ** 2
        + (dy_kidney / kidney_y_radius) ** 2
        + ((dz_mm3 - kidney_center_dz) / kidney_z_radius) ** 2
        <= 1.0
    ) & body_mask

    kidneys_mask = left_kidney_mask | right_kidney_mask
    ct[kidneys_mask] = float(soft_tissue_hu)
    act[kidneys_mask] = float(background_activity_bq_per_voxel) * float(hot_organ_ratio)

    # ===== PELVIS (lower body) =====
    pelvis_z_min = 80.0
    pelvis_z_max = 250.0
    pelvis_mask_z = (dz_mm3 >= pelvis_z_min) & (dz_mm3 <= pelvis_z_max)

    # Pelvic bone (ring structure)
    pelvis_inner_radius = 70.0
    pelvis_outer_radius = 105.0
    dx_pelvis = np.sqrt(dx_mm3**2 + dy_mm3**2)
    pelvis_bone_mask = (
        (dx_pelvis >= pelvis_inner_radius)
        & (dx_pelvis <= pelvis_outer_radius)
        & pelvis_mask_z
        & body_mask
    )

    ct[pelvis_bone_mask] = float(bone_hu)
    act[pelvis_bone_mask] = 0

    # Pelvic organs (soft tissue)
    pelvic_organs_mask = (dx_pelvis < pelvis_inner_radius) & pelvis_mask_z & body_mask
    ct[pelvic_organs_mask] = float(soft_tissue_hu)
    act[pelvic_organs_mask] = float(background_activity_bq_per_voxel)

    # ===== FAT LAYER (subcutaneous fat on outer surface) =====
    # Fat forms an annular region between organs and body surface
    outer_radius = 95.0  # Just inside body boundary
    fat_inner_radius = 75.0

    dx_fat = np.sqrt(dx_mm3**2 + dy_mm3**2)
    fat_mask = (
        (dx_fat > fat_inner_radius) & (dx_fat < outer_radius) & body_mask
    )
    fat_mask = fat_mask & ~ribs_mask & ~vertebra_mask  # Don't overwrite bone
    ct[fat_mask] = float(fat_hu)
    act[fat_mask] = 0.5 * float(background_activity_bq_per_voxel)  # Slight uptake

    # Create NIfTI volumes
    ct_vol = _make_nifti_volume_from_array(data=ct, affine=affine, voxel_size_mm=voxel_size_mm)
    act_vol = _make_nifti_volume_from_array(
        data=act, affine=affine, voxel_size_mm=voxel_size_mm
    )

    return ct_vol, act_vol


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a simplified ICRP110-inspired reference phantom (CT HU + activity) as NIfTI."
    )
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--shape", default="150,160,160", help="Shape as Z,Y,X (default: 150,160,160)"
    )
    ap.add_argument(
        "--voxel_mm", default="2.5,2.5,2.5", help="Voxel size in mm (default: 2.5,2.5,2.5)"
    )
    ap.add_argument(
        "--background_activity",
        type=float,
        default=50.0,
        help="Background activity Bq/voxel (default: 50)",
    )
    ap.add_argument(
        "--hot_organ_ratio",
        type=float,
        default=3.0,
        help="Hot organ to background ratio (default: 3.0)",
    )
    args = ap.parse_args()

    Z, Y, X = [int(x) for x in args.shape.split(",")]
    vx, vy, vz = [float(x) for x in args.voxel_mm.split(",")]

    print("Generating ICRP110-inspired phantom...")
    print(f"  Shape: {Z} × {Y} × {X}")
    print(f"  Voxel size: {vx} × {vy} × {vz} mm")
    print(f"  Background activity: {args.background_activity} Bq/voxel")
    print(f"  Hot organ ratio: {args.hot_organ_ratio}×")

    ct_vol, act_vol = make_icrp110_phantom(
        shape_zyx=(Z, Y, X),
        voxel_size_mm=(vx, vy, vz),
        background_activity_bq_per_voxel=args.background_activity,
        hot_organ_ratio=args.hot_organ_ratio,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ct_path = os.path.join(args.out_dir, "icrp110_ct_hu.nii.gz")
    act_path = os.path.join(args.out_dir, "icrp110_activity_bqs.nii.gz")

    save_nifti_like(ct_vol, ct_vol.data, ct_path)
    save_nifti_like(act_vol, act_vol.data, act_path)

    print(f"\nWrote:")
    print(f"  {ct_path}")
    print(f"  {act_path}")


if __name__ == "__main__":
    main()
