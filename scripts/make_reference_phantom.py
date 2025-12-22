from __future__ import annotations

import argparse
import os

import nibabel as nib
from gpumcrpt.phantoms.phantoms import make_water_slab_with_bone_cylinder


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a simple reference phantom (CT HU + activity) as NIfTI.")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--shape", default="64,64,64", help="Shape as Z,Y,X (default: 64,64,64)")
    ap.add_argument("--voxel_mm", default="2,2,2", help="Voxel size in mm as X,Y,Z (default: 2,2,2)")
    ap.add_argument("--bone_radius_mm", type=float, default=20.0)
    ap.add_argument("--activity_bq_per_voxel", type=float, default=1.0)
    ap.add_argument("--activity_sphere_radius_mm", type=float, default=25.0)
    args = ap.parse_args()

    Z, Y, X = [int(x) for x in args.shape.split(",")]
    vx, vy, vz = [float(x) for x in args.voxel_mm.split(",")]

    phantom = make_water_slab_with_bone_cylinder(
        shape_zyx=(Z, Y, X),
        voxel_size_mm=(vx, vy, vz),
        bone_radius_mm=float(args.bone_radius_mm),
        activity_bq_per_voxel=float(args.activity_bq_per_voxel),
        activity_sphere_radius_mm=float(args.activity_sphere_radius_mm),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ct_path = os.path.join(args.out_dir, "phantom_ct_hu.nii.gz")
    act_path = os.path.join(args.out_dir, "phantom_activity_bqs.nii.gz")

    nib.save(phantom.ct_hu, ct_path)
    nib.save(phantom.activity_bqs, act_path)

    print("Wrote:")
    print(f"  {ct_path}")
    print(f"  {act_path}")


if __name__ == "__main__":
    main()
