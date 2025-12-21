from __future__ import annotations

import argparse
import os

import nibabel as nib
from gpumcrpt.materials.phantoms import make_nema_iec_body_phantom


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a simplified NEMA IEC body phantom (CT HU + activity) as NIfTI.")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--shape", default="96,128,128", help="Shape as Z,Y,X (default: 96,128,128)")
    ap.add_argument("--voxel_mm", default="2,2,2", help="Voxel size in mm as X,Y,Z (default: 2,2,2)")
    ap.add_argument("--body_radius_mm", type=float, default=100.0)
    ap.add_argument("--body_length_mm", type=float, default=0.0, help="If >0, truncates cylinder along Z")
    ap.add_argument("--ring_radius_mm", type=float, default=57.5)
    ap.add_argument("--background_bq_per_voxel", type=float, default=1.0)
    ap.add_argument("--hot_ratio", type=float, default=4.0)
    ap.add_argument("--no_label", action="store_true", help="Do not write sphere label volume")
    args = ap.parse_args()

    Z, Y, X = [int(x) for x in args.shape.split(",")]
    vx, vy, vz = [float(x) for x in args.voxel_mm.split(",")]

    body_length_mm = None if float(args.body_length_mm) <= 0.0 else float(args.body_length_mm)

    phantom = make_nema_iec_body_phantom(
        shape_zyx=(Z, Y, X),
        voxel_size_mm=(vx, vy, vz),
        body_cylinder_radius_mm=float(args.body_radius_mm),
        body_length_mm=body_length_mm,
        sphere_ring_radius_mm=float(args.ring_radius_mm),
        background_bq_per_voxel=float(args.background_bq_per_voxel),
        hot_sphere_ratio=float(args.hot_ratio),
        include_sphere_label=not bool(args.no_label),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ct_path = os.path.join(args.out_dir, "nema_ct_hu.nii.gz")
    act_path = os.path.join(args.out_dir, "nema_activity_bqs.nii.gz")
    lbl_path = os.path.join(args.out_dir, "nema_sphere_label.nii.gz")

    nib.save(phantom.ct_hu, ct_path)
    nib.save(phantom.activity_bqs, act_path)
    if phantom.sphere_label is not None:
        nib.save(phantom.sphere_label, lbl_path)

    print("Wrote:")
    print(f"  {ct_path}")
    print(f"  {act_path}")
    if phantom.sphere_label is not None:
        print(f"  {lbl_path}")


if __name__ == "__main__":
    main()
