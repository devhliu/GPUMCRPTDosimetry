from __future__ import annotations

import argparse
from pathlib import Path

from gpumcrpt.python_api.pipeline import run_dosimetry


def main() -> None:
    ap = argparse.ArgumentParser(description="Run gpumcrpt dosimetry on CT/activity NIfTI inputs.")
    ap.add_argument("--ct", required=True, help="CT HU NIfTI path")
    ap.add_argument("--activity", required=True, help="Activity (Bq*s per voxel) NIfTI path")
    ap.add_argument("--sim_yaml", required=True, help="Simulation YAML (see configs/example_simulation.yaml)")
    ap.add_argument("--out_dose", required=True, help="Output dose NIfTI path")
    ap.add_argument("--out_unc", required=True, help="Output uncertainty NIfTI path")
    ap.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda)")
    args = ap.parse_args()

    yaml = __import__("yaml")

    sim_cfg = yaml.safe_load(Path(args.sim_yaml).read_text(encoding="utf-8"))

    run_dosimetry(
        activity_nifti_path=str(args.activity),
        ct_nifti_path=str(args.ct),
        sim_config=sim_cfg,
        output_dose_path=str(args.out_dose),
        output_unc_path=str(args.out_unc),
        device=args.device,
    )


if __name__ == "__main__":
    main()
