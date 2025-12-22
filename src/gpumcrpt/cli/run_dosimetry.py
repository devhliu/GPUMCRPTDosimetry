#!/usr/bin/env python
"""Run dosimetry from a multi-method YAML config.

This CLI is a thin wrapper that delegates execution to
`gpumcrpt.python_api.pipeline.run_dosimetry` to avoid duplicating:

- materials construction
- physics table selection/loading
- decay sampling (activity is interpreted as Bq*s per voxel)
- transport engine wiring
- output writing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Add the package root (the `src/` directory) to Python path when running as a script.
# File is `src/gpumcrpt/cli/run_dosimetry.py` so `parents[2]` is `src/`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpumcrpt.python_api.pipeline import run_dosimetry as run_dosimetry_pipeline


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dosimetry(config_path: str, method_name: str) -> None:
    main_config = load_config(config_path)

    config_dir = Path(config_path).parent
    output_dir = config_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)

    method_cfg_rel = main_config["methods"][method_name]
    method_cfg_path = Path(method_cfg_rel)
    if not method_cfg_path.is_absolute():
        method_cfg_path = config_dir / method_cfg_path

    method_config = load_config(method_cfg_path)

    ct_path = config_dir / main_config["input"]["ct"]
    tia_path = config_dir / main_config["input"]["tia"]
    nuclide_name = main_config["input"]["nuclide"]

    sim_config = dict(method_config)
    sim_config.setdefault("io", {})
    sim_config["io"]["resample_ct_to_activity"] = True
    sim_config["nuclide"] = {"name": str(nuclide_name)}

    output_dose_path = output_dir / f"{str(nuclide_name).lower()}_dose_{method_name}.nii.gz"
    output_unc_path = output_dir / f"{str(nuclide_name).lower()}_uncertainty_{method_name}.nii.gz"

    run_dosimetry_pipeline(
        activity_nifti_path=str(tia_path),
        ct_nifti_path=str(ct_path),
        sim_config=sim_config,
        output_dose_path=str(output_dose_path),
        output_unc_path=str(output_unc_path),
        device=sim_config.get("device", None),
    )

    print(f"Dosimetry calculation completed for {method_name}")
    print(f"Results saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dosimetry calculation")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument(
        "--method",
        required=True,
        choices=["local_deposit", "photon_em_condensed", "photon_only"],
        help="Method to use for dosimetry calculation",
    )
    args = parser.parse_args()

    try:
        run_dosimetry(args.config, args.method)
    except Exception as e:
        print(f"Error running dosimetry: {e}")
        raise


if __name__ == "__main__":
    main()