#!/usr/bin/env python
"""Run dosimetry calculation from configuration or direct NIfTI paths.

This CLI provides two modes of operation:

1. Config-based mode (multi-method YAML):
   - Uses --config and --method arguments
   - Loads CT/activity paths from config
   - Auto-generates output paths based on nuclide name and method
   - Ideal for batch processing with multiple methods

2. Direct path mode (explicit paths):
   - Uses --ct, --activity, --sim_yaml arguments
   - User specifies all paths explicitly
   - More flexible for single-run scenarios
   - Supports custom output paths via --out_dose and --out_unc

Both modes delegate execution to `gpumcrpt.python_api.pipeline.run_dosimetry`
to avoid duplicating:
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
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dosimetry_from_config(
    config_path: str,
    method_name: str,
    out_dose: str | None = None,
    out_unc: str | None = None,
) -> None:
    """Run dosimetry using multi-method YAML configuration."""
    main_config = load_config(config_path)

    config_dir = Path(config_path).parent
    output_dir = config_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)

    method_cfg_rel = main_config["methods"][method_name]
    method_cfg_path = Path(method_cfg_rel)
    if not method_cfg_path.is_absolute():
        method_cfg_path = config_dir / method_cfg_path

    method_config = load_config(method_cfg_path)

    ct_path = Path(main_config["input"]["ct"])
    if not ct_path.is_absolute():
        ct_path = config_dir / ct_path

    tia_path = Path(main_config["input"]["tia"])
    if not tia_path.is_absolute():
        tia_path = config_dir / tia_path

    nuclide_name = main_config["input"]["nuclide"]

    sim_config = dict(method_config)
    sim_config.setdefault("io", {})
    sim_config["io"]["resample_ct_to_activity"] = True
    sim_config["nuclide"] = {"name": str(nuclide_name)}

    # Use provided output paths or auto-generate based on config
    if out_dose is None:
        output_dose_path = output_dir / f"{str(nuclide_name).lower()}_dose_{method_name}.nii.gz"
    else:
        output_dose_path = Path(out_dose)

    if out_unc is None:
        output_unc_path = output_dir / f"{str(nuclide_name).lower()}_uncertainty_{method_name}.nii.gz"
    else:
        output_unc_path = Path(out_unc)

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


def run_dosimetry_direct(
    ct_path: str,
    activity_path: str,
    sim_yaml_path: str,
    out_dose: str,
    out_unc: str,
    device: str | None = None,
) -> None:
    """Run dosimetry using direct NIfTI paths."""
    sim_config = load_config(sim_yaml_path)

    run_dosimetry_pipeline(
        activity_nifti_path=str(activity_path),
        ct_nifti_path=str(ct_path),
        sim_config=sim_config,
        output_dose_path=str(out_dose),
        output_unc_path=str(out_unc),
        device=device,
    )

    print(f"Dosimetry calculation completed")
    print(f"Dose output: {out_dose}")
    print(f"Uncertainty output: {out_unc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dosimetry calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Config-based mode (multi-method YAML):
  python run_dosimetry.py --config config.yaml --method photon_em_condensed

  # Config-based mode with custom output paths:
  python run_dosimetry.py --config config.yaml --method photon_em_condensed \\
      --out_dose /path/to/dose.nii.gz --out_unc /path/to/unc.nii.gz

  # Direct path mode (explicit paths):
  python run_dosimetry.py --ct ct.nii.gz --activity activity.nii.gz \\
      --sim_yaml sim.yaml --out_dose dose.nii.gz --out_unc unc.nii.gz

  # Direct path mode with device override:
  python run_dosimetry.py --ct ct.nii.gz --activity activity.nii.gz \\
      --sim_yaml sim.yaml --out_dose dose.nii.gz --out_unc unc.nii.gz --device cuda
        """,
    )

    # Config-based mode arguments
    parser.add_argument(
        "--config",
        help="Path to multi-method YAML configuration file (use with --method)",
    )
    parser.add_argument(
        "--method",
        choices=["local_deposit", "photon_em_condensed", "photon_em_energybucketed", "photon_only"],
        help="Method to use for dosimetry calculation (use with --config)",
    )

    # Direct path mode arguments
    parser.add_argument(
        "--ct",
        help="CT HU NIfTI path (use with --activity, --sim_yaml, --out_dose, --out_unc)",
    )
    parser.add_argument(
        "--activity",
        help="Activity (Bq*s per voxel) NIfTI path (use with --ct, --sim_yaml, --out_dose, --out_unc)",
    )
    parser.add_argument(
        "--sim_yaml",
        help="Simulation YAML path (use with --ct, --activity, --out_dose, --out_unc)",
    )

    # Output paths (optional for config mode, required for direct mode)
    parser.add_argument(
        "--out_dose",
        help="Output dose NIfTI path (optional for config mode, required for direct mode)",
    )
    parser.add_argument(
        "--out_unc",
        help="Output uncertainty NIfTI path (optional for config mode, required for direct mode)",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        help="Override device (e.g. cpu, cuda)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    config_mode = args.config is not None and args.method is not None
    direct_mode = all(
        [
            args.ct is not None,
            args.activity is not None,
            args.sim_yaml is not None,
            args.out_dose is not None,
            args.out_unc is not None,
        ]
    )

    if not config_mode and not direct_mode:
        parser.error(
            "Either provide --config and --method for config-based mode, "
            "or provide --ct, --activity, --sim_yaml, --out_dose, and --out_unc for direct path mode."
        )

    if config_mode and direct_mode:
        parser.error(
            "Cannot mix config-based mode and direct path mode. "
            "Use either --config/--method or --ct/--activity/--sim_yaml/--out_dose/--out_unc."
        )

    try:
        if config_mode:
            run_dosimetry_from_config(
                config_path=args.config,
                method_name=args.method,
                out_dose=args.out_dose,
                out_unc=args.out_unc,
            )
        else:
            run_dosimetry_direct(
                ct_path=args.ct,
                activity_path=args.activity,
                sim_yaml_path=args.sim_yaml,
                out_dose=args.out_dose,
                out_unc=args.out_unc,
                device=args.device,
            )
    except Exception as e:
        print(f"Error running dosimetry: {e}")
        raise


if __name__ == "__main__":
    main()
