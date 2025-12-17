from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from gpumcrpt.io.nifti import load_nifti


@dataclass
class CompareResult:
    mean_abs_percent: float
    p95_abs_percent: float
    max_abs_percent: float


def _safe_percent_diff(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    denom = np.maximum(np.abs(b), eps)
    return 100.0 * (a - b) / denom


def compare_dose(
    *,
    dose_path: str,
    reference_path: str,
    mask_path: str | None,
    eps: float,
) -> CompareResult:
    dose_img = load_nifti(dose_path)
    ref_img = load_nifti(reference_path)

    dose = np.asarray(dose_img.data, dtype=np.float64)
    ref = np.asarray(ref_img.data, dtype=np.float64)
    if dose.shape != ref.shape:
        raise ValueError(f"Shape mismatch: dose={dose.shape} ref={ref.shape}")

    if mask_path is not None:
        mask_img = load_nifti(mask_path)
        mask = np.asarray(mask_img.data) > 0
        if mask.shape != dose.shape:
            raise ValueError(f"Mask shape mismatch: mask={mask.shape} dose={dose.shape}")
    else:
        mask = np.ones_like(dose, dtype=bool)

    diff_pct = _safe_percent_diff(dose, ref, eps=eps)
    v = np.abs(diff_pct[mask])
    if v.size == 0:
        raise ValueError("Mask selects no voxels")

    return CompareResult(
        mean_abs_percent=float(np.mean(v)),
        p95_abs_percent=float(np.percentile(v, 95.0)),
        max_abs_percent=float(np.max(v)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare dose NIfTI against a Geant4/GATE reference dose.")
    ap.add_argument("--dose", required=True, help="Path to your dose NIfTI")
    ap.add_argument("--ref", required=True, help="Path to reference dose NIfTI")
    ap.add_argument("--mask", default=None, help="Optional binary mask NIfTI (ROI)")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon for percent difference denominator")
    args = ap.parse_args()

    r = compare_dose(dose_path=args.dose, reference_path=args.ref, mask_path=args.mask, eps=float(args.eps))
    print("Dose comparison (% diff vs reference):")
    print(f"  mean(|%|): {r.mean_abs_percent:.3f}")
    print(f"  p95(|%|):  {r.p95_abs_percent:.3f}")
    print(f"  max(|%|):  {r.max_abs_percent:.3f}")


if __name__ == "__main__":
    main()
