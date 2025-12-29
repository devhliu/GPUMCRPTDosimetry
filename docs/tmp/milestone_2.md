Implemented Milestone 2’s first runnable “photon-electron-local” transport path (Woodcock flight + interaction sampling + Compton/Rayleigh/PE handling) and wired it into the existing engine selector.

What Changed

Added a new Triton backend that actually runs photon transport steps: engine_gpu_triton_photon_electron_local.py
Updated the transport selector to choose between Milestone 1 MVP and Milestone 2 photon-electron-local via config: engine.py
Added monte_carlo.triton.engine to the example config: example_simulation.yaml
Added CUDA-skipped smoke tests (RNG consumption in classify + engine run sanity): test_milestone2_photon_electron_local_smoke.py
Added a toy runner for the photon-electron-local engine: run_toy_photon_electron_local.py
How To Use

Set monte_carlo.triton.engine: photon_electron_local in your simulation YAML (see the example config above).
Or run the toy script with PYTHONPATH=src:
cd MCGPURPTDosimetry/GPUMCRPTDosimetry && PYTHONPATH=src python scripts/run_toy_photon_electron_local.py --tables toy_physics.h5
Notes / Current Limits

If /samplers/photon/compton/inv_cdf is missing, the engine falls back to an isotropic 
cos
⁡
𝜃
cosθ table (not physically correct, but unblocks Milestone 2 execution).
Pair production is currently “deposit-all-and-kill” (MVP choice).
I couldn’t execute pytest here because pytest isn’t installed in this environment (the tests are in place and will run once your environment has it).

--------------------------------------------------------------------------------------
# PyTIA — Design Document (v1, Ready-to-Implement)

## 0. Confirmed Defaults (from review)

1. **Output units**: **TIA per voxel = Bq·s (absolute per voxel)**  
   - Implementation: convert activity density (Bq/ml) to absolute activity per voxel (Bq) using voxel volume (ml), then integrate over time (s).
   - `TIA_voxel [Bq·s] = ∫ A_density(t)[Bq/ml] * V_voxel[ml] dt`

2. **Noise floor behavior**: **exclude from fitting** (treat as missing)  
   - Points below floor are not used in regression/optimization.
   - If too few valid points remain → fallback to Hybrid or return zero depending on config.

3. **Rising curves**:
   - Default: **trapezoid + physical tail**
   - Option: **assume peak at last time** (alternative tail mode)

4. **Uncertainty in Hybrid**:
   - Default: **bootstrap uncertainty** for Hybrid (and optionally also for other models if enabled)

5. **Auto model selection** and **N=2 vs N≥3 logic**: accepted as designed.

---

## 1. Package Summary

**PyTIA** computes voxel-wise TIA maps from 3D activity maps acquired at multiple timepoints. It supports multiple kinetic models with an auto-selection decision tree, enforces physical decay constraints, reduces noise, estimates peak times when possible, and exports NIfTI maps for TIA, R², uncertainty, and model ID.

---

## 2. Inputs / Outputs

### Inputs
- Activity maps: list of either:
  - NIfTI file paths (`.nii`, `.nii.gz`), or
  - `nibabel` image objects (`nibabel.spatialimages.SpatialImage`)
- Acquisition times: list/array (seconds internally)
- YAML config path or config dict

Optional:
- Mask image (or auto-generated)
- Radionuclide half-life (required for phys tail/constraints)

### Outputs (NIfTI, same affine/grid/header as input)
- `tia.nii.gz` : TIA per voxel **Bq·s**
- `r2.nii.gz` : R² (NaN where undefined)
- `sigma_tia.nii.gz` : bootstrap std dev of TIA
- `model_id.nii.gz` : uint8 model codes (classification + final model used)
- Optional:
  - `tpeak.nii.gz` : estimated peak time (seconds; NaN if not applicable)
  - `denoised_*.nii.gz`, `mask.nii.gz`

---

## 3. Configuration (YAML) — Key Fields

```yaml
io:
  output_dir: "./out"
  save_intermediate: false
  dtype: "float32"

time:
  unit: "seconds"
  sort_timepoints: true

physics:
  half_life_seconds: 23040
  enforce_lambda_ge_phys: true

mask:
  mode: "otsu"              # provided | otsu | none
  provided_path: null
  min_fraction_of_max: 0.02

denoise:
  enabled: true
  method: "masked_gaussian"
  sigma_vox: 1.2

noise_floor:
  enabled: true
  mode: "relative"          # absolute | relative
  absolute_bq_per_ml: 0.0
  relative_fraction_of_voxel_max: 0.01
  behavior: "exclude"       # exclude (missing)

model_selection:
  mode: "auto"              # auto | gamma | exp | hybrid
  min_points_for_gamma: 3

integration:
  start_time_seconds: 0
  rising_tail_mode: "phys"  # phys (default) | peak_at_last
  tail_mode: "phys"         # phys | fit (for non-rising if enabled)

uncertainty:
  enabled: true
  method: "bootstrap"       # bootstrap as default for v1
  bootstrap:
    n: 50
    seed: 0
```

---

## 4. Core Pipeline

1. Load images (paths or nibabel objects)
2. Validate consistent shape + affine; stack to 4D `(X, Y, Z, T)`
3. Convert times into seconds; sort by time if enabled
4. Compute voxel volume from header affine: `V_voxel_ml`
5. Mask generation (Otsu on sum-image, optionally combined with min fraction)
6. Denoising (masked Gaussian, no bleed outside mask)
7. Noise floor exclusion: for each voxel TAC, mark points below floor as invalid
8. For each voxel: classify TAC and select model (auto or user-forced)
9. Fit model (if parametric), integrate to TIA (Bq·s), compute R², estimate uncertainty
10. Assemble full volumes, export NIfTI outputs

Parallelization: multiprocessing over masked voxel indices (chunked).

---

## 5. Models & Auto Selection

### TAC Classification
Given valid points `(t_i, A_i)` after noise-floor exclusion:
- If <2 valid points → return `TIA=0`, metrics NaN (or config option)
- `idx_max = argmax(A_i)`
- **Falling**: max at first valid timepoint
- **Rising**: max at last valid timepoint
- **Hump**: interior max
- Ambiguous/noisy → Hybrid

### Model A — Gamma Variate (Hump, N≥3)
`A(t) = K * t^α * exp(-β t)`
- constraints: `K>0`, `α>0`, `β>=λ_phys` (if enabled)
- peak: `t_peak = α/β`
- TIA (absolute per voxel):
  - integrate density to ∞ analytically: `∫0∞ K t^α exp(-β t) dt`
  - then multiply by voxel volume ml to get Bq·s (or fit directly in Bq)

### Model B — Mono-exponential Tail (Falling or fallback)
- Determine peak time as measured peak (`t_peak_meas`)
- Tail: `A(t)=A_peak * exp(-λ_eff (t - t_peak))`
- λ bounds: `λ_eff >= λ_phys` if enabled
- Uptake area: triangle approx `0.5 * A_peak * t_peak`
- Tail area: `A_peak / λ_eff`
- Sum and multiply by voxel volume → Bq·s

### Model C — Hybrid (Trapezoid + Tail) — Default fallback, and for Rising
Observed:
- trapezoid integration on valid points, include `(0,0)` if `start_time_seconds=0`
Tail:
- Default tail is **physical**: `A_last / λ_phys`
- For Rising curves:
  - default `rising_tail_mode=phys`
  - optional `peak_at_last`: treat last point as peak for tail (same formula but semantics differ; still phys decay rate)

---

## 6. Uncertainty (v1 default = Bootstrap)

Bootstrap approach (per voxel):
- Fit the chosen model (or evaluate piecewise curve for Hybrid)
- Compute residuals at valid timepoints
- Resample residuals and regenerate pseudo-observations
- Recompute TIA `n` times
- `sigma_tia = std(TIA_boot)`
- Outputs NaN if bootstrap cannot run due to insufficient valid points

Notes:
- Bootstrap is used for Hybrid as requested.
- We will ensure reproducibility via seed control.

---

## 7. R² / Goodness of Fit

- For parametric models: R² computed on valid points used in fitting.
- For Hybrid:
  - We can compute R² by evaluating the hybrid predicted values at sampled times (trapezoid piecewise linear + exponential tail if needed only for evaluation).
  - If too few points, R² = NaN.

---

## 8. Model Codes (example)

- 0: background/outside mask
- 1: invalid/insufficient points
- 10: Hybrid (rising)
- 11: Hybrid (general)
- 20: Monoexp (falling)
- 30: Gamma variate (hump)
- 40: Forced user model (optional flags in high bits)

(Exact mapping will be defined in `pytia/models/base.py`.)

---

## 9. Proposed Package Structure (to implement next)

- `pytia/`
  - `__init__.py`
  - `engine.py` (run_tia orchestration)
  - `config.py` (YAML loader + schema)
  - `io.py` (nibabel load/save, stacking, voxel volume)
  - `masking.py` (otsu + provided mask)
  - `denoise.py` (masked Gaussian)
  - `noise.py` (noise floor exclusion utilities)
  - `classify.py`
  - `models/`
    - `base.py`
    - `gamma_variate.py`
    - `monoexp.py`
    - `hybrid.py`
  - `uncertainty.py` (bootstrap)
  - `metrics.py` (R²)
  - `cli.py`
- `tests/` (pytest)
- `docs/` (mkdocs or sphinx, v1 basic usage + config reference)

---

## 10. Implementation Acceptance Criteria (v1)

- Loads NIfTI / nibabel images reliably; preserves affine/header.
- Produces correct units **Bq·s per voxel** using voxel volume conversion.
- Auto model selection works on synthetic falling/rising/hump cases.
- Physical tail and λ constraints applied correctly.
- Bootstrap uncertainty map produced and reproducible with seed.
- CLI + Python API functional.
- Unit tests cover:
  - units conversion
  - each model on synthetic TACs
  - noise floor exclusion behavior
  - rising curve handling
  - bootstrap reproducibility

---

## 11. Remaining Confirmations (small)
1. Should we treat **negative activity values** (can occur after reconstruction) as:
   - clamp to 0 before noise-floor exclusion (recommended), or
   - allow but exclude by floor, or
   - error?
2. Do you want output filenames fixed (`tia.nii.gz`, etc.) or configurable prefix?

(If you confirm these, we can start building the package.)

------------------------------------------------------------------------------

Updated the Milestone-2 photon-electron-local engine to always use an isotropic cosθ inverse-CDF table for Compton (ignores any H5-provided Compton table): engine_gpu_triton_photon_electron_local.py
Fixed a real import-time SyntaxError in the Triton Compton kernel call (must pass K as positional tl.constexpr, not K=...): engine_gpu_triton_photon_electron_local.py
Re-ran pytest successfully (with PYTHONPATH=src): 2 passed, 7 skipped (skips are CUDA/Triton-gated tests)
