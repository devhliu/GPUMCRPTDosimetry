
# Milestone 4: Clinical realism

Goal
- Move from MVP "material_id + density" toward clinically-meaningful materials with elemental compositions, improved atomic relaxation mapping for mixtures, and uncertainty outputs suitable for validation against reference MC (Geant4/GATE).

In scope
- HU → density → material-class mapping that produces per-material elemental compositions (mass fractions) and reference densities.
- Relaxation mapping for mixtures (materials composed of multiple elements).
- Dose uncertainty outputs (batch-based standard error; relative/absolute modes).
- A reference-phantom comparison workflow (not running Geant4 here, but providing loaders + metrics so users can compare against exported Geant4/GATE dose).

Out of scope (for this milestone)
- Full CT calibration model fitting from DICOM (we accept a piecewise-linear HU→ρ curve).
- Full ICRP/ICRU library import tooling (we hardcode a small, documented tissue set).
- Full gamma-index suite (we provide basic profile comparisons and summary metrics).

Deliverables
1) Materials model
	- A small set of tissue materials with elemental compositions (mass fractions) and reference densities.
	- A runtime mapping: HU → (material_id, ρ) per voxel, plus a lookup table: material_id → composition.

2) Relaxation mapping improvements
	- A material→relaxation mapping that supports mixtures via a mixture rule.
	- MVP approach: sample element according to composition weight, then apply element-Z-specific relaxation tables.

3) Uncertainty outputs
	- Use batch-to-batch variance as estimator for uncertainty.
	- Expose both absolute (Gy) and relative standard error.

4) Validation harness
	- Reference phantom definition in Python (water slab + bone insert is sufficient).
	- Script to compare your dose/uncertainty against a provided Geant4/GATE reference dose volume:
		 - relative difference maps
		 - 1D profiles (central axis)
		 - summary stats (mean abs %, 95th percentile)

Acceptance criteria
- HU mapping produces stable material IDs and reasonable densities (monotone ρ vs HU).
- Material compositions sum to 1.0 within tolerance.
- Relaxation mapping for a mixture behaves deterministically (given RNG seed) and conserves energy.
- Dose uncertainty output is non-negative, finite, and scales ~1/sqrt(N_batches).
- Comparison script can load a reference dose volume and produce profile + summary metrics.

Scripts
- scripts/make_reference_phantom.py: generates CT HU + activity NIfTI (water slab + bone cylinder).
- scripts/make_nema_phantom.py: generates a simplified NEMA IEC body phantom (CT HU + activity + optional sphere labels) as NIfTI.
- scripts/run_dosimetry_nifti.py: runs the full pipeline on arbitrary CT/activity NIfTI inputs (for NEMA, ICRP110-like phantoms, etc.).
- scripts/compare_dose_to_reference.py: compares your dose NIfTI to a Geant4/GATE reference dose NIfTI.

Validation phantoms
- NEMA IEC body phantom: generated in-repo via scripts/make_nema_phantom.py.
- ICRP110-like phantom: provide your own CT HU NIfTI + activity NIfTI (same grid/affine, or enable io.resample_ct_to_activity) and run via scripts/run_dosimetry_nifti.py.

NIfTI conventions (expected)
- CT: HU values (float32) in the CT volume.
- Activity: "Bq/s per voxel" (float32). The pipeline samples histories proportional to this voxel value.
Notes
- This milestone is primarily about data/modeling and I/O/validation glue. It should not regress performance-critical kernels.



- enhance HU-materials with elemental composition
- improve relaxation tables for multi-element materials
- add uncertainty tracking and outputs
- create validation phantom data structure
- add Geant4/GATE comparison utilities
- cteate milestone 4 test suite


--------------------------------------
What I changed (Milestone 4 “Clinical realism” groundwork)

Added a concrete Milestone 4 scope + acceptance criteria in milestone_4.md.
Upgraded HU→materials to support elemental compositions via a new MaterialsLibrary and optional per-material effective Z (for MVP relaxation mapping) in hu_materials.py.
Wired the materials library into the runnable pipeline and fixed uncertainty output saving in pipeline.py.
Added a mixture helper (per-material element CDF) for future relaxation sampling in relaxation_tables.py.
Fixed a blocking bug: load_physics_tables_h5() now returns the constructed tables in tables.py.
Validation + comparison utilities

New CPU-only tests for material compositions + CDF sanity: test_milestone4_materials_composition.py (ran: 3 passed).
New comparison script for Geant4/GATE reference dose volumes (NIfTI): compare_dose_to_reference.py.
Notes / next step to fully complete Milestone 4

The default tissue compositions are reasonable MVP defaults (and configurable via materials.material_library), but they’re not yet sourced from an imported ICRU/ICRP dataset.
The remaining big missing piece is the reference phantom generator + an automated “compare to reference” harness (your todo step 5). If you tell me what phantom you want first (water slab? ICRP110-like? Catphan inserts?) and the expected reference format (NIfTI vs other), I’ll implement that next.
---------------------------------------

