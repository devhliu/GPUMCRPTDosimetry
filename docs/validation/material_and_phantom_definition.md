# Material and Phantom Definition

Accurate simulation of radiation transport requires realistic models of patient tissues and well-defined phantoms for validation. This document outlines how materials are defined from CT images and how reference phantoms are used in the GPUMCRPTDosimetry project.

## Material Definition from CT Images

The simulation uses patient-specific tissue models derived from CT images. The process involves converting the Hounsfield Units (HU) from the CT scan into material definitions and densities.

### HU to Material and Density Mapping

A piecewise-linear mapping is used to convert HU values to a material ID and a corresponding mass density (in g/cm³). The project includes a default 5-compartment model based on ICRU Report 44, which is suitable for general-purpose dosimetry:

| Material      | HU Range      | Density (g/cm³) | Key Elements |
|---------------|---------------|-----------------|--------------|
| Air           | < -850        | 0.0012          | N₂, O₂, Ar   |
| Lung          | -850 to -910  | 0.355           | C, H, N, O   |
| Fat           | -100 to -50   | 0.95            | C, H, O      |
| Muscle        | +10 to +40    | 1.06            | C, H, N, O   |
| Soft Tissue   | 0 to +50      | 1.00            | C, H, N, O   |
| Bone          | +150 to +3000 | 1.507           | C, H, N, O, P, Ca |

### Material Library and Elemental Composition

The project uses a `MaterialsLibrary` that stores the elemental composition (by mass fraction) for each material. This is crucial for calculating the macroscopic cross-sections for physical interactions. The validation process ensures that:

*   The elemental compositions for each material correctly sum to 1.0.
*   The effective atomic number (Z) is calculated correctly for each material.
*   The cross-sections are scaled consistently with the material density.

## Reference Phantoms for Validation

Reference phantoms are essential for validating the simulation results against known values or other Monte Carlo codes. The project includes scripts to generate standard phantoms.

### NEMA IEC Body Phantom

A simplified version of the NEMA IEC body phantom can be generated using `scripts/make_nema_phantom.py`. This script produces:
*   A CT image with HU values.
*   An activity map.
*   Optional sphere labels.

This phantom is useful for testing the full simulation pipeline on a geometry with known features.

### ICRP-110 Like Phantom

The simulation can be run on any user-provided phantom that follows the NIfTI format. The `scripts/run_dosimetry_nifti.py` script can be used to run a simulation on a phantom resembling the ICRP-110 adult reference phantom, provided the user supplies:
*   A CT HU NIfTI file.
*   An activity NIfTI file with the same grid and affine transformation.

### Homogeneous Phantoms

Simple homogeneous phantoms (e.g., a water slab with a bone insert) are used for specific validation tasks, such as:
*   **Energy Conservation:** A fully absorbing phantom is used to check that the total deposited energy equals the total emitted energy.
*   **Physics Checks:** A simple geometry allows for easier verification of individual physics processes, like the transport and interaction of monoenergetic photons.
*   The `scripts/make_reference_phantom.py` script can be used to generate a water slab with a bone cylinder insert.
