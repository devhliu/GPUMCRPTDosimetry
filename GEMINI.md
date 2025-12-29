# Gemini Code-Aware System Instruction

## Project Overview

This project, `GPUMCRPTDosimetry`, is a research prototype for a high-performance, GPU-accelerated Monte Carlo particle transport simulation package. Its primary application is internal dosimetry for Radiopharmaceutical Therapy (RPT), calculating absorbed radiation doses at the voxel level.

The system is built in Python and leverages `PyTorch` for tensor operations and `Triton` (v3.5.1) for writing high-performance GPU kernels. It is designed to work directly with medical imaging data in NIfTI format (for CT scans and activity maps).

The architecture is modular, with distinct sub-packages for handling different aspects of the simulation:
-   `materials`: Manages material definitions and conversion from CT Hounsfield Units (HU) to tissue properties.
-   `physics_tables`: Manages pre-computed physics data like cross-sections and stopping powers.
-   `decaydb`: Handles the ICRP-107 radionuclide decay database.
-   `source`: Samples initial particles from the given activity distribution based on the selected radionuclide.
-   `transport`: The core of the simulation. It contains the particle transport engines, which use `Triton` kernels for GPU acceleration. It simulates photon, electron, and positron interactions.
-   `dose`: Scores the energy deposited during transport and converts it to an absorbed dose.
-   `python_api` & `cli`: Provide high-level interfaces for running simulations, both programmatically and from the command line.

## Building and Running

### Installation

The project uses a standard Python packaging setup (`pyproject.toml` and `setup.py`).

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install for development:** For running tests or modifying the source code, perform an editable install.
    ```bash
    pip install -e ".[dev]"
    ```

### Running Simulations

Simulations are configured using YAML files (see `src/gpumcrpt/configs/` for examples).

There are two primary ways to run a simulation:

1.  **Command-Line Interface (CLI):** The project provides a `gpumcrpt-run` command.

    ```bash
    gpumcrpt-run \
        --ct /path/to/ct.nii.gz \
        --activity /path/to/activity.nii.gz \
        --sim_yaml /path/to/simulation_config.yaml \
        --out_dose /path/to/output_dose.nii.gz \
        --out_unc /path/to/output_uncertainty.nii.gz
    ```

2.  **Python API:** Use the `run_dosimetry` function from the `gpumcrpt.python_api.pipeline` module.

    ```python
    from gpumcrpt.python_api.pipeline import run_dosimetry
    import yaml

    # Load configuration
    with open("src/gpumcrpt/configs/method_photon_electron_condensed.yaml", "r") as f:
        sim_config = yaml.safe_load(f)

    # Run dosimetry calculation
    run_dosimetry(
        activity_nifti_path="activity.nii.gz",
        ct_nifti_path="ct.nii.gz",
        sim_config=sim_config,
        output_dose_path="dose.nii.gz",
        output_unc_path="uncertainty.nii.gz",
        device="cuda"
    )
    ```

### Running Tests

The project uses `pytest` for testing. To run the test suite:

```bash
PYTHONPATH=src pytest
```
*(Note: The `tests/` directory is assumed to exist based on project conventions).*

## Development Conventions

The core of this project's performance comes from custom GPU kernels written in Triton. Adherence to the project's Triton coding style is critical.

### Triton Kernel Development

Based on `.github/triton-coding-instructions.md`, all Triton kernels should follow these guidelines:

-   **Target Version:** Strictly use **Triton 3.5.1**.
-   **Structure:**
    -   Use the `@triton.jit` decorator.
    -   Use `@triton.autotune` to find the best block sizes and configurations.
    -   Define kernels to operate on pointers to tensors (`*_ptr`).
    -   Pass problem dimensions and block sizes as `tl.constexpr`.
    -   Use `tl.program_id`, `tl.arange`, and `tl.load`/`tl.store` with masks for safe, parallel memory access.
-   **Memory:**
    -   Use modern block pointers (`tl.make_block_ptr`) for multi-dimensional access.
    -   Use cache modifiers (`.cg`, `.ca`) where appropriate.
    -   Ensure memory access is coalesced.
-   **RNG:** Use the provided Philox RNG implementation for random number generation within kernels.
-   **Data Layout:** Particle data is stored in a Structure-of-Arrays (SoA) layout for coalesced memory access (e.g., separate pointers for `x_ptr`, `y_ptr`, `energy_ptr`).
-   **Atomics:** Use atomic operations (`tl.atomic_add`) for thread-safe updates to shared memory, such as the energy deposition grid.
-   **Testing:** Use `triton.testing` for unit tests and benchmarking of kernels.
-   **Style:** Keep kernels focused on a single task. Use meaningful names and document constraints.
