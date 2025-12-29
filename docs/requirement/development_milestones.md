# Development Milestones

The development of the GPUMCRPTDosimetry package is divided into four major milestones, each building upon the last to progressively add features and increase the physical realism of the simulation.

## Milestone 1: Runnable MVP

The first milestone focused on creating a "Minimum Viable Product" (MVP) that could run end-to-end.

*   **Goal:** Establish a runnable pipeline that can take input, run on the GPU, and produce an output energy deposition tensor.
*   **Key Features:**
    *   A basic transport engine that deposits all particle energy locally. This was a placeholder to make the system runnable.
    *   Resolution of all blocking import and dependency issues.
    *   Addition of missing subpackages for materials, NIfTI I/O, and dose scoring.
    *   An end-to-end runnable script (`run_toy_mvp.py`) and an example configuration file.

## Milestone 2: Photon-Electron Local Transport

This milestone replaced the MVP local-deposition engine with a real photon transport model.

*   **Goal:** Implement a "photon-electron-local" transport path with realistic physics.
*   **Key Features:**
    *   A new Triton backend for Woodcock "wavefront" photon transport.
    *   Interaction sampling for Compton scattering, Rayleigh scattering, and the photoelectric effect.
    *   The engine was made selectable via the configuration file.
    *   Smoke tests to verify the new engine's functionality.
*   **Limitations:**
    *   Pair production was treated as a local energy deposit.
    *   Compton scattering used a simplified isotropic model for initial bring-up.

## Milestone 3: Condensed History for Electrons and Positrons

Milestone 3 extended the transport model to include charged particles.

*   **Goal:** Implement condensed-history transport for electrons and positrons, including positron annihilation.
*   **Key Features:**
    *   A new "photon_electron_condensed" engine selectable via configuration.
    *   Condensed-history stepping for electrons and positrons, depositing energy continuously.
    *   Positron annihilation at rest, producing two 511 keV photons that are then transported.
    *   Bounded secondary particle generation (bremsstrahlung photons and delta electrons) to control the simulation complexity.
*   **Limitations:**
    *   Full photoelectric atomic relaxation was not yet implemented.
    *   Pair production kinematics were still treated as local energy deposition.

## Milestone 4: Clinical Realism

The final milestone focused on adding the necessary features to make the simulation clinically realistic and verifiable.

*   **Goal:** Move from a physics MVP to a tool suitable for validation against reference Monte Carlo codes.
*   **Key Features:**
    *   **Clinically-relevant materials:** A system for mapping Hounsfield Units (HU) from CT images to material definitions with elemental compositions.
    *   **Improved atomic relaxation:** Support for mixtures in the relaxation mapping.
    *   **Uncertainty outputs:** Batch-based standard error calculation for dose outputs.
    *   **Validation harness:**
        *   Scripts to create reference phantoms (e.g., water slab with bone insert, NEMA phantom).
        *   A tool to compare the simulation output against a reference dose volume from Geant4/GATE.
