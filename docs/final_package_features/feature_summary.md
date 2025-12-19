# Final Package Feature Summary

This document provides a high-level overview of the key features of the GPUMCRPTDosimetry package. The system is a GPU-accelerated Monte Carlo simulation tool for radionuclide dosimetry, designed for performance, accuracy, and clinical research use.

## Core Engine and Architecture

*   **GPU-Accelerated Monte Carlo:** The core of the package is a high-performance transport engine that runs on NVIDIA GPUs.
*   **Triton-Based Kernels:** All GPU kernels are written using Triton (version 3.5.1), which allows for rapid development in a Python environment while achieving high performance.
*   **Wavefront Architecture:** The simulation uses a "wavefront" or "history-based" approach, processing large batches of particles in parallel to maximize GPU utilization.
*   **Structure-of-Arrays (SoA) Data Layout:** Particle data is stored in an SoA layout to ensure coalesced memory access, which is critical for GPU memory bandwidth.
*   **Advanced Compaction and Synchronization:** The engine uses a highly optimized, GPU-native compaction pipeline with a "lazy sync" policy to minimize CPU-GPU synchronization, which is a common performance bottleneck.

## Physics Simulation Capabilities

The simulation includes a comprehensive set of physics models for radionuclide decay and the transport of photons, electrons, and positrons.

### Radionuclide Decay
*   **ICRP-107 Database:** The package uses a local JSON-based database derived from ICRP-107 for radionuclide decay data.
*   **Full Decay Chains:** The simulation can model the full decay chain of radionuclides, including the emission of:
    *   Alpha particles
    *   Beta particles (from continuous spectra)
    *   Gamma photons
    *   X-rays
    *   Auger electrons
    *   Internal conversion electrons

### Photon Transport
*   **Woodcock Tracking:** Photon transport is handled using the efficient Woodcock (delta) tracking algorithm.
*   **Photon Interactions:** The following photon interactions are simulated:
    *   Compton Scattering (using Klein-Nishina kinematics)
    *   Photoelectric Effect
    *   Rayleigh Scattering
    *   Pair Production (for photons with E > 1.022 MeV)
*   **Atomic Relaxation:** The simulation includes a model for atomic relaxation following the photoelectric effect, allowing for the transport of characteristic X-rays and Auger electrons.

### Charged Particle Transport
*   **Condensed History:** The transport of electrons and positrons is simulated using a condensed-history approach.
*   **Secondary Particles:** The generation of "hard" secondary particles (bremsstrahlung photons and delta-rays) is supported, with a budgeting system to control the simulation complexity.
*   **Positron Annihilation:** The annihilation of positrons at rest is simulated, producing two 511 keV photons.

## Materials and Phantoms

*   **CT-Based Materials:** The simulation can use patient-specific material definitions derived from CT images by mapping Hounsfield Units (HU) to material properties.
*   **5-Compartment Model:** A default 5-compartment material model (Air, Lung, Fat, Muscle, Soft Tissue, Bone) is included, with elemental compositions based on ICRU Report 44.
*   **Reference Phantoms:** The package includes scripts to generate standard validation phantoms, such as the NEMA IEC body phantom and simple geometric phantoms.

## Performance and Validation

*   **High Performance:** The package is designed for high throughput, with benchmarks showing performance in the range of 20-60 seconds per 10^8 histories on an NVIDIA RTX 4090.
*   **Comprehensive Test Suite:** The package includes a suite of over 37 tests that cover physics correctness, GPU kernel performance, and end-to-end integration.
*   **Validated:** The package has been thoroughly validated for energy conservation, numerical stability, and physics accuracy.
*   **Triton 3.5.1:** The package is specifically optimized for and requires Triton version 3.5.1.
