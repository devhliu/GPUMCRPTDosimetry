# GPU Architecture and Transport Engines

The GPUMCRPTDosimetry simulation leverages a GPU-accelerated wavefront architecture to achieve high performance. This document provides an overview of the architecture and the different transport engines implemented.

## Wavefront Architecture with Triton

The simulation employs a "wavefront" or "history" based approach to Monte Carlo particle transport. Instead of tracking one particle at a time from start to finish, the wavefront approach processes large batches of particles (the "wavefront") in lock-step through the simulation. This is highly amenable to the massively parallel nature of GPUs.

The core of the GPU implementation is built using **Triton**, a Python-based language and compiler for writing highly efficient GPU kernels. The choice of Triton allows for rapid development and iteration directly within the Python ecosystem, while still achieving performance comparable to lower-level languages like CUDA C++.

### Particle Queues and Data Layout

Particles are managed in packed Structure-of-Arrays (SoA) queues on the GPU. Each particle type (photons, electrons, positrons) has its own set of queues, storing properties like:

*   Position (`pos_cm[N,3]`)
*   Direction (`dir[N,3]`)
*   Energy (`E[N]`)
*   Weight (`w[N]`)
*   RNG state (`rng_state[N]`)

This SoA layout is crucial for **memory coalescing**, ensuring that the GPU can read and write data from global memory in an efficient manner.

### The Wavefront Loop

The simulation proceeds in a loop, with each iteration representing a "step" for the entire wavefront of particles:

1.  **Physics Kernels:** A series of Triton kernels are launched to simulate the physics of the particles. For example, a photon wavefront might go through:
    *   A **Woodcock flight kernel** to advance the photons through the geometry.
    *   A **classification kernel** to determine if a collision occurred and what type of interaction it was.
    *   **Interaction-specific kernels** (e.g., for Compton scattering or the photoelectric effect).
2.  **Secondary Particle Generation:** If an interaction produces secondary particles (e.g., a Compton scatter producing a recoil electron), these new particles are added to their respective queues.
3.  **Compaction:** At the end of a step, "dead" particles (those that have been absorbed or have left the simulation geometry) are removed from the queues in a process called compaction. This ensures that subsequent kernels only work on active particles.
4.  **Loop Termination:** The loop continues until all particle queues are empty or a maximum number of iterations is reached.

## Evolution of Transport Engines

The project has seen the development of several transport engines, each with increasing physical realism and performance.

### 1. `LocalDepositOnly–NoTransport`
*   **Purpose:** A minimal engine to get the pipeline running.
*   **Functionality:** All particle energy is deposited locally in the voxel where the particle was created. There is no actual particle transport.

### 2. `PhotonWavefront–WoodcockFlight (Milestone 2)`
*   **Purpose:** The first real transport engine, focusing on photon physics.
*   **Functionality:** Implements Woodcock tracking for photons and handles interactions like Compton scattering, Rayleigh scattering, and the photoelectric effect. Charged secondaries are deposited locally.

### 3. `EM–CondensedHistoryMultiParticle (Milestone 3)`
*   **Purpose:** A full-featured engine that includes transport for charged particles.
*   **Functionality:**
    *   Full photon transport.
    *   **Condensed-history** stepping for electrons and positrons, which is a computationally efficient method for simulating charged particle transport in a medium.
    *   Handles **positron annihilation-at-rest**, creating two 511 keV photons.
    *   Supports the generation of **secondary particles** like bremsstrahlung photons and delta electrons, with a budgeting system to control the complexity of the simulation.

### 4. Advanced and Research Engines
*   **`EM–EnergyBucketedPersistentGraphs (Phase 7)`:** This engine introduces **CUDA Graphs** to reduce kernel launch overhead, which can be a significant bottleneck. Particles are bucketed by energy to allow for graph reuse.
*   **`EM–BankSoAVacancyRelaxation (Phase 10)`:** This engine implements a more advanced "bank" architecture and includes the physics of **atomic relaxation**, allowing for the simulation of characteristic X-rays and Auger electrons following photoelectric events. It also uses a deterministic **Philox counter-based RNG**.
*   **`EM–LazyCompactionSingleSync (Phase 11)`:** A research prototype that minimizes CPU-GPU synchronization by using a "lazy compaction" scheme, further boosting throughput.
