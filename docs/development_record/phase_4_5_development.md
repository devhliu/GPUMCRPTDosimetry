# Development Record: Phases 4 & 5

This document records the development process for Phases 4 and 5 of the GPUMCRPTDosimetry project, which focused on implementing the initial GPU wavefront Monte Carlo engine using Triton.

## Phase 4: Minimum Viable Product (MVP)

The primary goal of Phase 4 was to create a functional, albeit incomplete, Triton-based GPU transport engine. The focus was on establishing the core infrastructure and getting a runnable pipeline.

### Initial Skeleton
The first version of the Triton engine included:
*   A placeholder `photon_woodcock_flight_kernel` for photon propagation.
*   A `LocalDepositOnly–NoTransport (MVP)` orchestrator as a starting point.
*   Basic queue compaction using `torch.boolean_indexing`, which was recognized as a temporary solution for correctness, to be replaced later for performance.

### Identified Next Steps for MVP Completion
The initial MVP was intentionally incomplete. The development plan laid out the following next steps:
1.  **Energy Binning:** The placeholder kernel used a fixed `ebin=0`. This needed to be replaced with a proper energy binning mechanism, such as a binary search in Triton or pre-computation with `torch.bucketize`.
2.  **Interaction Classification:** The kernel could flag that a "real" collision occurred, but did not yet process the interaction. The plan was to add kernels for the photoelectric effect and Compton scattering.
3.  **Electron Transport:** The condensed-history transport for electrons was not yet integrated into the GPU engine.
4.  **Dose Scoring:** The energy deposition tally was not yet being updated by the GPU kernels.

## Phase 5: Physics Expansion and Performance Tuning

Phase 5 aimed to build upon the MVP by adding more complete physics and starting the process of performance optimization.

### Key Implementations in Phase 5
1.  **Prefix-Sum Compaction:** The simple `torch.boolean_indexing` was replaced with a more performant prefix-sum compaction algorithm implemented in Triton. This was a key step towards a more efficient engine.
2.  **Real-Collision Classification:** The `photon_classify_kernel` was enhanced to classify real collisions into Photoelectric, Compton, Rayleigh, and Pair events based on material cross-sections.
3.  **Photoelectric Kernel:** An MVP of the photoelectric kernel was implemented. It spawns a photoelectron with kinetic energy equal to the photon's energy (ignoring binding energy for the initial implementation).
4.  **Electron Condensed-History Kernel:** The first version of the electron condensed-history step was implemented. It included:
    *   Continuous energy loss calculation.
    *   Step size limited by voxel fraction and range fraction.
    *   Atomic scoring of energy deposition using `tl.atomic_add`.

### Remaining Work at the End of Phase 5
While Phase 5 made significant progress, the engine was still not physically complete. The plan for subsequent development included:
*   **Completing Interaction Kernels:** Implementing the full kinematics for Compton scattering, Rayleigh scattering, and Pair production.
*   **Secondary Particle Generation:** Adding the logic to generate bremsstrahlung photons and delta-ray electrons from charged particle transport.
*   **Positron Transport:** Implementing positron transport and the annihilation-at-rest mechanism.
*   **Performance Optimizations:**
    *   Investigating particle sorting to improve memory access coherence.
    *   Capturing the kernel sequence in **CUDA Graphs** to reduce launch overhead.

At the end of Phase 5, the GPU path was a functional scaffold and a baseline for performance testing, but it was not yet energy-conserving due to the missing physics implementations.
