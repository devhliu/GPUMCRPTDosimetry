# Development Record: Phases 6 & 7

This document outlines the development process for Phases 6 and 7, which focused on performance optimization through CUDA Graphs and the implementation of more complete physics for charged particles.

## Phase 6: Bucketed CUDA Graphs

The primary goal of Phase 6 was to integrate **CUDA Graphs** into the transport engine to reduce kernel launch overhead. Since CUDA Graphs require static tensor shapes and the particle queues are dynamic, a **bucketing** strategy was adopted.

### Key Features of Phase 6:
1.  **Bucketed Graph Capture:**
    *   The engine pre-captures a set of CUDA Graphs for different maximum queue sizes (e.g., 4k, 16k, 64k particles).
    *   At runtime, the engine selects the smallest graph bucket that can accommodate the current number of active particles.
    *   Particle queues are padded to the bucket size before replaying the graph.

2.  **Separate Graphs for Particle Types:**
    *   To improve efficiency, especially when the number of photons and charged particles differs significantly, separate graphs are captured for photon and electron/positron transport.

3.  **Micro-Cycle Capture:**
    *   Initially, only the "hottest" and most stable parts of the simulation were captured in graphs. This included:
        *   The **photon Woodcock flight kernel**.
        *   The **electron condensed-history step kernel**.
    *   More complex logic, such as interaction classification, secondary particle generation, and queue compaction, remained outside the graphs. This was a crucial decision to maintain correctness while still gaining a performance advantage.

## Phase 7: Expanding Graph Coverage and Physics Realism

Phase 7 built upon the Phase 6 foundation by expanding the scope of what was captured in the CUDA Graphs and by implementing more of the critical physics for charged particles.

### Key Developments in Phase 7:
1.  **Expanded Graph Capture:**
    *   The photon graph was expanded to include the **interaction classification** step (`Woodcock flight → classify interaction`). This further reduced the amount of work done by the Python-based outer loop.
    *   The engine was made ready for **sorted-voxel tallying** with CUDA Graphs by adding a "record mode" to the charged particle kernels. Instead of using atomic adds for dose deposition inside the graph (which can cause contention), the kernels record the energy depositions to a buffer. The final tallying is then performed outside the graph.

2.  **Physics Correctness and Completion:**
    *   **Parent Energy Subtraction:** A critical energy conservation bug was addressed. When a bremsstrahlung photon or a delta-ray is emitted, the emitted energy is now correctly subtracted from the parent charged particle.
    *   **Positron Transport:** Full condensed-history transport for positrons was implemented, using the same stepping logic as electrons.
    *   **Annihilation-at-Rest:** The physics of positron annihilation was added. When a positron stops, its remaining kinetic energy is deposited locally, and two 511 keV photons are generated and added to the photon transport queue.
    *   **GPU-based Cutoff Deposition:** The deposition of energy from particles that fall below the energy cutoff was moved into a Triton kernel using `tl.atomic_add`. This replaced a host-side operation, further reducing CPU-GPU synchronization and making the process graph-friendly.

By the end of Phase 7, the engine was significantly more performant and physically complete, with a robust CUDA Graph implementation and a more accurate simulation of charged particle physics.
