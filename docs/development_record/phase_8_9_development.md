# Development Record: Phases 8 & 9

This document records the development process for Phases 8 and 9, which focused on optimizing the dose tallying process and upgrading the random number generation (RNG) system.

## Phase 8: Advanced Tallying Modes

Phase 8 was dedicated to improving the performance and reducing the contention of the dose deposition (tallying) step, which can be a bottleneck in simulations with "hot spots" of high dose. The simple `atomic_add` operation can lead to significant contention in these scenarios.

### Key Developments in Phase 8:
1.  **"Record Mode" Kernels:** The condensed-history kernels for charged particles were modified to run in a "record mode". Instead of directly adding energy to the global dose grid, they record each energy deposition event as a `(voxel_index, energy_value)` pair into a buffer.

2.  **Sorted-Voxel Tally:**
    *   This mode was introduced to reduce atomic contention.
    *   The recorded deposition events are sorted by their voxel index.
    *   A run-length sum is performed to combine all depositions for each unique voxel.
    *   Finally, a single `index_add_` operation is performed for each unique voxel, significantly reducing the number of atomic operations.

3.  **Hashed-Tile Tally:**
    *   This is an alternative, more complex tally mode that avoids a full global sort.
    *   Deposition events are hashed into bins based on their "tile" (a small group of voxels).
    *   A per-bin reduction is performed using a small hash table, and the results are then flushed to the global dose grid.

4.  **GPU-Native Scans and Workspaces:**
    *   To make the tallying process more efficient and to remove dependencies on PyTorch operations like `torch.cumsum` and `torch.nonzero` in the hot loop, custom Triton kernels were developed to perform **exclusive scans** directly on the GPU.
    *   A **workspace manager** was introduced to pre-allocate and cache the temporary buffers needed for these scans, avoiding repeated memory allocations in the main simulation loop. This also made the tallying process more compatible with CUDA Graphs.

## Phase 9: RNG Upgrade and System Unification

Phase 9 focused on a critical component of any Monte Carlo simulation: the Random Number Generator (RNG). The goal was to replace the placeholder RNG with a more robust, performant, and deterministic solution.

### Key Developments in Phase 9:
1.  **Upgrade to Philox RNG:**
    *   The existing RNG was replaced with **Philox**, a counter-based RNG that is well-suited for parallel execution on the GPU.
    *   Counter-based RNGs are deterministic and do not require global state or expensive locking, making them ideal for a wavefront architecture.
    *   The particle data structures were updated to include the Philox state (keys and counters) in a Structure-of-Arrays (SoA) layout.

2.  **RNG Interface Unification:**
    *   A significant effort was made to unify the RNG interface across different physics kernels.
    *   Kernels for atomic relaxation and the photoelectric effect were migrated to use the new Philox SoA RNG.
    *   A transitional "bridge" was created to allow legacy code to work with the new system during the migration.

3.  **GPU-Side Energy Binning:**
    *   A Triton kernel was created to compute the energy bin (`ebin`) for particles on the GPU, specifically for **log-uniform energy bins**. This is important for accurately looking up cross-section data for particles emitted from processes like atomic relaxation, which can have a wide range of energies.
