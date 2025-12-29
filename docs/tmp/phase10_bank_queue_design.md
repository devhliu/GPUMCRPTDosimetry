````markdown
# Phase 10 (Bank + active_indices): completed design & implementation blocks

You confirmed mode **B**:
- single preallocated **Bank** tensors per particle type (photons/electrons/vacancies)
- **active_indices** drives which entries are processed each iteration
- kernels **atomic_add** to a global counter and append new entries to the tail
- end-of-iteration **compact()** removes dead entries and produces new active_indices (defrag)

This phase finishes:
1. **Bank-append API** for Photoelectric and Atomic Relaxation products (X-rays, Auger)
2. **Philox SoA RNG end-to-end** for PE + relaxation (no legacy rng hot-path)
3. **log-uniform ebin** computation for appended X-rays/Auger in-kernel (so no extra compaction-time pass needed)
4. A recommended **compaction contract** (what flags to mark, what to compact)

Recommended constants:
- Shell count `S=4` (K, L1, L2, L3)
- Relaxation: single-step (bounded, GPU-friendly). Cascade extension can be Phase 11.

Below are the file blocks you can paste into your repo in Phase 10.