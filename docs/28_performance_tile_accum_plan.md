# Performance: reduce e-/e+ continuous-loss atomic contention

## Problem
Electron/positron condensed-history step uses:
- `tl.atomic_add(edep_flat[lin], dE*w)`

Near compact sources, large numbers of charged steps hit the same (or nearby) voxels, causing heavy atomic contention and slowing down kernels.

This is explicitly called out in `physics_rpt_design4GPUMC.md` §4.7.

## Solution options (increasing complexity)

### Option 1 (MVP, practical): sort-by-voxel (or tile) + run-length reduce
1. During condensed-history kernel, write per-step deposits into a temporary list:
   - `lin[i]` voxel linear index
   - `val[i] = dE*w`
2. Sort by `lin` (or by coarse `tile_id = lin >> tile_shift`).
3. Do run-length reduction to sum `val` for identical `lin`.
4. Do a final atomic add for each unique `lin` (much fewer atomics).

Pros:
- Immediately effective, simple to validate.
Cons:
- Sorting cost can be high; must be applied selectively (only if queue is large and/or hotspots strong).

### Option 2 (GPU-native): block-local tile accumulation then flush
1. Bin particles into tiles (e.g., 8×8×8 voxels blocks) using a key.
2. Launch one block per tile, accumulate into shared memory tile array.
3. Flush tile array to global `edep` with fewer atomics.

Pros:
- Avoids full sort; very fast when tuned.
Cons:
- Requires a robust binning and scheduling system; more code.

### Option 3 (hybrid): occasional sorting by Morton tile key
- Sort queue by a key that combines:
  - energy bin, material class, and Morton(tile)
- Improves both table locality and tally locality.

## Recommended implementation sequence
1. Add a config switch:
   - `electron_transport.tally_mode = "atomic" | "sorted_voxel"`
2. Implement sorted_voxel in the orchestrator using torch sort and scatter-add (GPU).
3. Add heuristics to enable it only when beneficial:
   - if `N_electrons > N_threshold` and/or `atomic_contention_metric` is high.

---