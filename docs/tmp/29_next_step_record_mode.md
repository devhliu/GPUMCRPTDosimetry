# Next step required to enable sorted voxel tally for e-/e+

To actually reduce contention, we must avoid doing atomic adds inside the condensed-history kernels.
Instead, we need a **record mode**:

## Record mode idea
Modify electron/positron step kernels to optionally:
- compute `lin` voxel index
- compute continuous loss deposit `val = dE*w`
- write `(lin, val)` to output arrays (one per particle per micro-step)
- do NOT atomic add in-kernel

Then the orchestrator calls `sorted_voxel_accumulate(edep_flat, lin, val, cfg)`.

## Why we need this
If the kernel already atomic-added, sorting afterward would double count.

## Recommended incremental path
1. Add `record=True/False` paths:
   - Keep current `atomic_add` mode as default.
2. Enable `record` mode only when:
   - `N` is large
   - expected contention high
3. Validate energy conservation on small phantoms.

---