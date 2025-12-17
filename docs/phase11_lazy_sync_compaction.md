Phase 11 (Triton scan, inplace_pack), **Lazy Sync (one CPU sync per step)**

This implements the strategy you described:

1) **One CPU sync at step start**:
   - Read `n_dirty = global_counters[idx].item()` for photon/electron/vacancy.
   - Use `n_dirty` to size all launches safely.

2) **Oversubscribed kernels**:
   - Launch with `n_dirty`.
   - Threads beyond true live region see `status=0` and early-exit.
   - No second sync needed to learn `new_n`.

3) **Blind compaction**:
   - `mask = (status==1)` for `[0:n_dirty]` via Triton kernel
   - Exclusive scan mask via your existing Triton scan workspace
   - Compute `new_n` as a GPU scalar:
     - `new_n = prefix[n_dirty-1] + mask[n_dirty-1]`  (stored in ws.total_i32[0])
   - Scatter indices then pack into dst bank using a *guarded pack* kernel that reads `new_n` from GPU.
   - Update `global_counters[idx] = new_n` on GPU.

4) **Ping-pong banks**:
   - compaction writes into the alternate bank, avoiding in-place hazards.
   - swap pointers at end of step.

## Notes / Requirements
- This avoids any `.item()` except the single `n_dirty` read at start of step.
- Uses one additional `int32[1]` scalar per bank workspace (`ws.total_i32`) to carry `new_n` through GPU.
- Your physics kernels should treat `status!=1` as inactive (early exit) to make oversubscription safe.

## Why guarded pack is needed
Triton grid sizes are CPU integers.
We can still pack without knowing `new_n` on CPU by:
- launching pack with `n_dirty` threads
- reading `new_n` from a GPU scalar inside the kernel
- masking `j < new_n`

This yields correct packing with no second synchronization.

## Next recommended micro-optimization
You can remove the intermediate `idx_i32` list by doing direct scatter-pack from (mask,prefix),
but the idx-list version usually improves memory coalescing for heavy SoA banks.