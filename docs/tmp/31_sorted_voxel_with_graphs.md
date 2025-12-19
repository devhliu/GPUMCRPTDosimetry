# Sorted-voxel tally with CUDA graphs (next step)

Previously:
- `tally_mode = sorted_voxel` used record-mode kernels **outside** CUDA graphs due to dynamic record buffers.

Now (this step):
- record buffers (`rec_lin`, `rec_val`) are **preallocated per bucket** in the graph static state.
- electron/positron micro-cycles can be **CUDA-graphed again** even in record-mode.

## Pipeline
1. Replay electron graph:
   - runs `electron_condensed_step_record_kernel`
   - writes `rec_lin/rec_val` for each lane
2. Outside graph:
   - `sorted_voxel_accumulate(edep_flat, rec_lin, rec_val)`
3. Replay positron graph:
   - runs `positron_condensed_step_record_kernel`
   - writes `rec_lin/rec_val` and `stop` flags
4. Outside graph:
   - `sorted_voxel_accumulate(...)`
   - annihilation-at-rest kernel for `stop` subset

## Remaining contention source
`sorted_voxel_accumulate` uses `torch.argsort` + `index_add_`. That is still GPU-native,
but it is not captured in graphs and has an O(N log N) cost.

## Next optimization
Replace full sort with:
- coarse binning by tile id (radix/counting) + per-tile local reductions, or
- occasional sorting only when contention heuristic indicates it helps.