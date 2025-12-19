# Sorted-voxel / tile tally mode (contention reduction)

This implements the **§4.7 strategy** from `physics_rpt_design4GPUMC.md` in a practical MVP form.

## Motivation
Per-step `atomic_add` into `Edep` can bottleneck near sources (hot voxels).
This mode reduces atomics by:
1. recording `(lin, val=dE*w)` during charged steps
2. sorting by a key that groups voxels (optionally by tile first)
3. run-length summing identical voxels
4. doing one `index_add_` per unique voxel

## Config
```yaml
electron_transport:
  tally_mode: sorted_voxel   # or "atomic"
  tile_shift: 9              # 8x8x8 voxels per tile -> 512 -> shift 9
  min_events: 200000
```

## Notes
- When `tally_mode=sorted_voxel`, the electron/positron condensed-history step uses **record-mode kernels**
  and therefore cannot currently be captured in CUDA graphs (record buffers are dynamic).
- Photon transport remains CUDA-graph accelerated.

Next performance step:
- bucket and preallocate the record buffers (`rec_lin`, `rec_val`) so charged steps can be graph-captured again.