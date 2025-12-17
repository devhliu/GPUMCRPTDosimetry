# Phase 7.4: Sorted-voxel tally with charged CUDA graphs (bucketed)

This step restores CUDA-graph capture for charged particle microcycles even when using `tally_mode: sorted_voxel`.

## What changed
- Electron/positron bucket graphs now include static buffers:
  - `rec_lin[int32]`, `rec_val[float32]`
- Charged step kernels run in **record mode** inside the captured graphs:
  - no `tl.atomic_add` for continuous loss
  - instead, record (lin, dE*w)

## What remains outside graphs
- `sorted_voxel_accumulate()` (torch sort + run-length sum + index_add_)
- secondary generation kernels (brems/delta) and queue append
- annihilation-at-rest kernel for stopped positrons

This follows the design guidance:
- `physics_rpt_design4GPUMC.md` §4.7 (reduce atomics in hotspots)
- Non-functional req: avoid dynamic allocations and frequent CPU↔GPU sync.

## Config
```yaml
electron_transport:
  tally_mode: sorted_voxel
  tile_shift: 9
  min_events: 200000
```

With this setting:
- photon graph runs as before
- electron/positron graphs run in record mode
- tally accumulation is reduced-contention and still GPU-native