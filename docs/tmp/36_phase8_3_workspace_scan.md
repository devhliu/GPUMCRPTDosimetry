# Phase 8.3: workspace-driven scans (hashed-tile tally)

This phase removes per-call allocations inside the int32 scan path used by hashed-tile tally.

## What changes
We introduce `Int32ScanWorkspace` and `exclusive_scan_int32_into(...)`:

- `exclusive_scan_int32_into(x, ws, out=...)` performs an exclusive scan using Triton kernels
- all intermediate buffers (`block_sums`, `block_offsets`, `tmp`) are preallocated
- the scan is deterministic and GPU-native

## Why it matters
Even though scanning `n_bins=131072` is small compared to transport, executing it *every wavefront iteration* can create measurable overhead if it allocates each time.

By moving scans into a cached workspace (keyed by `(device, n_bins, capacity, scan_block)`), we:
- avoid frequent dynamic allocations (non-functional req §1.2)
- improve stream/graph friendliness
- make it easier to capture tally kernels under CUDA graphs later (fixed buffers/launches)

## Remaining allocation (next phase)
The current Phase 8.3 implementation still casts `active_mask_i32 -> int8` for `build_active_bins_padded_kernel`.

Next phase (8.4) will remove this by:
- generating `active_mask_i8` directly in the `mask_gt0_to_i32_kernel` replacement, producing both:
  - `active_mask_i32` for scan
  - `active_mask_i8` for padded active list construction