# Phase 8.1 + 8.2: Hashed-tile tally (R1) improvements

This update advances the §4.7 "tile-based reduction" path toward the non-functional requirements §1.2 / §5.3:

## Phase 8.1 — decouple from compaction internals (no `return_prefix=True`)
Previously, building `active_bins` required `prefix_sum_compact(mask, return_prefix=True)` (needing prefix positions).

Now:
- `active_mask = (bin_counts > 0)`
- `active_prefix = exclusive_scan_int32(active_mask.int32())`
- a Triton kernel writes a **padded active list**:
  - `active_bins[active_prefix[i]] = i` for active bins
  - unused entries remain `-1`

This removes:
- `torch.nonzero`
- `.item()` / CPU sync
- dependency on a particular compaction implementation detail

## Phase 8.2 — bucket-capacity workspace (no per-iteration allocation)
We introduce a cached workspace keyed by:
- device
- `n_bins_pow2`
- `capacity` (typically the graph bucket size)

Buffers such as `out_lin/out_val` are allocated once at bucket size.
Each wavefront tally call processes only the first `N_good` events and ignores the remainder.

This reduces per-iteration overhead and makes it possible (later) to capture tally kernels inside CUDA graphs if desired.

## Current hashed-tile tally pipeline
1. Histogram hashed bins (`bin_counts`) via Triton
2. Exclusive scan of `bin_counts` to create `bin_offsets` via Triton scan
3. Scatter events into `out_lin/out_val` via Triton cursors
4. Exclusive scan of `active_mask` to build padded `active_bins` list (no sync)
5. Reduce each active bin using R1 per-bin hash reducer; empty padded entries early-return

## Config reminder
Recommended defaults for RTX 4090:

```yaml
electron_transport:
  tally_mode: hashed_tile
  hashed_auto: true
  min_events: 200000

  tile_shift: 9
  n_bins_pow2: 131072

  hash_block: 256
  hash_H: 256
  hash_probes: 8
  hash_num_warps: 4

  scan_block: 1024
```