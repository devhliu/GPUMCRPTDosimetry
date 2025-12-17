# Phase 8 (hashed tile tally): remove `torch.nonzero` and `torch.cumsum`

This step keeps the hashed-tile (Option B2) tally reduction **fully GPU-native** and removes
two frequent PyTorch ops from the hot loop:

- `torch.nonzero(bin_counts)` to build `active_bins`
- `torch.cumsum(bin_counts)` to build `bin_offsets`

## New approach

### 1) `bin_offsets` via Triton exclusive scan (int32)
We implement `exclusive_scan_int32(bin_counts)` using two-pass block scan:
- block-local exclusive scan
- scan block sums
- add block offsets

This computes `bin_offsets[b] = sum_{k<b} bin_counts[k]`.

### 2) `active_bins` via prefix-sum compaction (padded)
We construct a padded active list `active_bins[int32, n_bins]` where inactive entries are `-1`:

1. `active_mask = (bin_counts > 0)`
2. `prefix_sum_compact(active_mask, return_prefix=True)` provides per-element compact positions `prefix`
3. A Triton kernel writes:
   - `active_bins[prefix[i]] = i` for each active bin `i`
4. The reducer launches with a **fixed grid** `n_bins` and exits early for `active_bins[pid] == -1`.

This avoids `.item()` and avoids dynamic grid launch sizes, respecting
`physics_rpt_design4GPUMC.md` non-functional requirement §1.2.1 and §5.3.

## Remaining non-graph operation
The tally kernels are still run outside the CUDA graphs. If desired, their kernels can be bucketed too, once:
- `n_bins_pow2` and buffer sizes are fixed
- record buffers are fixed per bucket

## Config reminder
Recommended start:

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
  scan_block: 1024
```