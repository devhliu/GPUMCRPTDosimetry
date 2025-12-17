# Hashed-tile tally mode (O(N) binning, no global sort)

This mode implements the §4.7 "tile-based reduction" idea without CUB radix sort,
using only Triton kernels + PyTorch tensor ops.

## Overview
Input: recorded charged-step deposits as `(lin, val)` where:
- `lin`: voxel linear index (int32)
- `val`: deposited energy (float32), already includes weight

Algorithm (per wavefront microcycle batch):
1. Compute bin for each event:
   - tile = lin >> tile_shift
   - bin = tile & (n_bins-1)   (hashed bins)
2. Histogram bin counts (Triton atomic adds).
3. Prefix sum → exclusive bin offsets (torch.cumsum).
4. Scatter events into contiguous arrays by bin (Triton atomic cursor).
5. Reduce each *active* bin using a small hash table keyed by `lin` (Triton), then flush to `edep` with fewer atomics.

## Why hashed bins
It bounds the size of bin arrays (`n_bins`) irrespective of phantom size, and avoids a full sort.
Collisions do not affect correctness because the reduction key is the full voxel `lin`.

## Config
```yaml
electron_transport:
  tally_mode: hashed_tile
  tile_shift: 9
  min_events: 200000

  n_bins_pow2: 65536
  hash_H: 512
  hash_probes: 8
  hash_load_block: 256
  hash_block: 256
  hash_num_warps: 4
```

## Tuning guidance (RTX 4090)
- Increase `n_bins_pow2` if bins get too full (reduces reducer work).
- Increase `hash_H` if collision rate is high (but more compute per bin).
- Keep `hash_H` modest (256–512) for performance; prefer increasing `n_bins_pow2` first.