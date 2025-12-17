"""
Extend prefix_sum_compact to optionally return the per-element prefix positions.

Existing:
  idx, count = prefix_sum_compact(mask)

New:
  idx, count, prefix = prefix_sum_compact(mask, return_prefix=True)

`prefix` is int32 same length as mask, where prefix[i] is the exclusive prefix sum
(i.e. compact write position) for element i.
"""
from __future__ import annotations

from typing import Tuple
import torch

# NOTE: keep your current implementation; add return_prefix plumbing.
# Below is only interface scaffolding, because your repo already has a working scan.
# Wire it to return the scan buffer it already computes internally.

def prefix_sum_compact(mask: torch.Tensor, return_prefix: bool = False):
    """
    Returns:
      idx: int64 packed indices (capacity N, may be larger than count)
      count: int32 scalar tensor (#true)
      prefix (optional): int32 [N] exclusive prefix sums
    """
    # --- USE YOUR EXISTING IMPLEMENTATION HERE ---
    # Suppose existing code produces:
    #   prefix_i32: [N] exclusive scan of mask
    #   count_i32: scalar
    #   idx_i64: packed indices
    #
    # Then:
    #   if return_prefix: return idx_i64, count_i32, prefix_i32
    #   else: return idx_i64, count_i32
    raise NotImplementedError("Wire return_prefix=True to existing scan output")