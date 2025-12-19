Phase 11 update: Triton scan workspace (no torch.cumsum)

You asked explicitly to use the **Triton scan workspace** rather than `torch.cumsum`.
The `lazy_compact_*_pingpong` functions now do:

1) `mask = (status==ALIVE)` via Triton kernel
2) `exclusive_scan_int32_into(mask)` using your scan workspace (`Int32ScanWorkspace`)
3) compute `total_alive` on GPU: `prefix[n-1] + mask[n-1]`
4) scatter indices `idx[prefix[i]] = i`
5) guarded pack into dst bank with an oversubscribed kernel (`n_dirty` threads) that reads `total_alive` from GPU

Only **one** `.item()` remains: reading `n_dirty` at step start, as per your policy.

If you want to remove that last sync later, the next step would be:
- scan full capacity or
- implement a CUDA extension that can launch with GPU-known counts.