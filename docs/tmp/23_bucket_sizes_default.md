# Default CUDA graph buckets

Default bucket sizes (as requested):

```text
[4096, 16384, 65536, 262144, 1048576]
```

## Notes
- `1048576` (1M) is intended for “large batch” regimes; it will allocate larger static buffers and increases memory footprint.
- If GPU memory becomes tight, remove the 1M bucket and rely on 262k, or reduce per-batch history count.

Recommended micro-steps for starting:
- photons: 2
- electrons: 2

Then tune:
- increase photon micro-steps when Σ_max is small (long Woodcock flights)
- increase electron micro-steps cautiously due to atomic scoring contention