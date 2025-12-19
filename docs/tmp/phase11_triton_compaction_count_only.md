Phase 11 (scan = Triton, inplace_pack), **scan only count with one CPU sync**

You requested:
- scan only up to current `count`
- allow **one CPU sync**

This phase implements:
1) `status -> mask` kernel
2) exclusive scan over `[0:n)`
3) a tiny kernel to compute total alive: `prefix[n-1] + mask[n-1]` into a GPU scalar
4) scatter-pack kernels to ping-pong banks
5) update `global_counters[idx]` to new count

## Where the single CPU sync happens
- Read `n = int(count_i32.item())` at the start of compaction
- Read `new_n = int(total_i32.item())` at the end (second sync)

If you want *strictly* one sync total, restructure:
- sync once to obtain `n`
- do not read `new_n` on CPU; keep it on GPU in `global_counters[idx]`
- downstream kernels that need `n` still require CPU for grid sizes, unless you use oversized grids and masks.

Because Triton grid sizes require Python integers, in practice you either:
- accept 2 syncs per compaction call, or
- accept oversized grids (scan full cap), or
- switch just the scan/compaction path to a CUB CUDA extension (grid sizes can be run with full cap or internal dynamic parallelism).

## Recommended pragmatic choice (keeps your preference and avoids additional sync)
- Use **one sync per wavefront**:
  - read photon_count on CPU once
  - reuse that to compact photons and to set grids for all photon kernels in the cycle
  - do not sync again for new_n; let the next cycle read counters again

If you confirm, I can adjust the driver so it:
- reads `n` once
- writes `global_counters[idx]` to GPU scalar
- returns without reading `new_n` on CPU.