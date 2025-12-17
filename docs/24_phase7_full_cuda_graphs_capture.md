# Phase 7: Expand bucketed CUDA graphs coverage (photons/electrons separated)

We completed Phase 6 scaffolding:
- bucket sizes: `[4096, 16384, 65536, 262144, 1048576]`
- separate graphs for photon micro-cycles and electron micro-cycles

## Goal of Phase 7
Move from “micro-cycles only” to **capturing a larger fixed sequence** while keeping correctness:
1. Photon graph: `Woodcock flight → classify interaction` (still no compaction inside graph)
2. Electron graph: `condensed step → flag brems/delta` (already done)
3. Outer loop remains responsible for:
   - compaction and per-interaction kernels (PE/Compton/Rayleigh/Pair)
   - emission kernels (brems/delta) and queue append
   - cutoff handling and positron annihilation-at-rest

## Why this phase matters
- Reduces Python overhead further by adding **classification work** into the captured graphs.
- Keeps dynamic queue operations outside the graphs.

## Constraints (non-negotiable)
- No allocations, no `.item()`, no shape changes inside graphs.
- All buffers used inside graphs must be bucket-sized static tensors.

## Next phase after Phase 7
Phase 8: in-graph compaction (prefix sum) + more kernels captured, enabling near fully GPU-resident wavefront cycles.
