# Phase 9.3: RNG interface unification (Philox SoA) for relaxation

Per `physics_rpt_design4GPUMC.md` §4.2, RNG should be counter-based and deterministic.

## Problem
Earlier Phase 9 blocks used a legacy `rng:int32` for convenience.  
Atomic relaxation kernel has now been upgraded to **Philox SoA** fields:

- `rng_key0, rng_key1`
- `rng_ctr0, rng_ctr1, rng_ctr2, rng_ctr3`

## Solution (this phase)
`run_atomic_relaxation()` accepts either:
- legacy `vac_q["rng"]` (int32), or
- Philox SoA fields

If legacy fields are present, it **upgrades** to Philox SoA in a transitional way using a seed.
This lets you keep the engine compiling while migrating other kernels.

## Next recommended step
Update the photoelectric kernel to also use the same Philox SoA RNG so:
- shell selection draws are deterministic and consistent
- no legacy RNG fields remain in hot path