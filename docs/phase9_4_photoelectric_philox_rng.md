# Phase 9.4: Photoelectric kernel migrated to Philox SoA RNG

Goal (per `physics_rpt_design4GPUMC.md` §4.2):
- deterministic under parallel execution (counter-based RNG)
- no legacy `rng:int32` in hot path
- fixed-cost draw usage per kernel stage

## What changed
- `photon_photoelectric_kernel` now takes RNG as:
  - `rng_key0, rng_key1`
  - `rng_ctr0..rng_ctr3`
- Shell selection consumes **exactly 1 uniform** (`u0`)
- The updated counter state is written to both:
  - the produced photoelectron RNG fields
  - the produced vacancy RNG fields

## Engine changes required
- PE dispatch must pass Philox SoA RNG fields from the photon queue.
- Electron and vacancy output queues must store Philox SoA RNG fields.
- Remove legacy `rng:int32` usage in PE dispatch.

## Remaining work
- Ensure **all** other photon interaction kernels also use the same Philox SoA RNG fields (Compton, Rayleigh, pair).
- Remove the transitional `rng_bridge.py` once all stages have migrated.