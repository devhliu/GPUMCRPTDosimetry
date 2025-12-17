# Phase 9.2: Replace placeholder RNG with counter-based Philox (GPU-friendly)

Per `physics_rpt_design4GPUMC.md` §4.2, transport RNG should be counter-based (Philox/Threefry):
- deterministic under parallel execution
- no global state
- no rejection loops
- fixed-cost per kernel stage

This phase upgrades atomic relaxation (and later PE shell selection) from a placeholder LCG
to a Philox4x32-style counter RNG.

## Data layout
Each particle/vacancy carries:
- `rng_key0, rng_key1` (uint32)
- `rng_ctr0..ctr3` (uint32)

This matches common Philox usage patterns.

## Notes
- If the repo already contains a Philox implementation, use that instead of `rng_philox.py`.
- This patch advances only the atomic-relaxation kernel. The PE kernel should be updated next
  to match the same RNG plan (shell selection uses one uniform).