# Phase 5 (Triton) — Implemented vs Remaining

## Implemented in this update
1. **Prefix-sum compaction** (Phase 5 MVP)
   - `src/gpumcrpt/transport/triton/compaction.py`
   - Used when `use_prefixsum_compaction: true`
2. **Photon real-collision classification hook**
   - `photon_classify_kernel` computes PE/Compton/Rayleigh/Pair based on partial macroscopic XS.
3. **Photoelectric kernel** (MVP)
   - spawns a photoelectron with `T=Eγ` (binding ignored; binding can be added as local deposit later)
4. **Electron condensed-history step kernel**
   - continuous loss `dE = S_restricted * ds`
   - step limited by voxel fraction and range fraction
   - atomic scoring via `tl.atomic_add` to flattened edep

## Still remaining (next increments)
- Compton kernel (inverse-CDF sampler from `.h5`, else fixed-iteration fallback in Triton) + recoil electron + scattered photon.
- Rayleigh kernel (tabulated sampler; for now can isotropically scatter).
- Pair kernel (tabulated split; create e-/e+, no local 1.022 MeV).
- Brems/delta secondaries:
  - emit based on `P_brem_per_cm`, `P_delta_per_cm`
  - sample spectrum from `.h5` inverse-CDF and subtract from parent energy.
- Positron transport + annihilation at rest (2×511 keV photons).
- Optional sorting/coherence (hook in orchestrator; use `torch.sort` MVP, replace later).
- CUDA graphs capture (optional via config once kernel sequence stabilizes).

## Important note
Current GPU path is still **not energy-conserving** because Compton/Rayleigh/Pair and brems/delta emission are not yet spawned/processed.
Use GPU path now only as a scaffolding/performance baseline; physics validation requires completing the missing kernels above.