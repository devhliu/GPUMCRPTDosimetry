Phase 10 complete (Bank architecture, Philox SoA, log-uniform bins)

## What Phase 10 delivers
- Banks upgraded to store:
  - **ebin:int32**
  - **Philox SoA RNG**: rng_key0/rng_key1/rng_ctr0..3 (int32)
- No legacy rng_offset in hot path (removed)
- Photoelectric dispatch:
  - PE photons → photoelectrons + vacancies (bank atomic appends)
  - produces vacancy records: (x,y,z, atom_Z, shell_idx, w, RNG)
- Vacancy relaxation dispatch:
  - vacancies → (fluorescence X-rays) + (Auger electrons) or local deposit
  - above cutoffs: appended to photon/electron banks with log-uniform ebin computed in append kernels
  - below cutoffs: deposited locally in edep_flat
- Fully GPU-resident control flow:
  - no `.item()` sync
  - appends use `self.global_counters` atomics
  - cleanup uses your existing compaction (prefix sum)

## Required physics tables / metadata
- Log-uniform binning:
  - `tables.common_log_E_min`
  - `tables.common_log_step_inv`
  - `tables.NB`
- PE shell tables by material:
  - `tables.relax_shell_cdf[M,S]`
  - `tables.relax_E_bind_MeV[M,S]`
  - `mat_table.atom_Z[M]` effective Z per material (int32)
- Relaxation tables by atomic Z:
  - `tables.relax_fluor_yield[Zmax+1,S]`
  - `tables.relax_E_xray_MeV[Zmax+1,S]`
  - `tables.relax_E_auger_MeV[Zmax+1,S]`
Recommended: `S=4` (K,L1,L2,L3)

## Remaining integration points (engine-specific)
You still need to hook:
1) `compact_photons/compact_electrons/compact_vacancies` (your prefix-sum compactor)
2) `get_photon_itype_for_active` (your classifier output aligned to active_indices)
3) Make sure initial source emission fills:
   - status=ALIVE
   - ebin computed (log-uniform)
   - RNG key/ctr set (deterministic seeds)

Once those are wired, PE + relaxation are end-to-end complete in the wavefront loop.