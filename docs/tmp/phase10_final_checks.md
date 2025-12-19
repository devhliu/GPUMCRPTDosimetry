# Phase 10 final checks (Option A)

This patch is "Phase 10 finished" for PE + vacancy + relaxation under the **Bank + active_indices** architecture.

## 1) No CPU sync in hot loop
- No `.item()` in `step()`
- Bank growth uses `tl.atomic_add()` into `self.global_counters` (GPU int32)

## 2) RNG correctness
- Each particle and vacancy carries:
  - `rng_key0, rng_key1, rng_ctr0..ctr3` (int32)
- Splitting is deterministic:
  - PE: photoelectron and vacancy inherit *advanced* RNG state from PE kernel output
  - Relaxation: X-ray/Auger inherit the updated RNG state after sampling

## 3) Energy binning
- `ebin` stored in each ParticleBank
- Appending kernels compute `ebin` using:
  - `tables.common_log_E_min`
  - `tables.common_log_step_inv`
  - `tables.NB`

## 4) Tables expected
### PE by material:
- `tables.relax_shell_cdf[M,S]`
- `tables.relax_E_bind_MeV[M,S]`
- `mat_table.atom_Z[M]`

### Relaxation by atomic Z:
- `tables.relax_fluor_yield[Zmax+1,S]`
- `tables.relax_E_xray_MeV[Zmax+1,S]`
- `tables.relax_E_auger_MeV[Zmax+1,S]`

Recommended: `S=4` (K,L1,L2,L3)

## 5) Compaction contract (engine-owned)
`compact_photons/compact_electrons/compact_vacancies` must:
- Pack ALIVE entries (status==1)
- Produce active_indices (int32) or update-bank-in-place + return range indices
- Update `self.global_counters[0/1/2]` accordingly (on GPU)

## 6) Physics bookkeeping constraints
- PE kernel MUST NOT deposit full Eγ locally (only electron + vacancy created)
- Relaxation deposits only below cutoff products locally; above cutoff are appended as particles.
- Cutoff termination elsewhere: deposit remaining kinetic energy locally (Domain B rule §12)