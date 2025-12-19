Phase 10 final wiring notes (your engine schema)

## What this wiring block assumes (matches your message)
- SoA banks for photons/electrons
- `status:int8` (1 alive, 0 dead)
- `self.global_counters:int32` with indices:
  - 0 photon_count
  - 1 electron_count
  - 2 vacancy_count
- Vacancy bank includes:
  - x,y,z (float32)
  - atom_Z (int32)
  - shell_idx (int32)
  - status (int8)
  - and for correctness in Domain B relaxation: w + Philox RNG SoA fields

## Important mismatch resolved: rng_offset vs Philox SoA
Your banks currently store `rng_offset:int64` (single scalar), but Phase 10 kernels use Philox SoA:
- rng_key0, rng_key1, rng_ctr0..3

This patch includes a transitional bridge:
- key0/key1 fixed constants
- ctr0 derived from rng_offset low 32 bits
- ctr1..ctr3 = 0

For best performance and determinism across stages, you should eventually store
rng_key0/rng_key1/rng_ctr0..3 in the banks directly and delete the rng_offset bridge.

## Engine methods you must route
The wiring file references 5 methods you already effectively have in your engine:
- _compact_photons_to_active_indices()
- _compact_vacancies_to_active_indices()
- _compact_all_banks_end_of_step()
- _photon_interaction_type_for_active()
- _compute_photon_ebin_for_bank_indices()

Implement these by connecting to your existing:
- prefix-sum compaction / stream compaction
- interaction classifier output
- log-uniform ebin computation kernel

Once connected, PE -> vacancy -> relaxation will run fully in Bank mode B.

## Recommended default settings
- Shell count S=4 (K, L1, L2, L3)
- photon_cut_MeV = 0.005 (5 keV)
- e_cut_MeV = 0.010 (10 keV)
- log-uniform bins via (common_log_E_min, common_log_step_inv)