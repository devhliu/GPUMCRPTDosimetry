Phase 9: where to paste in `engine_gpu_triton.py`

## Insert Block 1: initialize vacancy queue + AR config
Place near the start of `run_one_batch()` (or equivalent), before the photon loop:

- `vac_q = _empty_vac_q(device)`
- `ar_cfg = AtomicRelaxationConfig(...)`

## Insert Block 2: PE dispatch block
Place inside the photon real-interaction stage after you have a packed real-collision queue and an interaction id array:

Inputs expected:
- `ph_real["pos_cm"], ph_real["dir"], ph_real["E_MeV"], ph_real["w"], ph_real["rng"], ph_real["ebin"]`
- `ph_itype` with PE-coded entries (`PE_CODE`)
- `q_e` electron queue (or None)
- `vac_q` vacancy queue

Outputs:
- appended photoelectrons in `q_e`
- appended vacancies in `vac_q`

## Insert Block 3: post-photon-step atomic relaxation
Place after all photon interactions for the *iteration*:

- consumes `vac_q`
- appends fluorescence photons to the photon queue
- appends Auger electrons to electron queue
- deposits below-cutoff relaxation energy locally

## Critical TODOs
1) Set `PE_CODE` to match your classifier encoding.
2) If your photon/electron queues require `ebin`, compute it for emitted X-rays/Auger electrons (recommended).
3) Replace placeholder LCG RNG in kernels with your repo RNG helper for determinism.