# Phase 4 MVP (Triton) — Current Status

This commit introduces the **skeleton** of the Triton GPU transport engine:
- Triton kernel: `photon_woodcock_flight_kernel` (MVP, placeholder energy binning)
- Orchestrator: `TritonTransportEngine`

## What works now
- GPU-side photon propagation with Woodcock majorant sampling
- Queue compaction with torch boolean indexing (MVP path)

## What is intentionally incomplete (next steps)
1. **Energy binning**: current kernel placeholder `ebin=0`.
   - Next: implement binary search in Triton or precompute `ebin` per particle with torch.bucketize.
2. **Interaction classification**: real collisions are flagged but not processed.
   - Next: split real events and run:
     - photoelectric: kill photon, spawn electron
     - Compton: KN via inverse-CDF (from `.h5`) or CPU-like fallback for MVP
3. **Electron condensed-history**: currently not integrated in GPU engine.
4. **Dose scoring**: edep tally is not updated yet inside GPU kernels.
5. **Compaction optimization**: replace boolean indexing with prefix-sum compaction for performance (Phase 5).

## Validation plan
- Compare GPU MVP against CPU oracle on tiny phantoms for:
  - energy conservation
  - deterministic behavior (as far as atomic ordering allows)
  - interaction bookkeeping once implemented

---