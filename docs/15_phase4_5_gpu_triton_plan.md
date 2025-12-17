# Phase 4/5: GPU Wavefront Monte Carlo with Triton (Implementation Plan)

Decision: **Use Triton** for GPU kernels.

This plan follows the physics and GPU constraints in:
- `physics_rpt_design_principle.md`
- `physics_rpt_design4GPUMC.md`

## Scope (Phase 4/5)
### Phase 4 (MVP, correctness-first)
Implement a working GPU wavefront engine with:
- Photon transport: **Woodcock (delta) tracking**
- Photon interactions: **photoelectric + Compton** (minimum)
- Electron transport: **condensed-history MVP**
- Positron: **(optional in MVP)** treated same as electron; annihilation at rest after cutoff
- Dose tally: `Edep[Z,Y,X]` (MeV) on GPU
- Batch estimator for voxelwise **relative uncertainty σ/mean**

### Phase 5 (physics expansion + tuning + validation)
Add:
- Rayleigh
- Pair production
- Bremsstrahlung photons + delta rays (hard events)
- Optional atomic relaxation from photoelectric vacancy (explicit above cutoff, local below)
- Performance tuning: compaction, (optional) sorting/coherence improvements

---

## Architecture summary (GPU wavefront)
Maintain packed SoA queues:
- photons: `pos_cm[N,3], dir[N,3], E[N], w[N], rng_state[N]`
- electrons: same
- positrons: same
- optional: vacancy queue (later)

Wavefront loop:
1. Photon Woodcock flight kernel → mask classify virtual/real collisions
2. Real collision classifier → compact into per-interaction queues
3. Interaction kernels (one per type)
4. Charged step kernel → score continuous loss and create secondaries hard events
5. Compaction between stages
6. repeat until queues empty or max iters

---

## Triton strategy
### Why Triton
- rapid iteration in Python package
- good performance for arithmetic + table lookups + compaction-like operations
- supports CUDA via JIT compilation

### What Triton will implement
- per-stage kernels (flight, classify, interaction, charged step)
- simple compaction primitives (mask → indices → scatter) using torch + triton
  - v1: use torch.nonzero / boolean indexing on GPU for compaction (correctness)
  - v2: implement prefix-sum compaction on GPU (Triton scan) or use torch.compile + CUDA primitives

> Note: Torch boolean indexing may trigger internal kernels; acceptable for Phase 4 MVP, then replace for Phase 5 performance.

---

## Deliverables
- `src/gpumcrpt/transport/triton/` kernels + orchestration
- `src/gpumcrpt/transport/engine_gpu_triton.py` engine
- Test cases comparing GPU vs CPU oracle on small phantoms
- Bench hooks for profiling

---

## Validation gates
1. Energy conservation (absorbing boundary mode for oracle; GPU should match within uncertainty)
2. No-double-count: Compton, photoelectric, brems/delta subtraction, pair no-local-1.022
3. Determinism: same seed produces same results (within expected GPU non-associativity for atomics; supported by fixed reduction patterns)
4. Dose map affine consistency: output matches activity affine

---