# Phase 7 note: brems/delta parent energy subtraction

Brems/delta emission kernels already compute:
- `E_secondary = Efrac * E_parent`
- `E_parent' = E_parent - E_secondary`

However, in the current Phase 7 orchestrator wiring, the updated parent energies are **not yet written back** into the electron queue because we compacted emitters without retaining their indices into the post-step survivor queue.

## Why this matters (physics correctness)
Per `physics_rpt_design_principle.md` and `physics_rpt_design4GPUMC.md`:
- When emitting brems photons or δ-rays, the emitted energy must be subtracted from the parent charged particle so it is not deposited twice.

## Next action (Phase 7.1)
Implement indexed emission:
1. Carry an `id` field per particle through the electron graph replay (bucket buffer).
2. When `emit_brem` / `emit_delta` is true, collect emitter ids and apply an indexed update:
   - `electrons["E_MeV"][id] = new_parent_E`
3. Then append secondaries.

This keeps energy bookkeeping correct without reintroducing heavy host synchronization.