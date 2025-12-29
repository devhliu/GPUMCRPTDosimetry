````markdown
# Next phase after Phase 10: Phase 11 (recommended)

Phase 10 (Option A) finishes the **data model + deterministic RNG + atomic append + vacancy relaxation** plumbing.

The next phase should focus on **correctness gates + performance stability**, not adding more physics yet.
This aligns with:
- `physics_rpt_design4GPUMC.md` §1.2 (avoid sync/allocation), §4.2 RNG determinism, §8 validation
- `physics_rpt_design_principle.md` §15 checklist

I recommend Phase 11 be: **GPU stream compaction + overflow safety + energy bookkeeping validation**.

## Phase 11 deliverables
1) **GPU compaction kernel(s)** for ParticleBank and VacancyBank that:
   - pack all fields (SoA) using prefix sums
   - produces active_indices or packs in-place to the head of the Bank
   - updates `global_counters[idx]` on GPU
   - does not allocate each iteration (workspace reuse)

2) **Capacity/overflow safety** around atomic appends:
   - detect overflow `dst_idx >= cap` without corrupting memory
   - accumulate an overflow flag/counter in `global_counters` (or separate scalar)
   - allow engine to either (a) abort, (b) fall back to multi-pass, or (c) re-run with larger banks

3) **Energy bookkeeping validation counters** (debug mode):
   - track emitted energy, deposited energy, escaped energy
   - confirm: emitted ≈ deposited + escaped (per batch)
   - confirm no double counting for PE:
     - initial photon energy = photoelectron kinetic + relaxation energy (+ escape)
   - these should be GPU-accumulated scalars to avoid sync per step

4) **Replace “manual vacancy counter reset” with compaction-consistent reset**
   - In Phase 10 you reset vacancy counter to 0 after relaxation.
   - Phase 11 makes this robust by:
     - tracking `vacancy_count` only as "tail pointer"
     - compaction sets it to number of ALIVE vacancies after processing (normally 0)

This phase is a prerequisite for:
- Phase 12: photon Woodcock flight + accept/reject + classify in fully branch-minimized way
- Phase 13: electron condensed-history step + hard event queues
````markdown