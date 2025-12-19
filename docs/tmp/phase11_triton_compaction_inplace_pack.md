Phase 11 (scan = Triton, compact_style = inplace_pack)

This phase adds a **fully GPU-friendly** compaction pipeline for your SoA banks that matches the design docs:

- Use `status` (int8) as ALIVE/DEAD mask
- Convert to int32 mask in a Triton kernel
- Exclusive-scan the mask (reuse your existing Triton scan workspace)
- Scatter-pack *all* SoA fields from a src bank into a dst bank (ping-pong buffers)
- Update `global_counters[idx]` to new packed count

## Recommended implementation details

### A) Ping-pong banks
True in-place compaction is hard without hazards. Use ping-pong:
- `photons_A`, `photons_B`
- `electrons_A`, `electrons_B`
- `vacancies_A`, `vacancies_B`
Swap pointers each iteration.

### B) Avoid `.item()` for `n`
The placeholder code uses `count_i32.item()` which is a CPU sync.
To keep Phase 11 fully GPU-resident, implement one of:
1. Maintain `count` on CPU only at batch boundaries (still acceptable if not per-iteration), OR
2. Use a fixed launch size `cap` each pass (scan full cap) and use scanned total as GPU scalar, OR
3. Add a tiny kernel to read `count_i32` into a 1-element CPU tensor asynchronously (still a sync).

Given your design constraints (`physics_rpt_design4GPUMC.md` §1.2), the best is (2): scan full *current max* each pass and keep `count` as GPU scalar for later kernels via predicates.

If you want, I can provide the exact "scan full cap" variant that avoids needing `n` on CPU.

### C) Compute new_n without reduction
`new_n = mask.sum()` will sync if you call `.item()`.
Instead compute:
- after exclusive scan, total_alive = prefix[n-1] + mask[n-1]
We can produce this as a GPU scalar using a 1-block Triton kernel reading the last element.

## Next step after this file drop
Tell me:
- Do you prefer scanning full `cap` each compaction, or scanning only `count`?
- If scanning only `count`, do you allow one CPU sync per wavefront cycle?