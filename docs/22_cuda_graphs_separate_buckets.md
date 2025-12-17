# Bucketed CUDA graphs (separate photon/electron replays)

You chose: **Separate** graphs for photons and electrons.

## Behavior
Per outer wavefront iteration:
1. **Replay photon graph** for `photon_micro_steps` (Woodcock flight only).
2. Compact/split real collisions outside the graph:
   - photoelectric / Compton / Rayleigh / pair kernels
3. **Replay electron graph** for `electron_micro_steps` (condensed-history only).
4. Compact + handle brems/delta emission outside the graph.
5. Repeat until queues empty or max iterations.

## Pros
- Less padding waste if photon and electron queue sizes differ significantly.
- Separate tuning of micro-step counts for photons vs electrons.

## Cons
- 2 graph replays per outer iteration (higher replay count).

## Practical tuning
- Start with `photons: 2`, `electrons: 2`
- For long photon flights (low Σ_t), you can increase photon micro-steps (e.g., 4–8)
- For electron-heavy problems, increase electron micro-steps cautiously due to atomic scoring contention
