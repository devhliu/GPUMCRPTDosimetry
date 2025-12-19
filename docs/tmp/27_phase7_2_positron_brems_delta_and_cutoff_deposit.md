# Phase 7.2: Positron brems/delta emission + GPU cutoff deposition

## Positron brems/delta emission (wired)
We now treat positrons the same way as electrons for hard events:
- brems photons: sampled from `brem_inv_cdf_Efrac` (Efrac in [0,1])
- delta electrons: sampled from `delta_inv_cdf_Efrac`
- **parent subtraction is applied** using per-particle `id` so emitted energy is not double-counted

This matches the energy bookkeeping rules in:
- `physics_rpt_design_principle.md` (cutoff + no-double-count)
- `physics_rpt_design4GPUMC.md` §2.6 (Brems / δ-ray rule)

## GPU cutoff-local-deposit (performance upgrade)
Previously, some cutoff termination paths used host-side `index_add_` to deposit energy.
This causes CPU↔GPU sync and blocks CUDA graphs.

Now:
- cutoff deposition uses `deposit_local_energy_kernel` (Triton) with `tl.atomic_add`
- keeps execution asynchronous and graph-friendly

### What uses this kernel
- photon cutoff: deposit remaining photon energy below `E_cut_γ`
- electron cutoff: deposit remaining electron kinetic energy below `E_cut_e`
- (positron cutoff is handled via the annihilation-at-rest kernel, which deposits remaining kinetic energy + emits 2×511 keV)

## Next performance steps
1. Move more operations into graphs (e.g., cutoff deposition and some interaction kernels).
2. Reduce atomic contention for electron/positron continuous loss scoring (tile-based reduction, optional sorting by voxel tile).
3. Replace the current prefix-sum compactor with a fully in-graph scan to remove graph breaks.