````markdown
# Phase 11 validation gate (recommended)

Before adding more physics (Woodcock, Compton sampling, electron CH), validate the Phase 10 plumbing:

## A) Determinism
- Fix seed (`default_key0/default_key1`) and initial ctrs.
- Run identical batch twice:
  - counters and voxel edep should match bitwise (or within fp32 non-associativity if different ordering)
- Ensure split events (PE → electron + vacancy → relaxation → photon/electron) are deterministic.

## B) Energy conservation (homogeneous phantom)
Track these GPU scalars per batch:
- `E_emitted_total` (source energy excluding neutrinos)
- `E_deposited_total` (sum of edep grid)
- `E_escaped_total` (energy of particles leaving geometry, weighted)

Check: `E_emitted ≈ E_deposited + E_escaped`.

## C) No-double-count checks (PE-only mode)
Disable Compton/Rayleigh/Pair.
In a homogeneous phantom with monoenergetic photons:
- Total deposited energy should approach total emitted energy (if geometry contains them) minus escaped.
- Local deposition should *not* count the binding energy twice:
  - deposition includes binding only via relaxation (or local below-cutoff products).

## D) Overflow checks
Artificially set MAX_PHOTONS low and run:
- overflow counter should increment
- memory corruption should not occur
- engine should stop/recover cleanly

Once A–D pass, proceed to Phase 12 (Woodcock flight + classify) and Phase 13 (electron CH).