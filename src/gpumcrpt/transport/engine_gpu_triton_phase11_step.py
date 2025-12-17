from __future__ import annotations

import triton
import torch

from gpumcrpt.transport.compaction_lazy_driver import (
    alloc_lazy_compaction_ws,
    lazy_compact_particlebank_pingpong,
    lazy_compact_vacancybank_pingpong,
)


def phase11_step_lazy_sync(engine) -> None:
    """
    Phase 11 step structure with your chosen policy:
    - ONE sync per step to read dirty counts
    - physics kernels launched with n_dirty (status-guarded)
    - compaction uses Triton scan workspace (no torch.cumsum)
    - compaction updates global_counters on GPU
    - ping-pong banks

    Assumes engine has:
      engine.banks.cur(), engine.banks.nxt(), engine.banks.swap()
      engine.global_counters (int32[>=3])
      engine.dispatch_photon_stages(nP_dirty)
      engine.dispatch_electron_stages(nE_dirty)
      engine.dispatch_vacancy_relaxation(nV_dirty)  (or inside photon stages)
    """
    # ---- ONE CPU sync at step start ----
    nP_dirty = int(engine.global_counters[0].item())
    nE_dirty = int(engine.global_counters[1].item())
    nV_dirty = int(engine.global_counters[2].item())

    ph_src, el_src, vac_src = engine.banks.cur()

    # ---- oversubscribed physics launches ----
    # These kernels must all early-exit when status==0.
    engine.dispatch_photon_stages(ph_src, nP_dirty)
    engine.dispatch_electron_stages(el_src, nE_dirty)
    engine.dispatch_vacancy_relaxation(vac_src, nV_dirty)

    # ---- compaction (Triton scan workspace) ----
    ph_dst, el_dst, vac_dst = engine.banks.nxt()

    lazy_compact_particlebank_pingpong(
        src=ph_src, dst=ph_dst, n_dirty=nP_dirty,
        global_counters=engine.global_counters, counter_idx=0,
        ws=engine.ws_ph,
    )
    lazy_compact_particlebank_pingpong(
        src=el_src, dst=el_dst, n_dirty=nE_dirty,
        global_counters=engine.global_counters, counter_idx=1,
        ws=engine.ws_el,
    )
    lazy_compact_vacancybank_pingpong(
        src=vac_src, dst=vac_dst, n_dirty=nV_dirty,
        global_counters=engine.global_counters, counter_idx=2,
        ws=engine.ws_vac,
    )

    engine.banks.swap()