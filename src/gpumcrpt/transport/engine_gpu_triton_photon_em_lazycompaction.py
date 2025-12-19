"""
Phase 11: Lazy Sync compaction wiring (one CPU sync per step for dirty counts).

This integrates with your Phase 10 (Option A) banks:
- run physics kernels on n_dirty with status masking
- run lazy compaction that computes new_n on GPU and stores into global_counters
- ping-pong bank buffers to avoid in-place hazards

You must adapt this into your engine_gpu_triton.py.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

from gpumcrpt.transport.compaction_lazy_driver import (
    alloc_lazy_compaction_ws,
    lazy_compact_particlebank_pingpong,
    lazy_compact_vacancybank_pingpong,
)


@dataclass
class PingPongBanks:
    photons_a: object
    photons_b: object
    electrons_a: object
    electrons_b: object
    vacancies_a: object
    vacancies_b: object
    flip: bool = False

    def cur(self):
        if not self.flip:
            return self.photons_a, self.electrons_a, self.vacancies_a
        return self.photons_b, self.electrons_b, self.vacancies_b

    def nxt(self):
        if not self.flip:
            return self.photons_b, self.electrons_b, self.vacancies_b
        return self.photons_a, self.electrons_a, self.vacancies_a

    def swap(self):
        self.flip = not self.flip


def phase11_step_lazy(engine):
    """
    Example step structure:
      1) ONE SYNC: read dirty counts for grid sizing
      2) physics kernels launched with n_dirty (mask by status)
      3) compaction into ping-pong buffers; updates counters on GPU (no further sync)
      4) swap
    """
    # ---- ONE CPU sync for grid sizes ----
    nP_dirty = int(engine.global_counters[0].item())
    nE_dirty = int(engine.global_counters[1].item())
    nV_dirty = int(engine.global_counters[2].item())

    # ---- Oversubscribed physics launches (your kernels must check status==ALIVE) ----
    # engine.dispatch_photon_flight(nP_dirty)
    # engine.dispatch_photon_interactions(nP_dirty)
    # engine.dispatch_electron_step(nE_dirty)
    # engine.dispatch_relaxation(nV_dirty)
    # ... existing wavefront pipeline ...

    # ---- Lazy compaction (no sync for new_n) ----
    ph_src, el_src, vac_src = engine.banks.cur()
    ph_dst, el_dst, vac_dst = engine.banks.nxt()

    lazy_compact_particlebank_pingpong(
        src=ph_src, dst=ph_dst,
        n_dirty=nP_dirty,
        global_counters=engine.global_counters,
        counter_idx=0,
        ws=engine.compact_ws_ph,
    )
    lazy_compact_particlebank_pingpong(
        src=el_src, dst=el_dst,
        n_dirty=nE_dirty,
        global_counters=engine.global_counters,
        counter_idx=1,
        ws=engine.compact_ws_el,
    )
    lazy_compact_vacancybank_pingpong(
        src=vac_src, dst=vac_dst,
        n_dirty=nV_dirty,
        global_counters=engine.global_counters,
        counter_idx=2,
        ws=engine.compact_ws_vac,
    )

    engine.banks.swap()