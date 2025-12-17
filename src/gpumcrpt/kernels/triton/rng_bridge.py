from __future__ import annotations

import torch


def is_philox_soa(q: dict) -> bool:
    """
    Heuristic: queue carries Philox key/ctr SoA fields.
    """
    return (
        "rng_key0" in q and "rng_key1" in q and
        "rng_ctr0" in q and "rng_ctr1" in q and "rng_ctr2" in q and "rng_ctr3" in q
    )


def upgrade_rng_i32_to_philox_soa(q: dict, *, seed: int = 1234) -> dict:
    """
    Transitional helper if your current queues store rng as int32.
    Not deterministic w.r.t. previous runs; use only while refactoring.

    Produces Philox SoA fields:
      rng_key0, rng_key1, rng_ctr0..3

    Strategy:
      - key0/key1 derived from seed
      - ctr0 from old rng (or arange), ctr1..3 = 0
    """
    device = next(iter(q.values())).device
    N = int(next(iter(q.values())).shape[0])

    if is_philox_soa(q):
        return q

    rng_i32 = q.get("rng", None)
    if rng_i32 is None:
        rng_i32 = torch.arange(N, device=device, dtype=torch.int32)

    key0 = torch.full((N,), seed & 0xFFFFFFFF, device=device, dtype=torch.int32)
    key1 = torch.full((N,), (seed * 2654435761) & 0xFFFFFFFF, device=device, dtype=torch.int32)

    q = dict(q)
    q["rng_key0"] = key0
    q["rng_key1"] = key1
    q["rng_ctr0"] = rng_i32.to(torch.int32)
    q["rng_ctr1"] = torch.zeros((N,), device=device, dtype=torch.int32)
    q["rng_ctr2"] = torch.zeros((N,), device=device, dtype=torch.int32)
    q["rng_ctr3"] = torch.zeros((N,), device=device, dtype=torch.int32)

    # keep old rng for compatibility until fully removed
    return q