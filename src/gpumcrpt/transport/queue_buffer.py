from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable
import torch


def _ceil_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass
class QueueBuffer:
    bufs: Dict[str, torch.Tensor]
    size: int
    cap: int

    @staticmethod
    def allocate_photon(device: torch.device, cap: int) -> "QueueBuffer":
        cap = _ceil_pow2(int(cap))
        bufs = {
            "pos_cm": torch.empty((cap, 3), device=device, dtype=torch.float32),
            "dir": torch.empty((cap, 3), device=device, dtype=torch.float32),
            "E_MeV": torch.empty((cap,), device=device, dtype=torch.float32),
            "w": torch.empty((cap,), device=device, dtype=torch.float32),
            "ebin": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_key0": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_key1": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr0": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr1": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr2": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr3": torch.empty((cap,), device=device, dtype=torch.int32),
        }
        return QueueBuffer(bufs=bufs, size=0, cap=cap)

    @staticmethod
    def allocate_electron(device: torch.device, cap: int) -> "QueueBuffer":
        cap = _ceil_pow2(int(cap))
        bufs = {
            "pos_cm": torch.empty((cap, 3), device=device, dtype=torch.float32),
            "dir": torch.empty((cap, 3), device=device, dtype=torch.float32),
            "E_MeV": torch.empty((cap,), device=device, dtype=torch.float32),
            "w": torch.empty((cap,), device=device, dtype=torch.float32),
            "ebin": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_key0": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_key1": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr0": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr1": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr2": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr3": torch.empty((cap,), device=device, dtype=torch.int32),
        }
        return QueueBuffer(bufs=bufs, size=0, cap=cap)

    @staticmethod
    def allocate_vacancy(device: torch.device, cap: int) -> "QueueBuffer":
        cap = _ceil_pow2(int(cap))
        bufs = {
            "pos_cm": torch.empty((cap, 3), device=device, dtype=torch.float32),
            "mat": torch.empty((cap,), device=device, dtype=torch.int32),
            "shell": torch.empty((cap,), device=device, dtype=torch.int8),
            "w": torch.empty((cap,), device=device, dtype=torch.float32),
            "rng_key0": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_key1": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr0": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr1": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr2": torch.empty((cap,), device=device, dtype=torch.int32),
            "rng_ctr3": torch.empty((cap,), device=device, dtype=torch.int32),
        }
        return QueueBuffer(bufs=bufs, size=0, cap=cap)

    def clear(self) -> None:
        self.size = 0

    def view(self) -> Dict[str, torch.Tensor]:
        return {k: v[: self.size] for k, v in self.bufs.items()}

    def append(self, src: Dict[str, torch.Tensor], keys: Iterable[str] | None = None) -> None:
        if keys is None:
            keys = src.keys()
        n = int(next(iter(src.values())).shape[0])
        if n == 0:
            return
        if self.size + n > self.cap:
            raise RuntimeError(f"QueueBuffer overflow: size={self.size}, n={n}, cap={self.cap}")
        s = self.size
        for k in keys:
            self.bufs[k][s:s+n].copy_(src[k])
        self.size += n