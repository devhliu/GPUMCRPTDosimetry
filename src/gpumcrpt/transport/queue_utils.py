from __future__ import annotations

from typing import Dict, List
import torch


def compact_queue(q: Dict[str, torch.Tensor], mask: torch.Tensor, keys: List[str]) -> Dict[str, torch.Tensor]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.numel() == 0 or (not torch.any(mask)):
        # return empty views with correct shape/dtype
        out = {}
        for k in keys:
            t = q[k]
            out[k] = t[:0] if t.ndim == 1 else t[:0, ...]
        return out
    return {k: q[k][mask] for k in keys}


def append_queue(dst: Dict[str, torch.Tensor] | None, src: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    if dst is None:
        return {k: src[k] for k in keys}
    if src[keys[0]].numel() == 0:
        return dst
    return {k: torch.cat([dst[k], src[k]], dim=0) for k in keys}