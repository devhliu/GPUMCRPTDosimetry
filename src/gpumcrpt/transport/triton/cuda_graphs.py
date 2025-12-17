from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class GraphBucket:
    """
    One CUDA graph bucket for a fixed maximum queue size.
    We replay the graph with variable *active* counts by masking/compaction outside,
    but allocations and kernel launch sequence stay constant.
    """
    max_n: int
    graph: torch.cuda.CUDAGraph
    static: Dict[str, torch.Tensor]


class CUDAGraphBucketManager:
    """
    Bucketed CUDA graph manager:
      - Choose bucket size >= N
      - Copy current queue tensors into static buffers (slice [0:N])
      - Replay captured graph
      - Read back outputs from static buffers

    Capture strategy:
      - Provide a `capture_fn(static)` callback that runs the fixed sequence of kernels
        using the static buffers only.
      - The callback MUST NOT allocate new tensors, call .item(), or change shapes.
    """

    def __init__(self, device: str = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs requested but CUDA is not available")
        self.device = device
        self._buckets: List[GraphBucket] = []

    @staticmethod
    def pick_bucket(n: int, bucket_sizes: List[int]) -> int:
        for b in bucket_sizes:
            if n <= b:
                return b
        return bucket_sizes[-1]

    def get_or_capture(
        self,
        max_n: int,
        make_static: Callable[[int], Dict[str, torch.Tensor]],
        capture_fn: Callable[[Dict[str, torch.Tensor]], None],
        warmup: int = 3,
    ) -> GraphBucket:
        for b in self._buckets:
            if b.max_n == max_n:
                return b

        # allocate static buffers
        static = make_static(max_n)

        # warmup to ensure kernels compiled (Triton JIT etc.)
        for _ in range(warmup):
            capture_fn(static)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        # capture
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            capture_fn(static)

        bucket = GraphBucket(max_n=max_n, graph=g, static=static)
        self._buckets.append(bucket)
        return bucket