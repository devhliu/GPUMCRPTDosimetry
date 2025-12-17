# Next phase: Phase 6 — Bucketed CUDA Graphs integration (Triton)

You confirmed the faster strategy: **separate CUDA graphs for queue-size “buckets”**.

## What we do in this phase
1. Add a bucketed CUDA-graphs execution path for the **hot micro-cycles**:
   - photon Woodcock flight (no interactions inside graph)
   - electron condensed-history step (no secondary emission kernels inside graph)
2. Keep the outer wavefront loop (interaction splitting, secondary queues, cutoffs, emission kernels) outside the graph.
3. Allow enabling/disabling via config:
   - `monte_carlo.triton.enable_cuda_graphs: true`
   - `monte_carlo.triton.graph_bucket_sizes: [4096, 16384, 65536, 262144]`
   - `monte_carlo.triton.micro_steps.photons/electrons: int`

## Why we do micro-cycles only
CUDA graphs require static shapes and no allocations; our queue sizes and compaction are dynamic.
Bucketed graphs reduce overhead while keeping correctness by:
- padding queues to a bucket size
- replaying fixed kernel sequences
- compacting outside

## Remaining work after this phase
- Expand graph coverage to include interaction kernels once the pipeline is stable.
- Move compaction into-graph (prefix sum) if needed for maximum speed.
- Integrate optional sorting coherently with graphs.
