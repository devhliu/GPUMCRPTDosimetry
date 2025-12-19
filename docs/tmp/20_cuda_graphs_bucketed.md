# CUDA Graphs: Bucketed Capture Strategy (Phase 5 performance)

You requested: **capturing separate graphs for several queue-size buckets**.

## Why buckets
CUDA graphs require **static shapes**. Our wavefront queues are dynamic.
Buckets approximate dynamics by capturing multiple graphs for max sizes:

Example bucket list:
- 4k, 16k, 64k, 256k particles

At runtime:
- choose smallest bucket `B >= N_active`
- copy queue into bucket buffers (pad with zeros)
- replay the pre-captured graph
- compact results outside replay

## What is captured
Capture **fixed micro-cycles** of hot kernels that do not allocate and do not branch on CPU:
- photon Woodcock flight (a few steps)
- electron condensed step (a few steps)

Between replays, we still do:
- compaction / split by interaction type
- secondary queue append
- cutoff handling

This improves performance by reducing per-iteration launch overhead while keeping correctness.

## Constraints (must follow)
- No `.item()` inside capture
- No allocations inside capture
- All tensors used by kernels must be preallocated in the bucket static buffers

## Configuration
Add to sim config:

```yaml
monte_carlo:
  triton:
    enable_cuda_graphs: true
    graph_bucket_sizes: [4096, 16384, 65536, 262144]
    micro_steps:
      photons: 2
      electrons: 2
```

## Roadmap
After graphs are stable:
- expand capture to include interaction kernels (PE/Compton/Rayleigh/Pair)
- capture brems/delta emission kernels
- potentially capture compaction via an in-graph scan implementation