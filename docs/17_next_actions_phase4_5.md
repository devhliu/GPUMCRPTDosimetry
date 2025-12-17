## Phase 5 (performance + full physics)
- Replace torch compaction with prefix-sum compaction
- Add Rayleigh and Pair kernels with tabulated samplers
- Add brems/delta spectral samplers from `.h5`
- Add optional sorting/coherence improvements
- Add CUDA graphs optional via torch.compile / torch.cuda.graphs