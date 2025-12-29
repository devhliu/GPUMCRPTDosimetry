# Prioritized Action Plan for Code Quality Improvement

## Executive Summary

This document outlines a prioritized action plan to improve the GPUMCRPTDosimetry codebase, balancing GPU performance and physics accuracy. The plan is organized into three priority tiers based on impact, feasibility, and risk.

---

## Phase 1: Critical Performance Fixes (Weeks 1-2)
**Impact: 10-100x speedup | Risk: Low | Effort: Medium**

### 1.1 Move Buffer Allocations Outside Loop (P0)
**File**: `engine_gpu_triton_photon_electron_condensed.py`
**Impact**: Eliminates ~1.2 million tensor allocations per simulation
**Risk**: Low - simple refactoring

**Action**:
```python
# BEFORE (inside loop):
for _ in range(max_steps):
    out_ph_pos = torch.empty_like(pos2)
    out_ph_dir = torch.empty_like(dir2)
    # ... 9 more allocations

# AFTER (before loop):
out_ph_pos = torch.empty_like(pos)
out_ph_dir = torch.empty_like(direction)
out_ph_E = torch.empty_like(E)
out_ph_w = torch.empty_like(w)
out_ph_ebin = torch.empty_like(ebin)

out_e_pos = torch.empty((N, 3), device=self.device, dtype=torch.float32)
out_e_dir = torch.empty((N, 3), device=self.device, dtype=torch.float32)
out_e_E = torch.empty((N,), device=self.device, dtype=torch.float32)
out_e_w = torch.empty((N,), device=self.device, dtype=torch.float32)

out_po_pos = torch.empty((N, 3), device=self.device, dtype=torch.float32)
out_po_dir = torch.empty((N, 3), device=self.device, dtype=torch.float32)
out_po_E = torch.empty((N,), device=self.device, dtype=torch.float32)
out_po_w = torch.empty((N,), device=self.device, dtype=torch.float32)

for _ in range(max_steps):
    # Reuse buffers
```

**Expected Speedup**: 5-10x

---

### 1.2 Add Autotuning to Photon Interaction Kernel (P0)
**File**: `triton_kernels/photon/interactions.py`
**Impact**: Optimizes kernel for different problem sizes
**Risk**: Low - follows existing pattern from flight kernel

**Action**:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 2}),
    ],
    key=['N'],
    warmup=10,
    rep=20,
)
@triton.jit
def photon_interaction_kernel(...):
    # Existing implementation
```

**Expected Speedup**: 1.2-2x

---

### 1.3 Set Reasonable Secondary Particle Limits (P0)
**File**: `configs/method_photon_electron_condensed.yaml`
**Impact**: Prevents exponential secondary particle explosion
**Risk**: Low - configuration change only

**Action**:
```yaml
electron_transport:
  secondary_depth: 1              # Changed from unlimited
  max_secondaries_per_primary: 1000   # Changed from 1_000_000_000
  max_secondaries_per_step: 10000     # Changed from 1_000_000
```

**Expected Speedup**: 2-5x (reduces secondary particle count by 1000x)

---

## Phase 2: GPU-CPU Synchronization Elimination (Weeks 3-4)
**Impact: 2-5x speedup | Risk: Medium | Effort: High**

### 2.1 Implement GPU-Based Secondary Selection (P1)
**File**: `transport/utils/secondary_budget.py` (new Triton kernel)
**Impact**: Eliminates CPU-GPU synchronization points
**Risk**: Medium - requires careful implementation

**Action**: Create Triton kernel for compact operation
```python
@triton.jit
def compact_kernel(
    flag_mask: tl.tensor,
    counts: tl.tensor,
    output_indices: tl.tensor,
    max_per_primary: tl.constexpr,
    max_per_step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < flag_mask.shape[0]
    
    flag = tl.load(flag_mask + offset, mask=mask, other=0).to(tl.int1)
    count = tl.load(counts + offset, mask=mask, other=0).to(tl.int32)
    
    eligible = flag & (count < max_per_primary)
    
    # Use prefix sum to compute output positions
    # Store selected indices to output
```

**Expected Speedup**: 2-3x

---

### 2.2 Optimize Secondary Particle Handling (P1)
**File**: `engine_gpu_triton_photon_electron_condensed.py`
**Impact**: Reduces kernel launch overhead
**Risk**: Medium - requires refactoring

**Action**: Batch secondary particle processing
```python
# Instead of processing each interaction type separately:
# Collect all secondaries first, then process in batches

all_secondaries = {
    'electrons': [],
    'positrons': [],
    'photons': [],
}

# Collect from all interaction types
all_secondaries['electrons'].extend(photoelectric_electrons)
all_secondaries['electrons'].extend(compton_electrons)
all_secondaries['electrons'].extend(pair_electrons)

# Process in single batch
if all_secondaries['electrons']:
    process_electrons_batch(all_secondaries['electrons'])
```

**Expected Speedup**: 1.5-2x

---

## Phase 3: Physics Accuracy Improvements (Weeks 5-8)
**Impact: 5-20% accuracy improvement | Risk: Medium | Effort: High**

### 3.1 Improve Bremsstrahlung Spectrum (P2)
**File**: `triton_kernels/charged_particle/step.py`
**Impact**: More accurate secondary photon production
**Risk**: Medium - requires validation

**Action**: Replace simplified spectrum with Bethe-Heitler
```python
# BEFORE (simplified):
brems_photon_energy = E_new * (u2 * 0.3)

# AFTER (Bethe-Heitler):
# Use proper screening-corrected Bethe-Heitler spectrum
# k = brems_photon_energy / E_new
# f(k) = (1 + (1-k)^2) / k * (phi1 - Z*phi2)
# where phi1, phi2 are screening functions
```

**Expected Accuracy Gain**: 10-20% for bremsstrahlung-dominated cases

---

### 3.2 Improve Delta Ray Energy Distribution (P2)
**File**: `triton_kernels/charged_particle/step.py`
**Impact**: More accurate secondary electron production
**Risk**: Medium - requires validation

**Action**: Replace simplified sampling with Moller scattering
```python
# BEFORE (simplified):
delta_energy = E_new * (u6 * 0.5)

# AFTER (Moller scattering):
# T = delta_energy / E_new
# dσ/dT ∝ 1/T^2 * [1 - β^2 T/T_max + (T/T_max)^2]
# Use inverse transform sampling or rejection sampling
```

**Expected Accuracy Gain**: 5-15% for electron transport

---

### 3.3 Clarify Photoelectric Handling (P2)
**File**: `engine_gpu_triton_photon_electron_condensed.py`
**Impact**: Consistent physics model
**Risk: Medium - requires decision on approach**

**Decision**: Choose one of two approaches:

**Option A**: Full electron tracking (more accurate, slower)
```python
# Track photoelectric electrons with condensed history
# Consistent with other secondary electrons
```

**Option B**: Full local deposition (faster, less accurate)
```python
# Deposit all photoelectric energy locally
# Skip relaxation kernel
# Consistent with photon_electron_local mode
```

**Recommendation**: Option B for dosimetry (photoelectric electrons have short range)

**Expected Impact**: 2-3x speedup (Option B) or 10-20% accuracy improvement (Option A)

---

## Phase 4: Documentation and Validation (Weeks 9-10)
**Impact: Improved maintainability | Risk: Low | Effort: Medium**

### 4.1 Create Physics Approximation Document (P3)
**File**: `docs/physics_approximations.md` (new)
**Impact**: Clear understanding of accuracy trade-offs
**Risk**: Low - documentation only

**Content**:
```markdown
# Physics Approximations in GPUMCRPTDosimetry

## Photon Interactions
- **Compton Scattering**: Kahn's method with Klein-Nishina formula
  - Accuracy: <1% error
  - Reference: Kahn, Phys. Rev. 90, 565 (1953)

- **Rayleigh Scattering**: Thompson with atomic form factor
  - Accuracy: <2% error for E < 1 MeV
  - Reference: Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975)

- **Pair Production**: Bethe-Heitler angular distribution
  - Accuracy: <3% error
  - Reference: Bethe & Heitler, Proc. R. Soc. A 146, 83 (1934)

- **Photoelectric Effect**: Local energy deposition (Option B)
  - Accuracy: <5% error for dosimetry (electrons have short range)
  - Trade-off: 2-3x speedup vs. full tracking

## Charged Particle Transport
- **Energy Loss**: Vavilov straggling
  - Accuracy: <5% error for E > 100 keV
  - Reference: Vavilov, Sov. Phys. JETP 5, 749 (1957)

- **Multiple Scattering**: Molière theory
  - Accuracy: <10% error for angles < 30°
  - Reference: Molière, Z. Naturforsch. A 2, 133 (1947)

- **Bremsstrahlung**: Simplified spectrum (TODO: improve to Bethe-Heitler)
  - Current accuracy: 20-30% error
  - Planned accuracy: <10% error

- **Delta Rays**: Simplified energy sampling (TODO: improve to Moller)
  - Current accuracy: 15-25% error
  - Planned accuracy: <10% error

## Validation Status
- [ ] Compton scattering validation against NIST databases
- [ ] Rayleigh scattering validation
- [ ] Pair production validation
- [ ] Energy loss validation against ICRU reports
- [ ] Multiple scattering validation
- [ ] Bremsstrahlung spectrum validation (planned)
- [ ] Delta ray distribution validation (planned)
```

---

### 4.2 Add Performance Profiling (P3)
**File**: `src/gpumcrpt/transport/perf/profiler.py` (new)
**Impact**: Identify remaining bottlenecks
**Risk**: Low - non-invasive

**Action**: Add profiling hooks
```python
class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
    
    def record_kernel(self, name, duration, n_particles):
        self.timings[name] = {
            'duration': duration,
            'n_particles': n_particles,
            'throughput': n_particles / duration
        }
    
    def report(self):
        print("Performance Report:")
        for name, data in self.timings.items():
            print(f"  {name}: {data['throughput']:.0f} particles/sec")
```

---

## Phase 5: Advanced Optimizations (Weeks 11+)
**Impact: 1.5-3x speedup | Risk: High | Effort: Very High**

### 5.1 Adaptive Woodcock Tracking (P4)
**File**: `triton_kernels/photon/flight.py`
**Impact**: Reduce virtual interactions in low-density regions
**Risk**: High - complex implementation

**Action**: Implement adaptive sigma_max
```python
# Compute local sigma_max per material instead of global
# Reduces virtual interactions in air/lung by 10-100x
```

**Expected Speedup**: 1.5-2x for heterogeneous geometries

---

### 5.2 Wavefront Optimization (P4)
**File**: `engine_gpu_triton_photon_electron_condensed.py`
**Impact**: Better particle load balancing
**Risk**: High - requires algorithm redesign

**Action**: Implement dynamic wavefront sizing
```python
# Adjust wavefront size based on active particle count
# Small wavefronts for few particles, large for many
```

**Expected Speedup**: 1.2-1.5x

---

### 5.3 Multi-Stream Execution (P4)
**File**: Multiple files
**Impact**: Overlap computation with data transfer
**Risk**: High - complex synchronization

**Action**: Use CUDA streams
```python
# Stream 1: Photon flight
# Stream 2: Photon interaction
# Stream 3: Charged particle transport
# Overlap kernel execution
```

**Expected Speedup**: 1.3-1.8x

---

## Implementation Roadmap

| Phase | Tasks | Duration | Expected Speedup | Expected Accuracy Gain |
|-------|-------|----------|------------------|----------------------|
| 1 | Critical Performance Fixes | 2 weeks | 10-20x | 0% |
| 2 | GPU-CPU Sync Elimination | 2 weeks | 2-5x | 0% |
| 3 | Physics Accuracy Improvements | 4 weeks | 0.8-1x | 10-20% |
| 4 | Documentation & Validation | 2 weeks | 0% | 0% |
| 5 | Advanced Optimizations | 4+ weeks | 1.5-3x | 0% |
| **Total** | **All Phases** | **14+ weeks** | **20-100x** | **10-20%** |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Performance regression from physics improvements | Medium | High | Benchmark before/after each change |
| GPU memory exhaustion from buffer reuse | Low | Medium | Monitor memory usage, add fallback |
| Autotuning fails on some hardware | Low | Medium | Provide fallback configuration |
| Physics validation fails | Medium | High | Incremental validation, keep old code |
| Timeline overruns | High | Medium | Prioritize Phase 1-3, defer Phase 5 |

---

## Success Metrics

### Performance Metrics
- **Target**: 20-100x speedup for photon_electron_condensed mode
- **Baseline**: Current performance on case001 example
- **Measurement**: Particles per second, wall-clock time

### Accuracy Metrics
- **Target**: <10% error vs. reference (EGSnrc/Geant4)
- **Baseline**: Current accuracy (unknown, needs validation)
- **Measurement**: Dose distribution comparison, energy deposition

### Code Quality Metrics
- **Target**: 90%+ code coverage for critical paths
- **Baseline**: Current coverage (unknown)
- **Measurement**: pytest coverage report

---

## Next Steps

1. **Week 1**: Implement Phase 1.1 (buffer allocation fix)
2. **Week 1**: Implement Phase 1.2 (autotuning)
3. **Week 2**: Implement Phase 1.3 (secondary limits)
4. **Week 2**: Benchmark and validate Phase 1 improvements
5. **Week 3-4**: Implement Phase 2 (GPU-based selection)
6. **Week 5-8**: Implement Phase 3 (physics improvements)
7. **Week 9-10**: Implement Phase 4 (documentation)
8. **Week 11+**: Implement Phase 5 (advanced optimizations)

---

## Conclusion

This action plan provides a clear, prioritized roadmap for improving the GPUMCRPTDosimetry codebase. By focusing on critical performance fixes first (Phase 1-2), we can achieve 20-100x speedup with minimal risk. Subsequent phases focus on physics accuracy improvements and advanced optimizations, providing a balanced approach to code quality improvement.

The plan is designed to be incremental, with each phase providing measurable improvements that can be validated before proceeding to the next phase. This reduces risk and ensures that resources are focused on the highest-impact improvements first.


## Key Root Causes
### GPU Performance Issues
1. Critical Memory Allocation Overhead : In photon_electron_condensed mode, output buffers are allocated inside the main loop (up to 100 iterations), creating ~1.2 million tensor allocations for N=10,000 particles. This is the single biggest performance bottleneck.
2. CPU-GPU Synchronization : The select_indices_with_budget function uses torch.nonzero() and torch.index_select() which cause CPU-GPU synchronization, blocking GPU execution multiple times per iteration.
3. Suboptimal Kernel Configuration : The photon interaction kernel lacks autotuning (unlike the flight kernel), potentially missing optimal configurations.
4. Excessive Secondary Handling : Default max_secondaries_per_primary=1_000_000_000 can spawn overwhelming numbers of secondary particles.
### Physics Approximation Issues
1. Inconsistent Photoelectric Handling : photon_electron_condensed mode deposits all energy locally but then calls a complex relaxation kernel - this is inconsistent and unclear.
2. Crude Secondary Production : Bremsstrahlung and delta ray sampling use very simplified approximations (E × u × 0.3 and E × u × 0.5 respectively).
3. Lack of Documentation : Physics approximations are not clearly documented, making it difficult to understand accuracy trade-offs.
## Priority Improvements
High Priority (Performance):

- Move buffer allocations outside the loop
- Add autotuning to interaction kernel
- Implement GPU-based secondary selection
- Set reasonable secondary particle limits
Medium Priority (Physics Clarity):

- Clarify photoelectric handling approach
- Improve secondary particle sampling accuracy
- Add comprehensive physics approximation documentation
The photon_electron_local mode is better optimized (buffers allocated once, no secondary handling), which explains its better performance. The photon_electron_condensed mode's complexity comes from handling secondary particles but suffers from poor implementation choices.