# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Python Environment Setup

```bash
# Set Python path for imports
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # Development installation with testing
```

### Testing

```bash
# Run all tests
PYTHONPATH=src pytest tests/

# Run specific test suites
PYTHONPATH=src pytest tests/test_physics_validation.py -v     # Physics correctness
PYTHONPATH=src pytest tests/test_integration.py -v            # Integration & workflow
PYTHONPATH=src pytest tests/test_gpu_performance.py -v        # GPU performance (requires CUDA)

# Run with coverage
PYTHONPATH=src pytest --cov=src/gpumcrpt tests/
```

### Physics Table Generation

```bash
# Generate physics tables if not present
python scripts/generate_physics_tables.py --out physics_tables.h5
```

### Running Simulations

```bash
# Command line interface
python scripts/run_dosimetry_nifti.py \
    --ct path/to/ct.nii.gz \
    --activity path/to/activity.nii.gz \
    --sim_yaml configs/example_simulation.yaml \
    --out_dose path/to/dose_output.nii.gz \
    --out_unc path/to/uncertainty_output.nii.gz
```

## High-Level Architecture

### Core Components

**GPUMCRPTDosimetry** is a GPU-accelerated Monte Carlo particle transport system for radionuclide dosimetry with the following key architecture:

1. **Pipeline Layer** (`src/gpumcrpt/python_api/pipeline.py`): Main entry point that orchestrates the entire dosimetry workflow from NIfTI input to dose output

2. **Transport Engines** (`src/gpumcrpt/transport/`): Multiple transport implementations:
   - `engine_gpu_triton_localdepositonly.py`: Local energy deposition
   - `engine_gpu_triton_photon_electron_condensed.py`: Full photon + electron/positron transport
   - `engine_gpu_triton_photon_electron_local.py`: Photon-electron local transport
   - Uses Triton 3.5.1 for GPU kernel compilation

3. **Physics System** (`src/gpumcrpt/physics_tables/`):
   - Cross-section and stopping power tables stored in HDF5
   - Energy ranges: Photons (10 keV - 3 MeV), Electrons/Positrons (0.1 keV - 3 MeV)
   - Supports photoelectric, Compton, Rayleigh, pair production, bremsstrahlung

4. **Material System** (`src/gpumcrpt/materials/`):
   - 5-compartment model: Air, Lung, Fat, Muscle/Soft Tissue, Bone
   - HU-to-material conversion with density mapping
   - Material compositions stored in registry

5. **Radionuclide Database** (`src/gpumcrpt/decaydb/`):
   - ICRP-107 decay data in JSON format
   - Supports gamma, beta, alpha emissions with branching fractions
   - Time-integrated activity handling

6. **GPU Kernels** (`src/gpumcrpt/transport/triton/`):
   - Wavefront transport pattern: flight → classify → interact → step
   - CUDA graphs for performance optimization
   - Philox RNG for reproducible random numbers

### Key Design Patterns

- **Structure-of-Arrays (SoA)**: Particle data organized for coalesced GPU memory access
- **Wavefront Transport**: Sequential processing stages for GPU efficiency
- **Engine Selector**: Physics mode mapping from config to appropriate transport engine
- **Lazy Compaction**: Particle queue management without frequent memory reorganization

### Configuration System

Simulations configured via YAML with key sections:
- `simulation`: Basic parameters (nuclide, histories, batches, device)
- `monte_carlo`: Transport engine selection (`local_deposit`, `photon_electron_local`, `photon_electron_condensed`)
- `cutoffs`: Energy cutoffs for particles (keV)
- `materials`: HU-to-density/class mappings
- `physics_tables`: Path to physics data file
- `io`: NIfTI processing options

### Critical Dependencies

- **Triton == 3.5.1**: Exact version required for GPU kernel compatibility
- **PyTorch >= 2.2**: Tensor operations and CUDA integration
- **nibabel >= 5.2**: Medical imaging NIfTI I/O
- **h5py >= 3.10**: Physics tables storage
- **pytest >= 7.4**: Testing framework

## Physics Requirements and Implementation

### Energy Ranges and Particles
- **Photons**: 10 keV – 3 MeV (including 511 keV annihilation photons)
- **Electrons/Positrons**: 0.1 keV – 3 MeV (condensed history transport)
- **Alpha particles**: 4–9 MeV (local deposition due to sub-voxel range)

### Physics Domain Separation
Two distinct physics domains to avoid double counting:

**Domain A - Decay/Source-Term Physics**:
- Decay modes and branching ratios from ICRP-107
- Emission yields and energies (discrete lines and continuous spectra)
- Atomic vacancies from electron capture (EC) and internal conversion (IC)
- Radioactive daughter creation and chain progression

**Domain B - Transport/Interaction Physics**:
- Photon interactions: photoelectric, Compton, Rayleigh, pair production
- Charged particle transport with condensed history
- Secondary particle generation and cascades
- Energy deposition accumulation in voxel grid

### Critical Energy Conservation Rules
- **Neutrinos**: Non-depositing, never counted in tissue dose
- **β+ annihilation**: Either inject annihilation photons at decay OR simulate during transport (not both)
- **Internal conversion**: Apply ICC once - either γ line intensities + ICC electrons OR explicit conversion electrons with reduced γ intensities
- **Atomic relaxation**: Explicit X-rays/Auger electrons above cutoff, local deposition below cutoff
- **Cutoff termination**: Deposit remaining kinetic energy locally when particle is stopped

### Material System
- **5-compartment model**: Air, Lung, Fat, Muscle/Soft Tissue, Bone
- **HU to density conversion**: Scanner-specific calibration curves
- **Elemental compositions**: Weight fractions + I values for each material class
- **Cross-section scaling**: Density-based scaling from reference values

### Interaction Physics Details

**Photon Transport**:
- Photoelectric absorption → photoelectron + atomic relaxation
- Compton scattering → recoil electron + scattered photon
- Rayleigh scattering (optional below 50 keV for speed)
- Pair production (>1.022 MeV) → e⁻/e⁺ pair

**Charged Particle Transport** (Condensed History):
- Continuous slowing down via restricted stopping power
- Multiple scattering deflection per step
- Hard discrete events: bremsstrahlung photons, δ-ray electrons
- Positron annihilation at rest → 2×511 keV photons

### Cutoff Strategy
- Photon cutoff: 1-10 keV (typical 3 keV)
- Electron/positron cutoff: 1-10 keV (typical 10 keV)
- Below cutoff: Local deposition of remaining energy
- Above cutoff: Full transport with secondary production

### Atomic Relaxation
- Fluorescence X-rays and Auger electrons from shell vacancies
- Explicit transport above cutoff, local deposition below
- Energy conservation through complete cascade tracking

## GPU Performance Requirements and Architecture

### Performance Targets
- **Clinical volume**: 256×256×1024 voxels with 2 mm isotropic resolution (≈ 6.71×10⁷ voxels)
- **Throughput goal**: 1.0×10⁸ decay-equivalent histories in < 1 minute on RTX 4090 16 GB
- **GPU memory usage**: 4-8 GB for typical clinical simulation
- **Supported hardware**: RTX A4000/A6000 (Ampere), RTX 4090 (Ada), CUDA Compute Capability 7.0+

### GPU-Architecture Requirements

**Avoid GPU-Hostile Operations**:
- Per-thread unbounded while loops
- Heavy if/else control flow in hot kernels
- Frequent dynamic memory allocation
- Excessive global atomics to hotspot voxels
- Rejection sampling loops with variable iterations
- Frequent CPU↔GPU synchronization

**Use GPU-Friendly Algorithms**:
- **Wavefront/queue-based transport**: Sequential processing stages
- **Woodcock tracking**: No boundary stepping for photons
- **Table-driven inverse-CDF sampling**: No rejection loops
- **Stream compaction**: Prefix sum to remove dead particles
- **Optional sorting**: By energy bin/material class/Morton code for coherence

### Wavefront Transport Pattern
Sequential kernel stages to maintain uniform work distribution:
1. **Photon flight**: Woodcock tracking + accept/reject + classify
2. **Photon interactions**: Split by interaction type
3. **Charged particle step**: Condensed history + event classification
4. **Compaction**: Stream compaction to next queues
5. **Repeat** until queues empty or max iterations

### Memory and Data Organization
- **Structure-of-Arrays (SoA)**: Particle data for coalesced memory access
- **Packed queues**: Maintain active particle arrays without gaps
- **CUDA graphs**: Capture repeated kernel sequences for minimal launch overhead
- **Async execution**: Use torch streams to reduce synchronization overhead

### Performance Optimization Strategies
- **Autotuning**: Per-kernel block size optimization for target hardware
- **Energy binning**: Process particles in energy ranges for cache efficiency
- **Material batching**: Process by material class for memory coherence
- **Atomic reduction**: Lock-free energy deposition via atomic_add
- **Workspace management**: Reuse GPU memory buffers across kernel launches

### GPU Memory Management
- Clinical volumes require 4-8 GB GPU memory
- Use `n_batches` parameter for memory management in large simulations
- Workspace buffers reused across transport cycles
- Lazy compaction reduces memory fragmentation

### Physics Validation Requirements
- Energy conservation across all transport stages
- Proper cutoff handling with local deposition
- Statistical uncertainty tracking in all dose calculations
- Validation against known reference solutions
- Test suite covering physics correctness, GPU performance, and integration

### Performance Benchmarks
- **RTX 4090**: Target < 60 seconds for 10⁸ decay-equivalent histories
- **RTX A6000**: Expected < 90 seconds for same workload
- **Memory scaling**: Linear with volume size up to GPU capacity limits
- **Speedup targets**: >100x vs single-threaded CPU implementation