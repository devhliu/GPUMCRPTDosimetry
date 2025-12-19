# GPUMCRPTDosimetry

GPU-accelerated Monte Carlo radionuclide particle transport dosimetry (research prototype).

## Overview

GPUMCRPTDosimetry is a high-performance Python package for internal dosimetry calculations using GPU-accelerated Monte Carlo particle transport. It is specifically designed for radionuclide pharmacy dosimetry (RPT) applications, providing fast and accurate voxel-level absorbed dose calculations for radiopharmaceuticals injected into the human body.

### Key Features

- **GPU Acceleration**: Leverages PyTorch and Triton 3.5.1 for high-performance GPU computing
- **Comprehensive Physics**: Supports photon, electron, and positron transport in the 10 keV to 3 MeV energy range
- **Medical Imaging Integration**: Direct support for CT (HU) and activity NIfTI files
- **Material Modeling**: Automatic conversion from CT Hounsfield Units to material compositions
- **Radionuclide Database**: Built-in support for ICRP-107 radionuclide decay data with fraction-based emissions
- **Wavefront Transport**: Efficient GPU-optimized particle transport algorithms with CUDA graphs
- **Physics Validation**: Comprehensive test suite covering physics correctness and GPU performance

## Quick Start

### Requirements

- NVIDIA GPU with CUDA Compute Capability 7.0 or higher (RTX series recommended)
- Python 3.10+
- CUDA 11.8+ (for GPU support)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GPUMCRPTDosimetry

# Install dependencies
pip install -r requirements.txt

# For development with testing
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.2
- **Triton == 3.5.1** (GPU kernel compilation)
- nibabel >= 5.2 (NIfTI I/O)
- numpy >= 2.2.6 (numerical computing)
- h5py >= 3.10 (physics tables storage)
- PyYAML >= 6.0 (configuration files)
- pytest >= 8.0 (testing)

### Basic Usage

```python
from gpumcrpt.pipeline import run_dosimetry
import yaml

# Load configuration
with open("configs/example_simulation.yaml", "r") as f:
    sim_config = yaml.safe_load(f)

# Run dosimetry calculation
run_dosimetry(
    activity_nifti_path="activity.nii.gz",
    ct_nifti_path="ct.nii.gz", 
    sim_config=sim_config,
    output_dose_path="dose.nii.gz",
    output_unc_path="uncertainty.nii.gz",
    device="cuda"
)
```

## Package Structure

```
src/gpumcrpt/
├── __init__.py              # Package initialization
├── pipeline.py              # Main dosimetry pipeline
├── materials/               # Material modeling (5-compartment model)
│   ├── hu_materials.py      # HU → material/density conversion
│   └── __init__.py
├── physics/                 # Physics data and tables
│   ├── tables.py            # Cross-section/stopping-power tables
│   └── relaxation_tables.py # Atomic relaxation data
├── transport/               # Particle transport engines
│   ├── engine.py            # Transport engine selector
	│   ├── engine_gpu_triton_localdepositonly.py          # Local deposition MVP
	│   ├── engine_gpu_triton_photon_em_condensedhistory.py # Full EM condensed history
│   ├── engine_gpu_triton_photon_only.py  # Photon-only mode
│   ├── triton/              # Triton GPU kernels
│   │   ├── photon_flight.py              # Woodcock flight
│   │   ├── photon_interactions.py        # Photon classification
│   │   ├── compton.py                    # Compton scattering
│   │   ├── electron_step.py              # Electron condensed history
│   │   ├── positron_step.py              # Positron condensed history
│   │   ├── pair.py                       # Pair production
│   │   ├── brems_delta.py                # Bremsstrahlung/delta rays
│   │   ├── edep_deposit.py               # Energy deposition
│   │   ├── rng.py                        # Philox RNG (Triton 3.5.1)
│   │   └── performance.py                # Performance optimization
│   └── queue_utils.py       # Particle queue management
├── source/                  # Particle source generation
│   └── sampling.py          # Decay sampling from ICRP107
├── decaydb/                 # Radionuclide decay database
│   ├── icrp107_json.py      # ICRP-107 JSON interface
│   └── icrp107_database/    # Decay data directory
├── io/                      # Input/output utilities
│   ├── nifti.py             # NIfTI file handling
│   └── __init__.py
└── dose/                    # Dose calculation
    ├── scoring.py           # Edep → dose conversion
    └── __init__.py
```

## Physics Capabilities

### Supported Particles
- **Photons** (γ/X-rays, including 511 keV annihilation photons)
- **Electrons** (including photoelectrons, Compton electrons, delta-rays)
- **Positrons** (with 2×511 keV annihilation at rest)
- **Alpha particles** (local energy deposition)

### Interaction Processes

#### Photon Transport
- **Photoelectric effect**: Creates photoelectrons with full photon energy (binding energy negligible in MVP)
- **Compton scattering**: Uses inverse-CDF sampling with cos(θ) convention
- **Rayleigh scattering**: Elastic scattering with minimal energy loss
- **Pair production**: Creates e⁺e⁻ pairs above 1.022 MeV

#### Charged Particle Transport
- **Condensed-history stepping**: 
  - Multiple scattering using Highland's formula
  - Energy loss via restricted stopping power (S_restricted)
  - Range-straggling approximation via CSDA range
- **Secondary production**:
  - Bremsstrahlung photons with energy-dependent cross section
  - Delta-ray electrons from ionization

#### Atomic Relaxation
- Fluorescence X-rays from photoelectric vacancies
- Auger electron emission (placeholder tables in MVP)

### Energy Ranges
- **Photons**: 10 keV – 3 MeV
- **Electrons/Positrons**: 0.1 keV – 3 MeV
- **Alpha particles**: 4–9 MeV (local deposition)

### Material Model
- **5-compartment model** with bone sub-types:
  - Air (HU < -850)
  - Lung (-850 to -910 HU)
  - Fat (-100 to -50 HU)
  - Muscle (+10 to +40 HU)
  - Soft Tissue (0 to +50 HU)
  - Bone (+150 to +3000 HU, averaged cortical/trabecular)

## Radionuclide Decay

Supports ICRP-107 radionuclide data in JSON format with:
- **Gamma emissions**: Discrete lines with branching fractions
- **Beta decays**: Continuous spectra (β⁻, β⁺)
- **X-rays**: Characteristic lines from atomic transitions
- **Auger electrons**: Monoenergetic from vacancy de-excitation
- **Alpha decay**: High-Z isotopes (local deposition)

Example: **Lu-177** (medical imaging radionuclide)
- 112 keV gamma (6.2%)
- 208 keV gamma (11%)
- 320 keV gamma (31%)
- Beta-minus with E_max ≈ 600 keV

## GPU Performance

### Optimization Strategies
- **Wavefront transport**: Sequential stages (flight → classify → interact → step)
- **CUDA graphs**: Capture repeated kernel sequences for minimal launch overhead
- **Structure-of-Arrays (SoA)**: Particle data layout for coalesced memory access
- **Triton autotuning**: Per-kernel block size optimization for target hardware
- **Atomic operations**: Lock-free energy deposition via atomic_add

### Expected Performance
- **Volume**: 256×256×1024 voxels (2 mm isotropic)
- **Throughput**: ~20-60 seconds for 10⁸ decay-equivalent histories
- **GPU Memory**: ~4-8 GB for typical clinical simulation
- **Hardware**: Tested on RTX A4000/A6000 (Ampere) and RTX 4090 (Ada)

## Testing & Validation

Comprehensive test suite included:

```bash
# Run all tests
PYTHONPATH=src pytest tests/

# Physics validation
pytest tests/test_physics_validation.py -v

# GPU performance
pytest tests/test_gpu_performance.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Test Coverage
- **Physics validation**: Discrete/continuous emission parsing, beta sampling, dose conversion, energy conservation
- **GPU performance**: Triton API compatibility, memory efficiency, kernel launch overhead, numerical stability
- **Integration**: Activity-to-primary conversion, energy conservation, material handling, boundary conditions

## Configuration

Simulations are configured via YAML files. Example: `configs/example_simulation.yaml`

```yaml
simulation:
  name: "Lu-177 dosimetry"
  
monte_carlo:
  n_histories: 1000000
  n_batches: 10
  triton:
    engine: "em_condensed"  # or "mvp", "photon_only"
    allow_placeholder_samplers: false

cutoffs:
  photon_keV: 3.0
  electron_keV: 20.0

physics_tables:
  h5_path: "physics_tables.h5"

materials:
  material_library: null  # Use default 5-compartment model
  hu_to_density: [[[-1000, 0.001], [0, 1.0], [3000, 1.8]]]
  hu_to_class: [[-1000, 0], [-850, 1], [-100, 2], [0, 3], [150, 5]]

radionuclide:
  name: "Lu-177"

io:
  resample_ct_to_activity: true
  output_uncertainty: "relative"
```

## Documentation

- **[User Guide](UserGuide.md)** - Detailed usage, configuration, and examples
- **[Physics Documentation](docs/)** - Theory, validation results, and optimization notes

## Development Status

### Completed Phases
- ✅ Phase 1-8: Photon transport with Woodcock flight
- ✅ Phase 9: Philox RNG integration (Triton 3.5.1 compatible)
- ✅ Phase 10: Electron/positron condensed-history transport
- ✅ Phase 11: Lazy compaction and performance optimization
- ✅ Milestone 4: Full physics validation and testing

### In Development
- [ ] Bremsstrahlung secondary production (phase 12)
- [ ] Advanced atomic relaxation tables (phase 13)
- [ ] Production validation for clinical use

## Contributing

Contributions welcome! Please adhere to:
- PEP 8 style guide
- Type hints for public APIs
- Unit tests for new features
- Comprehensive docstrings

## License

Proprietary - See [LICENSE](LICENSE) for details.

## Citation

Please cite this software and the physics validation in your publications.

## Support & Contact

For questions, bug reports, and feature requests, please open an issue or contact the development team.

---

**Last Updated**: December 2025  
**Triton Version**: 3.5.1  
**Python Version**: 3.10+  
**Status**: Research Prototype (Active Development)


### Energy Range
- Photons: 10 keV – 3 MeV
- Electrons/Positrons: 0.1 keV – 3 MeV
- Alpha particles: 4–9 MeV (local deposition)

## Performance

Designed for clinical-scale simulations:
- **Target volume**: 256×256×1024 voxels with 2 mm isotropic resolution
- **Expected performance**: ~20-60 seconds for 10^8 decay-equivalent histories on RTX 4090
- **GPU optimization**: Wavefront transport with Structure-of-Arrays (SoA) data layout

## Documentation

- [User Guide](UserGuide.md) - Detailed usage instructions and examples
- [Physics Design](docs/physics_rpt_design_principle.md) - Comprehensive physics implementation details
- [GPU Implementation](docs/physics_rpt_design4GPUMC.md) - GPU optimization strategies

## Development Status

This is a research prototype currently under active development. The package follows a milestone-based development approach:

- **Milestone 1**: Runnable MVP with local energy deposition
- **Milestone 2**: Photon-only transport implementation
- **Milestone 3**: Electron/positron condensed-history transport
- **Milestone 4**: Clinical realism with material compositions
- **Milestone 5**: Full physics validation and optimization

## Contributing

Contributions are welcome! Please see the development documentation for coding standards and contribution guidelines.

## License

Proprietary - See LICENSE file for details.

## Citation

If you use this software in your research, please cite the appropriate documentation and acknowledge the contributors.

## Support

For technical support and feature requests, please open an issue on the project repository.