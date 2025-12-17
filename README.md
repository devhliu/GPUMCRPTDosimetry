# GPUMCRPTDosimetry

GPU-accelerated Monte Carlo radionuclide particle transport dosimetry (research prototype).

## Overview

GPUMCRPTDosimetry is a high-performance Python package for internal dosimetry calculations using GPU-accelerated Monte Carlo particle transport. It is specifically designed for radionuclide pharmacy dosimetry (RPT) applications, providing fast and accurate voxel-level absorbed dose calculations for radiopharmaceuticals injected into the human body.

### Key Features

- **GPU Acceleration**: Leverages PyTorch and Triton for high-performance GPU computing
- **Comprehensive Physics**: Supports photon, electron, and positron transport in the 10 keV to 3 MeV energy range
- **Medical Imaging Integration**: Direct support for CT (HU) and activity NIfTI files
- **Material Modeling**: Automatic conversion from CT Hounsfield Units to material compositions
- **Radionuclide Database**: Built-in support for ICRP-107 radionuclide decay data
- **Wavefront Transport**: Efficient GPU-optimized particle transport algorithms

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GPUMCRPTDosimetry

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.2
- Triton >= 2.1
- nibabel >= 5.2
- numpy >= 1.26
- h5py >= 3.10
- PyYAML >= 6.0

### Basic Usage

```python
from gpumcrpt.pipeline import run_dosimetry

# Run dosimetry calculation
run_dosimetry(
    activity_nifti_path="activity.nii.gz",
    ct_nifti_path="ct.nii.gz", 
    sim_config=sim_config_dict,
    output_dose_path="dose.nii.gz",
    output_unc_path="uncertainty.nii.gz"
)
```

## Package Structure

```
src/gpumcrpt/
├── __init__.py              # Package initialization
├── pipeline.py              # Main dosimetry pipeline
├── materials/               # Material modeling and HU conversion
│   ├── hu_materials.py      # HU to material conversion
│   ├── phantoms.py          # Phantom definitions
│   └── __init__.py
├── physics/                 # Physics models and cross-sections
│   ├── tables.py            # Physics tables management
│   └── relaxation_tables.py # Atomic relaxation data
├── transport/               # Particle transport engines
│   ├── engine.py            # Main transport engine interface
│   ├── engine_gpu_triton.py # GPU transport implementations
│   └── triton/              # Triton GPU kernels
├── source/                  # Particle source generation
│   └── sampling.py          # Radionuclide sampling
├── decaydb/                 # Radionuclide database
│   └── icrp107_json.py      # ICRP-107 data interface
├── io/                      # Input/output utilities
│   ├── nifti.py             # NIfTI file handling
│   └── __init__.py
└── dose/                    # Dose calculation
    ├── scoring.py           # Energy deposition to dose conversion
    └── __init__.py
```

## Physics Capabilities

### Supported Particles
- **Photons** (γ/X-rays, including 511 keV annihilation photons)
- **Electrons** (including photoelectrons, Compton electrons)
- **Positrons** (with annihilation photon generation)
- **Alpha particles** (local energy deposition)

### Interaction Processes
- **Photon interactions**: Photoelectric effect, Compton scattering, Rayleigh scattering, Pair production
- **Charged particle transport**: Condensed-history electron/positron stepping with bremsstrahlung and delta-ray production
- **Atomic relaxation**: Fluorescence X-rays and Auger electrons from photoelectric vacancies

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