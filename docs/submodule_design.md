# GPUMCRPTDosimetry Submodule Description

## Overview

GPUMCRPTDosimetry is a highly effective GPU-accelerated Monte Carlo radiation dosimetry simulation framework for medical physics applications. The system implements comprehensive radiation transport physics with support for photon, electron, and positron interactions in human tissue materials.

## Package Architecture

dosimetry modes:
- local_deposit: local deposition of photons and electrons & positrons
- photon_electron_local: local deposition of electrons & positrons, transport of photons
- photon_electron_condensed: transport of photons, electron & positron transport with condensed history

### Core Submodules and Their Roles

#### 1. `materials/` - Materials Management System
**Purpose**: Handle HU-to-materials conversion and materials property management

**Functionality**:
- Convert CT HU volumes to materials volumes
- Manage materials libraries (density, composition, properties)
- Support multiple materials tables (default_icru44, schneider_parameterization, simple_5_compartments)
- Device-aware operations (CPU/GPU)
- Support for custom materials libraries
- Extensibility for additional materials models
- Integration with external materials databases
- Dump example material database for exteral materials database definition

**Workflow Integration**:
- Input: CT HU volumes
- Output: Materials volumes with material IDs and densities
- Used by: Physics tables, Transport engines

#### 2. `physics_tables/` - Physics Data Management
**Purpose**: Pre-compute and manage physics interaction tables

**Functionality**:
- Pre-compute attenuation coefficients, stopping powers
- Generate energy-binned physics tables
- Support HDF5 format for data persistence
- Device-aware data generation (CPU/GPU)
- precompute tables for different dosimetry modes and materials libraries

**Dependencies**:
- `materials` module for materials properties

**Workflow Integration**:
- Input: Materials library
- Output: Precomputed physics tables
- Used by: Transport engines

#### 3. `transport/` - Radiation Transport Engine
**Purpose**: Core Monte Carlo radiation transport simulation

**Physics Interactions**:
- Photon: Compton scattering, photoelectric effect, pair production, Rayleigh scattering
- Electron: Condensed history transport, bremsstrahlung
- Positron: Annihilation, transport

**Dependencies**:
- `materials` for volume data
- `physics_tables` for interaction data
- `triton` for GPU kernels

**Workflow Integration**:
- Input: Primary particles, materials volumes, physics tables
- Output: Energy deposition maps
- Used by: Pipeline for dose calculation

#### 4. `source/` - Radiation Source Sampling
**Purpose**: Generate primary radiation sources from activity distributions

**Functionality**:
- Sample radioactive decays from activity maps
- Generate photon, electron, positron primaries
- Support ICRP107 decay database
- Variable-length particle queues

**Dependencies**:
- `decaydb` for decay data

**Workflow Integration**:
- Input: Activity distributions, decay data
- Output: Primary particle queues
- Used by: Transport engines

#### 5. `decaydb/` - Radioactive Decay Database
**Purpose**: Manage radioactive decay data and emissions

**Key Components**:
- `icrp107_database/`: ICRP107 decay data
- `icrp107_json.py`: JSON-based decay data parser

**Functionality**:
- Load and parse ICRP107 decay data
- Provide emission spectra and yields
- Support multiple nuclides

**Dependencies**:
- JSON data files
- Standard Python libraries

**Workflow Integration**:
- Input: Nuclide specifications
- Output: Decay emission data
- Used by: Source sampling

#### 6. `dose/` - Dose Calculation and Scoring
**Purpose**: Convert energy deposition to dose and uncertainty

**Functionality**:
- Convert energy deposition to absorbed dose
- Calculate statistical uncertainties
- Support multiple uncertainty modes

**Workflow Integration**:
- Input: Energy deposition maps, material densities
- Output: Dose maps, uncertainty maps
- Used by: Pipeline for final output

#### 7. `python_api/` - High-Level API
**Purpose**: Provide user-friendly interface for dosimetry calculations

**Functionality**:
- Coordinate all submodules
- Handle NIfTI file I/O
- Manage simulation configuration
- Provide simple API for users
- resample CT images to activity image using resample_from_to (from nibabel.processing import resample_from_to)

**Dependencies**:
- All other submodules
- `nibabel` for medical image I/O

**Workflow Integration**:
- Input: CT images, activity images, configuration
- Output: Dose images, uncertainty images

#### 8. `cli/` - Command Line Interface
**Purpose**: Provide command-line tools for batch processing


**Functionality**:
- Command-line interface for dosimetry calculations
- Batch processing support
- Configuration file handling

**Dependencies**:
- `python_api` for core functionality
- Standard CLI libraries

#### 9. `phantoms/` - Test Phantom Generation
**Purpose**: Generate test phantoms for validation


**Functionality**:
- Create simple validation phantoms
- Generate NEMA IEC body phantoms
- Support water slabs with bone cylinders

**Dependencies**:
- `nibabel` for image generation

#### 10. `configs/` - Configuration Files
**Purpose**: Store simulation configuration templates

**Functionality**:
- YAML configuration templates
- Simulation parameter presets
- Materials configuration files

## Physics Workflows

### Primary Workflow: End-to-End Dosimetry

1. **Input Processing** (`python_api/pipeline.py`)
   - Load CT and activity NIfTI images
   - Resample CT to activity image grid
   - Convert to torch tensors

2. **Materials Conversion** (`materials/`)
   - Convert HU volumes to materials volumes
   - Apply materials table (HU-to-density, HU-to-class)
   - Generate materials library

3. **Physics Tables** (`physics_tables/`)
   - Load or generate precomputed physics tables
   - Prepare attenuation coefficients, stopping powers
   - Generate energy-binned interaction data

4. **Source Sampling** (`source/`)
   - Sample radioactive decays from activity distribution
   - Generate primary particles (photons, electrons, positrons)
   - Calculate local energy deposition from alpha particles

5. **Radiation Transport** (`transport/`)
   - Select appropriate transport engine
   - Simulate particle interactions through materials
   - Track energy deposition in voxels
   - Handle secondary particle production

6. **Dose Calculation** (`dose/`)
   - Convert energy deposition to absorbed dose
   - Calculate statistical uncertainties
   - Apply voxel volume normalization

7. **Output Generation**
   - Save dose and uncertainty maps as NIfTI files
   - Generate reports and statistics

### Alternative Workflows

- **Photon-Electron Local Transport**: Simplified workflow for photon_electron_local simulations
- **Local Deposition Only**: MVP workflow for validation and testing
- **CPU Oracle**: Reference implementation for validation

## Data Flow and Dependencies

### Primary Data Flow

```
CT Images → Materials Conversion → Physics Tables
Activity Images → Source Sampling → Transport Engine → Dose Calculation → Output
```

### Key Data Structures

- **MaterialsVolume**: Material IDs and densities per voxel
- **PhysicsTables**: Precomputed interaction data
- **ParticleQueues**: Primary particle data (photons, electrons, positrons)
- **EnergyDeposition**: 3D energy deposition maps

### Module Dependencies

```
python_api/
├── materials/
├── physics_tables/
├── source/
│   └── decaydb/
├── transport/
└── dose/
```

## GPU Acceleration Strategy

### Triton Kernel Architecture

The system uses Triton for GPU acceleration with specialized kernels:

- **Photon Transport**: Compton, photoelectric, pair production, Rayleigh
- **Electron Transport**: Condensed history, bremsstrahlung
- **Positron Transport**: Annihilation, transport
- **Utility Kernels**: RNG, compaction, tallying

### Performance Optimizations

- **Bank-based Particle Management**: Efficient particle queue handling
- **Lazy Compaction**: Optimized memory management
- **Tile-based Tallying**: Efficient energy deposition scoring
- **CUDA Graphs**: Reduced kernel launch overhead

## Configuration and Customization

### Simulation Configuration

- **Materials**: Selection of materials tables and conversion parameters
- **Physics**: Cutoff energies, interaction models
- **Monte Carlo**: Number of histories, batches, random seeds
- **Transport**: Engine selection, optimization parameters

### Extensibility Points

- **Materials Tables**: Add custom HU-to-materials conversions
- **Decay Database**: Support additional nuclides
- **Transport Engines**: Implement new physics models
- **Scoring Methods**: Custom dose calculation algorithms

## Validation and Testing

### Test Phantoms

- **Water Slab with Bone Cylinder**: Simple validation phantom
- **NEMA IEC Body Phantom**: Standard medical physics phantom

### Reference Implementations

- **CPU Oracle**: Reference transport implementation
- **Local Deposition Only**: Simplified validation mode

## Current Status and Future Directions

### Currently Implemented

- Basic materials system with ICRU44 tables
- Photon-electron local and local deposition transport
- ICRP107 decay database support
- GPU acceleration with Triton
- NIfTI file I/O

### Planned Enhancements

- Accurate physics models for pre-computed physics tables
- Full photon-electron transport
- Advanced materials models
- Additional decay databases
- Improved optimization and performance
- Extended validation and testing



  Recommendation: COMBINED electron/positron kernels

  Rationale for Combination:

  1. GPU Performance Benefits:
    - Reduced kernel launches: Single kernel handles both particle types
    - Better memory coalescing: Particles processed together improve cache locality
    - Simplified wavefront: No need for separate electron/positron queues
    - Lower register pressure: Shared code paths reduce GPU memory requirements
  2. Physics Similarities (>95% identical):
    - Multiple scattering (Molière theory): Identical
    - Energy loss/straggling (Bethe-Bloch): Nearly identical, only differs by sign at very low energies
    - Bremsstrahlung: Same cross-sections
    - Delta ray production: Same mechanisms
    - Cutoffs and stepping: Identical
  3. Key Differences (easily handled):
    - Annihilation: Only positrons, handled at end-of-life (E < cutoff)
    - Minor cross-section differences: Handled by particle_type flag
    - Sign flipping in magnetic fields: Not relevant in medical dosimetry