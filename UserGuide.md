# GPUMCRPTDosimetry User Guide

This guide provides comprehensive instructions for using the GPUMCRPTDosimetry package for radionuclide dosimetry calculations.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage](#basic-usage)
3. [Input Data Preparation](#input-data-preparation)
4. [Configuration Files](#configuration-files)
5. [Running Simulations](#running-simulations)
6. [Output Analysis](#output-analysis)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Prerequisites

- NVIDIA GPU with CUDA Compute Capability 7.0+ (RTX series recommended) or CPU-only mode
- Python 3.10 or later
- PyTorch 2.2 or later
- **Triton 3.5.1** (exact version required for GPU kernel compatibility)

### Installation Steps

```bash
# Clone the repository
git clone <repository-url>
cd GPUMCRPTDosimetry

# Install core dependencies with correct versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install triton==3.5.1  # CRITICAL: Exact version

# Install package dependencies
pip install nibabel numpy h5py pyyaml

# For development with testing
pip install -e ".[dev]"
```

### Verification

Test the installation:

```python
import torch
import triton

print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# For GPU mode
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
```

### Running Tests

After installation, verify everything works:

```bash
# Set Python path for imports
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_physics_validation.py -v     # Physics correctness
pytest tests/test_integration.py -v            # Integration & workflow
pytest tests/test_gpu_performance.py -v        # GPU performance (requires CUDA)
```

## Testing and Validation

### Test Suite Overview

The package includes comprehensive tests covering physics, GPU performance, and integration:

#### Physics Validation Tests

Tests core physics implementations and correctness:

```bash
pytest tests/test_physics_validation.py -v
```

Covers:
- **Discrete emission parsing**: ICRP-107 JSON data reading
- **Beta spectrum sampling**: Continuous spectrum sampling accuracy
- **Dose conversion**: Energy deposition to absorbed dose (Gy)
- **Material composition**: HU-to-material mapping validity
- **RNG determinism**: Reproducibility with fixed seeds
- **Energy conservation**: Total energy accounting across decay modes

#### Integration Tests

Validates complete workflows and physics consistency:

```bash
pytest tests/test_integration.py -v
```

Covers:
- **Activity-to-primaries**: Radioactive decay particle generation
- **Energy conservation**: Across all transport stages
- **Weighted sampling**: Voxel probability distributions
- **Compton kinematics**: Photon scattering physics
- **Stopping power**: Charged particle energy loss
- **Material handling**: Density and composition validity
- **Numerical stability**: Edge cases and extreme values

#### GPU Performance Tests

Validates GPU kernel performance and numerical stability:

```bash
pytest tests/test_gpu_performance.py -v -s  # -s for live output
```

Covers (requires CUDA):
- **Triton 3.5.1 API**: Version compatibility and decorators
- **Memory efficiency**: Coalesced access patterns
- **Kernel performance**: Launch overhead and bandwidth utilization
- **Numerical stability**: Large accumulations and small-number division
- **Boundary conditions**: Voxel containment checking

### Running Test Subsets

```bash
# Physics tests only
pytest tests/test_physics_validation.py::TestDiscreteEmissionParsing -v

# GPU tests only (requires CUDA)
pytest tests/test_gpu_performance.py -v

# With coverage report
pytest --cov=src/gpumcrpt tests/
```

### Test Troubleshooting

**Missing test dependencies?**
```bash
pip install pytest pytest-cov
```

**CUDA tests skip?**
GPU tests automatically skip if CUDA is unavailable. To force run:
```bash
pytest tests/test_gpu_performance.py -v --tb=short
```

**Performance test timeout?**
Adjust PyTest timeout for large simulations:
```bash
pytest tests/test_integration.py --timeout=600 -v
```

### Validation Checklist

Before running clinical simulations:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Triton version correct: `python -c "import triton; print(triton.__version__)"`
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Physics tables exist: Check `physics_tables.h5` or generate via `scripts/build_toy_physics_h5.py`
- [ ] Radionuclide data available: Check `src/gpumcrpt/decaydb/icrp107_database/`
- [ ] Example config loads: `yaml.load("configs/example_simulation.yaml")`

### Custom Physics Validation

To validate against known results:

```python
from gpumcrpt.pipeline import run_dosimetry
import numpy as np

# Run small test case
result = run_dosimetry(...)

# Compare to reference (if available)
reference = np.load("reference_dose.npy")
difference = np.abs(result - reference) / (np.abs(reference) + 1e-6)

# Statistical uncertainty should explain differences
print(f"Mean relative difference: {difference.mean():.4f}")
print(f"Max relative difference: {difference.max():.4f}")
```

## Basic Usage

### Command Line Interface

The primary interface is through the `run_dosimetry_nifti.py` script:

```bash
python scripts/run_dosimetry_nifti.py \
    --ct path/to/ct.nii.gz \
    --activity path/to/activity.nii.gz \
    --sim_yaml configs/example_simulation.yaml \
    --out_dose path/to/dose_output.nii.gz \
    --out_unc path/to/uncertainty_output.nii.gz
```

### Python API

For programmatic use:

```python
from gpumcrpt.pipeline import run_dosimetry
import yaml

# Load simulation configuration
with open("configs/example_simulation.yaml", "r") as f:
    sim_config = yaml.safe_load(f)

# Run dosimetry calculation
run_dosimetry(
    activity_nifti_path="activity.nii.gz",
    ct_nifti_path="ct.nii.gz",
    sim_config=sim_config,
    output_dose_path="dose.nii.gz",
    output_unc_path="uncertainty.nii.gz",
    device="cuda"  # or "cpu"
)
```

## Input Data Preparation

### CT Data Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Units**: Hounsfield Units (HU)
- **Orientation**: Standard medical imaging orientation (RAS)
- **Voxel size**: Typically 1-4 mm isotropic

### Activity Data Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Units**: Becquerels-second per voxel (Bq*s/voxel, Time-Integrated Activity)
- **Spatial registration**: Must align with CT data
- **Resolution**: Should match CT or be resampled

### Data Alignment

The package can automatically resample CT to match activity data:

```yaml
# In simulation configuration
io:
  resample_ct_to_activity: true  # Default: true
```

## Configuration Files

### Simulation Configuration Structure

Example configuration (`configs/example_simulation.yaml`):

```yaml
# Basic simulation parameters
simulation:
  nuclide: "F18"
  n_histories: 1000000
  n_batches: 10
  device: "cuda"

# Physics cutoffs (keV)
cutoffs:
  photon_keV: 10.0
  electron_keV: 10.0

# Material definitions
materials:
  material_library: "default"  # or path to custom library
  hu_ranges:
    air: [-1000, -200]
    lung: [-200, 0]
    soft_tissue: [0, 100]
    bone: [100, 3000]

# I/O settings
io:
  resample_ct_to_activity: true
  output_format: "nifti"

# Physics model selection
physics:
  transport_model: "em_condensed"  # Options: mvp, photon_only, em_condensed
  include_relaxation: false
  include_bremsstrahlung: true
```

### Available Radionuclides

Supported radionuclides from ICRP-107 database:
- F18, Tc99m, I131, Lu177, Y90, Ga68, In111, etc.

### Material Library Options

- `"default"`: Built-in material definitions
- Custom library: Path to YAML file with material compositions

## Running Simulations

### Quick Start Example

1. **Prepare input data**: Ensure CT and activity NIfTI files are properly formatted
2. **Create configuration**: Copy and modify `configs/example_simulation.yaml`
3. **Run simulation**: Use the command line interface or Python API
4. **Monitor progress**: The package provides progress indicators

### Batch Processing

For multiple simulations:

```python
import os
from gpumcrpt.pipeline import run_dosimetry
import yaml

# Load base configuration
with open("configs/base_simulation.yaml", "r") as f:
    base_config = yaml.safe_load(f)

# List of patient studies
patients = [
    {"ct": "patient1_ct.nii.gz", "activity": "patient1_activity.nii.gz"},
    {"ct": "patient2_ct.nii.gz", "activity": "patient2_activity.nii.gz"},
]

for i, patient in enumerate(patients):
    print(f"Processing patient {i+1}")
    
    # Update configuration for this patient
    config = base_config.copy()
    config["patient_id"] = f"patient_{i+1}"
    
    run_dosimetry(
        activity_nifti_path=patient["activity"],
        ct_nifti_path=patient["ct"],
        sim_config=config,
        output_dose_path=f"patient_{i+1}_dose.nii.gz",
        output_unc_path=f"patient_{i+1}_uncertainty.nii.gz"
    )
```

### Performance Optimization

#### GPU Memory Management

For large volumes, adjust batch size:

```yaml
simulation:
  n_batches: 20  # More batches = less GPU memory per batch
```

#### Physics Model Selection

Choose appropriate transport model based on accuracy requirements:

- `photon_only`: Fastest, photons only
- `em_condensed`: Balanced, electrons/positrons with condensed history
- `full_physics`: Most accurate, complete physics (when available)

## Output Analysis

### Output Files

Each simulation produces:

1. **Dose map** (`*_dose.nii.gz`): Absorbed dose in Gray (Gy) per voxel
2. **Uncertainty map** (`*_unc.nii.gz`): Relative uncertainty (σ/mean) per voxel

### Loading and Visualizing Results

```python
import nibabel as nib
import matplotlib.pyplot as plt

# Load dose results
dose_img = nib.load("dose_output.nii.gz")
dose_data = dose_img.get_fdata()

# Load uncertainty results
unc_img = nib.load("uncertainty_output.nii.gz")
unc_data = unc_img.get_fdata()

# Visualize middle slice
slice_idx = dose_data.shape[2] // 2
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(dose_data[:, :, slice_idx], cmap='hot')
plt.colorbar(label='Dose (Gy)')
plt.title('Absorbed Dose')

plt.subplot(1, 2, 2)
plt.imshow(unc_data[:, :, slice_idx], cmap='viridis')
plt.colorbar(label='Relative Uncertainty')
plt.title('Statistical Uncertainty')

plt.tight_layout()
plt.show()
```

### Dose Statistics

```python
import numpy as np

# Basic statistics
dose_values = dose_data[dose_data > 0]
print(f"Mean dose: {np.mean(dose_values):.4f} Gy")
print(f"Max dose: {np.max(dose_values):.4f} Gy")
print(f"Dose volume: {len(dose_values)} voxels")

# ROI analysis (example: high-dose region)
high_dose_mask = dose_data > np.percentile(dose_values, 90)
high_dose_mean = np.mean(dose_data[high_dose_mask])
print(f"High-dose region mean: {high_dose_mean:.4f} Gy")
```

## Advanced Features

### Custom Material Definitions

Create custom material library:

```yaml
# custom_materials.yaml
materials:
  water:
    density: 1.0
    composition:
      H: 0.111898
      O: 0.888102
  bone:
    density: 1.85
    composition:
      H: 0.064
      C: 0.278
      N: 0.027
      O: 0.410
      P: 0.070
      Ca: 0.151
```

### Physics Table Generation

Generate custom physics tables:

```bash
python scripts/build_toy_physics_h5.py --out custom_physics.h5
```

### Transport Engine Customization

Access low-level transport engines:

```python
from gpumcrpt.transport.engine_gpu_triton_photon_em_condensedhistory import TritonPhotonEMCondensedHistoryEngine

# Custom transport configuration
transport_engine = TritonPhotonEMCondensedHistoryEngine(
    mats=materials_volume,
    tables=physics_tables,
    sim_config=custom_config,
    voxel_size_cm=(0.2, 0.2, 0.2)  # 2mm voxels
)
```

## Troubleshooting

### Common Issues

#### GPU Memory Errors

**Problem**: CUDA out of memory error
**Solution**: Reduce batch size or use smaller voxel grids

```yaml
simulation:
  n_batches: 50  # Increase number of batches
```

#### File Format Issues

**Problem**: NIfTI file loading errors
**Solution**: Verify file integrity and orientation

```python
import nibabel as nib
img = nib.load("file.nii.gz")
print(f"Shape: {img.shape}")
print(f"Affine: {img.affine}")
```

#### Performance Issues

**Problem**: Slow simulation
**Solution**: Use appropriate transport model and GPU

- Use `photon_only` for quick estimates
- Ensure CUDA is properly configured
- Check GPU utilization during simulation

### Debug Mode

Enable verbose output for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validation Tests

Run validation tests to verify installation:

```bash
python -m pytest tests/ -v
```

## Best Practices

### Data Quality

- Verify CT and activity data alignment
- Check for negative activity values
- Ensure proper voxel size calibration

### Simulation Parameters

- Start with smaller `n_histories` for testing
- Use appropriate cutoffs for your energy range
- Validate results against known benchmarks

### Result Interpretation

- Consider statistical uncertainty in dose maps
- Validate against clinical expectations
- Use appropriate visualization scales

## Support and Resources

- **Documentation**: See `docs/` directory for detailed physics and implementation guides
- **Examples**: Check `scripts/` for usage examples
- **Issues**: Report problems via the project repository
- **Community**: Join discussions for feature requests and improvements

---

This user guide covers the essential aspects of using GPUMCRPTDosimetry. For advanced usage and development, refer to the detailed documentation in the `docs/` directory.