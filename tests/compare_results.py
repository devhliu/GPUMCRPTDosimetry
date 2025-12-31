import nibabel as nib
import numpy as np
import json
from pathlib import Path

# Load the results
condensed_dose = nib.load('examples/results/lu-177_dose_photon_electron_condensed.nii.gz').get_fdata()
local_dose = nib.load('examples/results/lu-177_dose_photon_electron_local.nii.gz').get_fdata()

# Load execution times
results_dir = Path('examples/results')
condensed_timing_file = results_dir / 'lu-177_timing_photon_electron_condensed.json'
local_timing_file = results_dir / 'lu-177_timing_photon_electron_local.json'

condensed_time = None
local_time = None

if condensed_timing_file.exists():
    with open(condensed_timing_file, 'r') as f:
        condensed_timing = json.load(f)
        condensed_time = condensed_timing.get('execution_time_seconds')

if local_timing_file.exists():
    with open(local_timing_file, 'r') as f:
        local_timing = json.load(f)
        local_time = local_timing.get('execution_time_seconds')

# Calculate statistics
condensed_nonzero = condensed_dose[condensed_dose > 0]
local_nonzero = local_dose[local_dose > 0]

print("=" * 80)
print("PHOTON_ELECTRON_CONDENSED vs PHOTON_ELECTRON_LOCAL COMPARISON")
print("=" * 80)

print("\n--- Condensed Engine Statistics ---")
print(f"Total voxels: {condensed_dose.size}")
print(f"Non-zero voxels: {len(condensed_nonzero)}")
print(f"Max dose: {condensed_dose.max():.6e}")
print(f"Mean dose (non-zero): {condensed_nonzero.mean():.6e}")
print(f"Min dose (non-zero): {condensed_nonzero.min():.6e}")
if condensed_time is not None:
    print(f"Execution time: {condensed_time:.2f} seconds")
else:
    print("Execution time: Not available")

print("\n--- Local Engine Statistics ---")
print(f"Total voxels: {local_dose.size}")
print(f"Non-zero voxels: {len(local_nonzero)}")
print(f"Max dose: {local_dose.max():.6e}")
print(f"Mean dose (non-zero): {local_nonzero.mean():.6e}")
print(f"Min dose (non-zero): {local_nonzero.min():.6e}")
if local_time is not None:
    print(f"Execution time: {local_time:.2f} seconds")
else:
    print("Execution time: Not available")

print("\n--- Ratio Analysis ---")
ratio = condensed_dose / np.maximum(local_dose, 1e-12)
ratio_nonzero = ratio[local_dose > 0]
print(f"Max ratio: {ratio.max():.6e}")
print(f"Min ratio: {ratio[ratio > 0].min():.6e}")
print(f"Mean ratio (non-zero): {ratio_nonzero.mean():.6e}")

# Check for extreme ratios
extreme_ratio_threshold = 1000
extreme_ratio_mask = ratio > extreme_ratio_threshold
print(f"\nExtreme ratios > {extreme_ratio_threshold}: {np.sum(extreme_ratio_mask)} voxels")

# Check dose distribution similarity
print("\n--- Dose Distribution Comparison ---")
print(f"Condensed total dose: {condensed_dose.sum():.6e}")
print(f"Local total dose: {local_dose.sum():.6e}")
print(f"Total dose ratio: {condensed_dose.sum() / local_dose.sum():.6e}")

# Check correlation
correlation = np.corrcoef(condensed_dose.flatten(), local_dose.flatten())[0, 1]
print(f"Correlation coefficient: {correlation:.6f}")

print("\n--- Spatial Overlap Analysis ---")
condensed_mask = condensed_dose > 0
local_mask = local_dose > 0
intersection = np.sum(condensed_mask & local_mask)
union = np.sum(condensed_mask | local_mask)
iou = intersection / union if union > 0 else 0
print(f"Intersection over Union (IoU): {iou:.6f}")
print(f"Condensed only voxels: {np.sum(condensed_mask & ~local_mask)}")
print(f"Local only voxels: {np.sum(~condensed_mask & local_mask)}")

if condensed_time is not None and local_time is not None:
    print("\n--- Execution Time Comparison ---")
    print(f"Condensed engine: {condensed_time:.2f} seconds")
    print(f"Local engine: {local_time:.2f} seconds")
    if local_time > 0:
        speedup = local_time / condensed_time
        print(f"Speedup (Local/Condensed): {speedup:.2f}x")
    else:
        print("Speedup: Cannot calculate (local time is zero)")

print("\n" + "=" * 80)
