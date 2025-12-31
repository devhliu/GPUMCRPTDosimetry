import nibabel as nib
import numpy as np

# Load the results
condensed_dose = nib.load('examples/results/lu-177_dose_photon_electron_condensed.nii.gz').get_fdata()
local_dose = nib.load('examples/results/lu-177_dose_photon_electron_local.nii.gz').get_fdata()
activity = nib.load('examples/case001/input/TIA.nii.gz').get_fdata()

print("=" * 80)
print("SPATIAL DISTRIBUTION ANALYSIS")
print("=" * 80)

# Define activity regions
activity_threshold = 1e-6
activity_mask = activity > activity_threshold
non_activity_mask = ~activity_mask

print(f"\nActivity voxels: {np.sum(activity_mask)}")
print(f"Non-activity voxels: {np.sum(non_activity_mask)}")

# Analyze dose in activity regions
condensed_activity = condensed_dose[activity_mask]
local_activity = local_dose[activity_mask]

print("\n--- Activity Region ---")
print(f"Condensed non-zero voxels: {np.sum(condensed_activity > 0)}")
print(f"Local non-zero voxels: {np.sum(local_activity > 0)}")
print(f"Condensed max dose: {condensed_activity.max():.6e}")
print(f"Local max dose: {local_activity.max():.6e}")
print(f"Condensed mean dose (non-zero): {condensed_activity[condensed_activity > 0].mean():.6e}")
print(f"Local mean dose (non-zero): {local_activity[local_activity > 0].mean():.6e}")

# Analyze dose in non-activity regions
condensed_non_activity = condensed_dose[non_activity_mask]
local_non_activity = local_dose[non_activity_mask]

print("\n--- Non-Activity Region ---")
print(f"Condensed non-zero voxels: {np.sum(condensed_non_activity > 0)}")
print(f"Local non-zero voxels: {np.sum(local_non_activity > 0)}")
print(f"Condensed max dose: {condensed_non_activity.max():.6e}")
print(f"Local max dose: {local_non_activity.max():.6e}")
print(f"Condensed mean dose (non-zero): {condensed_non_activity[condensed_non_activity > 0].mean():.6e}")
print(f"Local mean dose (non-zero): {local_non_activity[local_non_activity > 0].mean():.6e}")

# Calculate ratios in each region
ratio_activity = condensed_activity / np.maximum(local_activity, 1e-12)
ratio_non_activity = condensed_non_activity / np.maximum(local_non_activity, 1e-12)

print("\n--- Ratio Analysis ---")
print(f"Activity region - Max ratio: {ratio_activity[ratio_activity > 0].max():.6e}")
print(f"Activity region - Mean ratio: {ratio_activity[ratio_activity > 0].mean():.6e}")
print(f"Non-activity region - Max ratio: {ratio_non_activity[ratio_non_activity > 0].max():.6e}")
print(f"Non-activity region - Mean ratio: {ratio_non_activity[ratio_non_activity > 0].mean():.6e}")

# Check if condensed has dose in non-activity regions where local doesn't
condensed_only_non_activity = (condensed_non_activity > 0) & (local_non_activity == 0)
print(f"\nCondensed-only non-activity voxels: {np.sum(condensed_only_non_activity)}")
if np.sum(condensed_only_non_activity) > 0:
    print(f"  Max dose in these voxels: {condensed_non_activity[condensed_only_non_activity].max():.6e}")
    print(f"  Mean dose in these voxels: {condensed_non_activity[condensed_only_non_activity].mean():.6e}")

# Check if local has dose in non-activity regions where condensed doesn't
local_only_non_activity = (local_non_activity > 0) & (condensed_non_activity == 0)
print(f"Local-only non-activity voxels: {np.sum(local_only_non_activity)}")

print("\n" + "=" * 80)
