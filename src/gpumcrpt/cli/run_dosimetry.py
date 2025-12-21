#!/usr/bin/env python
"""
Dosimetry Calculation Script for Multiple Methods

This script performs Monte Carlo dosimetry calculations using three different methods:
1. Local deposit - Only local energy deposition
2. Photon-only - Only photon transport with local electron deposition  
3. Photon-EM condensed history - Full photon and electron transport

Usage:
    python run_dosimetry.py --config <config_file> --method <method_name>
    
Example:
    python run_dosimetry.py --config icrp110_lu177_phantom/config/icrp110_lu177_config.yaml --method local_deposit
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import torch

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import nibabel as nib
import h5py
import numpy as np
import torch
from gpumcrpt.materials.hu_materials import (
    build_default_materials_library,
    build_materials_from_hu,
    build_materials_library_from_config,
)
from gpumcrpt.physics_tables.tables import load_physics_tables_h5
from gpumcrpt.decaydb import load_icrp107_nuclide
from gpumcrpt.source.sampling import sample_weighted_decays_and_primaries
from gpumcrpt.transport.engine import TransportEngine
from gpumcrpt.dose.scoring import edep_to_dose_and_uncertainty


def load_local_deposit_physics_tables(path: str, device: str = "cuda"):
    """Load simplified physics tables for local deposit method.
    
    This function handles the simplified structure of local_deposit_physics.h5
    which doesn't have the '/meta' group required by load_physics_tables_h5.
    """
    import h5py
    import numpy as np
    import torch
    from gpumcrpt.physics_tables.tables import PhysicsTables
    
    with h5py.File(path, "r") as f:
        # Load energy deposition data
        energy_deposition_group = f["/energy_deposition"]
        energies = np.asarray(energy_deposition_group["energies"], dtype=np.float32)
        
        # Load mass energy absorption coefficients for different materials
        materials = ["air", "cortical_bone", "fat", "lung", "muscle", "soft_tissue", "trabecular_bone"]
        mu_en_rho_data = {}
        for material in materials:
            dataset_name = f"{material}_mu_en_rho"
            if dataset_name in energy_deposition_group:
                mu_en_rho_data[material] = np.asarray(energy_deposition_group[dataset_name], dtype=np.float32)
        
        # Convert to torch tensors
        energies_tensor = torch.from_numpy(energies).to(device=device)
        
        # Create a minimal PhysicsTables object
        # For local deposit method, we only need basic energy information
        # Other attributes can be set to dummy values since they won't be used
        
        # Create a simple energy grid (edges and centers)
        e_edges_MeV = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)  # Simple dummy energy range
        e_centers_MeV = torch.tensor([0.5], dtype=torch.float32, device=device)     # Single energy center
        
        # Create minimal material information
        material_names = materials
        ref_density_g_cm3 = torch.ones(len(materials), dtype=torch.float32, device=device)
        
        # Create dummy cross-section tensors (not used by local deposit)
        dummy_shape = (len(materials), 1)  # (materials, energy_bins)
        dummy_tensor = torch.zeros(dummy_shape, dtype=torch.float32, device=device)
        
        return PhysicsTables(
            e_edges_MeV=e_edges_MeV,
            e_centers_MeV=e_centers_MeV,
            material_names=material_names,
            ref_density_g_cm3=ref_density_g_cm3,
            sigma_photo=dummy_tensor,
            sigma_compton=dummy_tensor,
            sigma_rayleigh=dummy_tensor,
            sigma_pair=dummy_tensor,
            sigma_total=dummy_tensor,
            p_cum=dummy_tensor,
            sigma_max=dummy_tensor,
            S_restricted=dummy_tensor,
            range_csda_cm=dummy_tensor,
            P_brem_per_cm=None,
            P_delta_per_cm=None,
            compton_inv_cdf=None,
            compton_convention="cos_theta",
            brem_inv_cdf_Efrac=None,
            delta_inv_cdf_Efrac=None
        )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_dosimetry(config_path: str, method_name: str) -> None:
    """Run dosimetry calculation for a specific method."""
    
    # Load main configuration
    main_config = load_config(config_path)
    
    # Load method-specific configuration
    method_config_path = main_config['methods'][method_name]
    method_config = load_config(method_config_path)
    
    # Set up paths
    config_dir = Path(config_path).parent
    output_dir = config_dir.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Load input data (using relative paths from config file)
    ct_path = config_dir / main_config['input']['ct']
    tia_path = config_dir / main_config['input']['tia']
    nuclide_name = main_config['input']['nuclide']
    
    # Load CT and TIA data
    ct_img = nib.load(ct_path)
    tia_img = nib.load(tia_path)
    
    # Resample CT to match TIA coordinate system and matrix size
    from nibabel.processing import resample_from_to
    ct_img_resampled = resample_from_to(ct_img, tia_img)
    
    ct_data = ct_img_resampled.get_fdata()
    tia_data = tia_img.get_fdata()  # TIA in Bq*s
    
    # Build materials library
    try:
        materials_library = build_materials_library_from_config(method_config)
    except KeyError:
        # Fall back to default materials library if material_library section is not found
        materials_library = build_default_materials_library()
        # Add the hu_to_density and hu_to_class mappings from the config
        materials_library.hu_to_density = method_config['materials']['hu_to_density']
        materials_library.hu_to_class = method_config['materials']['hu_to_class']
    
    # Load physics tables based on method
    physics_tables_path = method_config['physics_tables']['h5_path']
    if method_name == 'local_deposit':
        # Use simplified loading for local deposit method
        physics_tables = load_local_deposit_physics_tables(physics_tables_path)
    else:
        # Use standard loading for other methods
        physics_tables = load_physics_tables_h5(physics_tables_path)
    
    # Load nuclide data
    nuclide_name = main_config['input']['nuclide']
    db_dir = Path(__file__).parent.parent.parent / "src" / "gpumcrpt" / "decaydb" / "icrp107_database" / "icrp107"
    nuclide = load_icrp107_nuclide(db_dir, nuclide_name)
    
    # Build materials from HU values
    # Convert CT data to PyTorch tensor
    ct_tensor = torch.from_numpy(ct_data).to(device=method_config['device'])
    materials = build_materials_from_hu(
        hu=ct_tensor, 
        hu_to_density=materials_library.hu_to_density, 
        hu_to_class=materials_library.hu_to_class,
        material_library=materials_library
    )
    
    # Sample decays using TIA (Bq*s) directly
    n_histories = method_config['monte_carlo']['n_histories']
    
    # Convert TIA data to PyTorch tensor and get voxel size from resampled CT image
    tia_tensor = torch.from_numpy(tia_data).to(device=method_config['device'])
    voxel_size_cm = tuple(float(x) * 0.1 for x in ct_img_resampled.header.get_zooms())  # Convert mm to cm
    
    primaries, local_edep = sample_weighted_decays_and_primaries(
        activity_bqs=tia_tensor,
        voxel_size_cm=voxel_size_cm,
        affine=ct_img_resampled.affine,
        nuclide=nuclide,
        n_histories=n_histories,
        seed=method_config['monte_carlo'].get('seed', 42),
        device=method_config['device'],
        cutoffs=method_config['cutoffs'],
        sampling_mode=method_config['monte_carlo'].get('sampling_mode', 'accurate')
    )

    # Initialize transport engine
    engine = TransportEngine(
        device=method_config['device'],
        mats=materials,
        tables=physics_tables,
        sim_config=method_config,
        voxel_size_cm=voxel_size_cm
    )
    
    # Run transport using the correct API
    n_batches = method_config['monte_carlo']['n_batches']
    edep_batches = engine.run_batches(
        primaries=primaries, 
        alpha_local_edep=local_edep, 
        n_batches=n_batches
    )
    
    # Calculate mean and uncertainty from batches
    edep = torch.mean(edep_batches, dim=0)
    edep_uncertainty = torch.std(edep_batches, dim=0) / torch.sqrt(torch.tensor(n_batches, dtype=torch.float32))
    
    # Convert energy deposition to dose
    # Calculate voxel volume in cm³
    voxel_volume_cm3 = voxel_size_cm[0] * voxel_size_cm[1] * voxel_size_cm[2]
    
    # Convert energy deposition to dose
    dose, dose_uncertainty = edep_to_dose_and_uncertainty(
        edep_batches=edep_batches,
        rho=materials.rho,
        voxel_volume_cm3=voxel_volume_cm3,
        uncertainty_mode="relative"
    )
    
    # Save results
    output_filename = f"{nuclide_name.lower()}_dose_{method_name}.nii.gz"
    uncertainty_filename = f"{nuclide_name.lower()}_uncertainty_{method_name}.nii.gz"
    
    # Move tensors to CPU and convert to numpy arrays
    dose_cpu = dose.cpu().numpy()
    dose_uncertainty_cpu = dose_uncertainty.cpu().numpy()
    
    dose_img = nib.Nifti1Image(dose_cpu, ct_img_resampled.affine, ct_img_resampled.header)
    uncertainty_img = nib.Nifti1Image(dose_uncertainty_cpu, ct_img_resampled.affine, ct_img_resampled.header)
    
    nib.save(dose_img, output_dir / output_filename)
    nib.save(uncertainty_img, output_dir / uncertainty_filename)
    
    # Save summary
    summary_path = output_dir / f"dosimetry_summary_{method_name}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Dosimetry Summary - {method_name}\n")
        f.write(f"Nuclide: {nuclide_name}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Number of histories: {n_histories}\n")
        f.write(f"Number of batches: {n_batches}\n")
        f.write(f"Max dose: {dose.max():.6f} Gy\n")
        f.write(f"Mean dose: {dose.mean():.6f} Gy\n")
        f.write(f"Dose file: {output_filename}\n")
        f.write(f"Uncertainty file: {uncertainty_filename}\n")
    
    print(f"Dosimetry calculation completed for {method_name}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run dosimetry calculation')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--method', required=True, 
                       choices=['local_deposit', 'photon_em_condensed', 'photon_only'],
                       help='Method to use for dosimetry calculation')
    
    args = parser.parse_args()
    
    try:
        run_dosimetry(args.config, args.method)
    except Exception as e:
        print(f"Error running dosimetry: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()