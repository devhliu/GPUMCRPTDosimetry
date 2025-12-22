#!/usr/bin/env python3
"""
This script generates pre-computed physics tables for GPUMCRPTDosimetry.

It uses the nist-calculators library to fetch real physics data from NIST databases
(XCOM for photons, ESTAR for electrons) and saves it into HDF5 files compatible
with the transport engine.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import h5py
import xcom
from star import electron

# Add src directory to path to allow imports from gpumcrpt
sys.path.append(str(Path(__file__).parent.parent))

from gpumcrpt.materials.materials_registry import list_material_tables, get_material_table
from gpumcrpt.materials.hu_materials import build_materials_library_from_config
from gpumcrpt.physics_tables.tables import PhysicsTables


def get_material_composition_for_xcom(material_library, material_name):
    """Formats material composition for the xcom library."""
    try:
        mat_idx = material_library.material_names.index(material_name)
    except ValueError:
        raise ValueError(f"Material '{material_name}' not found in library.")
    
    composition_vector = material_library.material_wfrac[mat_idx]
    elements = material_library.element_Z
    
    composition = {}
    for i, z in enumerate(elements):
        weight_fraction = composition_vector[i].item()
        if weight_fraction > 0:
            composition[int(z.item())] = weight_fraction
            
    return composition

def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed physics tables for GPUMCRPTDosimetry."
    )
    parser.add_argument(
        "--material_library",
        required=True,
        choices=[table.name for table in list_material_tables()],
        help="Name of the material library to use.",
    )
    parser.add_argument(
        "--physics_mode",
        required=True,
        choices=["local_deposit", "photon_only", "photon_em_condensedhistory"],
        help="The physics mode for which to generate the tables.",
    )
    parser.add_argument(
        "--output_dir",
        default="src/gpumcrpt/physics_tables/precomputed_tables",
        help="Directory to save the HDF5 files.",
    )
    parser.add_argument(
        "--num_energy_bins",
        type=int,
        default=500,
        help="Number of energy bins for the tables."
    )
    parser.add_argument(
        "--min_energy_mev",
        type=float,
        default=0.01, # 10 keV
        help="Minimum energy in MeV."
    )
    parser.add_argument(
        "--max_energy_mev",
        type=float,
        default=10.0, # 10 MeV
        help="Maximum energy in MeV."
    )
    args = parser.parse_args()

    print(f"Generating physics tables for material library: '{args.material_library}' and physics mode: '{args.physics_mode}'")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.material_library}-{args.physics_mode}.h5"
    output_path = output_dir / filename

    # 1. Load the material library
    mat_table_config = get_material_table(args.material_library)
    if mat_table_config is None:
        raise ValueError(f"Material table '{args.material_library}' not found.")
    
    materials_library = build_materials_library_from_config(
        cfg={"material_library": mat_table_config.material_library},
        device="cpu"
    )
    mat_names = materials_library.material_names
    num_materials = len(mat_names)

    # 2. Define energy grid
    e_edges = np.logspace(np.log10(args.min_energy_mev), np.log10(args.max_energy_mev), args.num_energy_bins + 1, dtype=np.float32)
    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:])

    # Initialize data arrays
    sigma_photo = np.zeros((num_materials, args.num_energy_bins), dtype=np.float32)
    sigma_compton = np.zeros((num_materials, args.num_energy_bins), dtype=np.float32)
    sigma_rayleigh = np.zeros((num_materials, args.num_energy_bins), dtype=np.float32)
    sigma_pair = np.zeros((num_materials, args.num_energy_bins), dtype=np.float32)
    s_restricted = np.zeros((num_materials, args.num_energy_bins), dtype=np.float32)
    range_csda = np.zeros((num_materials, args.num_energy_bins), dtype=np.float32)
    
    # 3. Fetch physics data
    if args.physics_mode in ["photon_only", "photon_em_condensedhistory"]:
        print("Fetching photon cross-sections from NIST XCOM...")
        for i, mat_name in enumerate(mat_names):
            print(f"  - Photon data for {mat_name}")
            composition = get_material_composition_for_xcom(materials_library, mat_name)
            
            # xcom expects energy in eV
            xcom_data = xcom.calculate_cross_section(composition, e_centers * 1e6)
            
            # xcom returns cross sections in cm^2/g
            sigma_photo[i, :] = xcom_data['photoelectric']
            sigma_compton[i, :] = xcom_data['compton']
            sigma_rayleigh[i, :] = xcom_data['rayleigh']
            sigma_pair[i, :] = xcom_data['pair']

    if args.physics_mode == "photon_em_condensedhistory":
        print("Fetching electron data from NIST ESTAR...")
        for i, mat_name in enumerate(mat_names):
            print(f"  - Electron data for {mat_name}")
            composition = get_material_composition_for_xcom(materials_library, mat_name)
            
            # estar expects energy in MeV
            estar_data = electron.calculate_stopping_power(composition, energy=e_centers)
            
            # estar returns total stopping power in MeV cm^2 / g
            s_restricted[i, :] = estar_data['total']
            
            # Get CSDA range
            range_data = electron.calculate_csda_range(composition, energy=e_centers)
            range_csda[i, :] = range_data['csda_range']


    sigma_total = sigma_photo + sigma_compton + sigma_rayleigh + sigma_pair
    # sigma_max is the max total cross section over all materials for each energy
    sigma_max = np.max(sigma_total, axis=0)
    
    # 4. Save to HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['material_library'] = args.material_library
        f.attrs['physics_mode'] = args.physics_mode
        
        meta = f.create_group("meta")
        meta.attrs['schema_version'] = '1.0'
        
        materials_group = meta.create_group("materials")
        materials_group.create_dataset("names", data=[m.encode('utf-8') for m in mat_names])
        materials_group.create_dataset("ref_density_g_cm3", data=materials_library.density_g_cm3.numpy())

        energy_group = f.create_group("energy")
        energy_group.create_dataset("edges_MeV", data=e_edges)
        energy_group.create_dataset("centers_MeV", data=e_centers)
        
        if args.physics_mode in ["photon_only", "photon_em_condensedhistory"]:
            photons_group = f.create_group("photons")
            photons_group.create_dataset("sigma_photo", data=sigma_photo)
            photons_group.create_dataset("sigma_compton", data=sigma_compton)
            photons_group.create_dataset("sigma_rayleigh", data=sigma_rayleigh)
            photons_group.create_dataset("sigma_pair", data=sigma_pair)
            photons_group.create_dataset("sigma_total", data=sigma_total)
            photons_group.create_dataset("sigma_max", data=sigma_max)

            # The kernel calculates cumulative probability on the fly, so we can store zeros here.
            p_cum = np.zeros_like(sigma_total)
            photons_group.create_dataset("p_cum", data=p_cum)

        if args.physics_mode == "photon_em_condensedhistory":
            electrons_group = f.create_group("electrons")
            electrons_group.create_dataset("S_restricted", data=s_restricted)
            electrons_group.create_dataset("range_csda_cm", data=range_csda)
            
            # Placeholders for brem and delta probabilities
            dummy_electron_data = np.zeros_like(s_restricted)
            electrons_group.create_dataset("P_brem_per_cm", data=dummy_electron_data)
            electrons_group.create_dataset("P_delta_per_cm", data=dummy_electron_data)

        # Placeholders for samplers
        samplers_group = f.create_group("samplers")
        photon_samplers = samplers_group.create_group("photon")
        compton_sampler = photon_samplers.create_group("compton")
        compton_sampler.attrs['convention'] = 'cos_theta'
        # Klein-Nishina inverse CDF is needed for compton sampling
        # This is a complex calculation, so we'll use a placeholder for now
        # Shape: [num_energy_bins, num_probability_bins]
        compton_inv_cdf = np.linspace(-1, 1, 256)[np.newaxis, :] * np.ones((args.num_energy_bins, 1))
        compton_sampler.create_dataset("inv_cdf", data=compton_inv_cdf)

    print(f"Successfully generated physics tables at: {output_path}")


if __name__ == "__main__":
    main()
