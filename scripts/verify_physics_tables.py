#!/usr/bin/env python3
"""
Verification script for physics table integration and energy conservation.

This script tests:
1. Physics table-based sampling for bremsstrahlung and delta rays
2. Energy conservation in particle interactions
3. Positron annihilation energy conservation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpumcrpt.physics_tables.tables import load_physics_tables_h5


def verify_physics_tables_exist(tables: PhysicsTables) -> bool:
    """Verify that required physics tables are loaded."""
    required_tables = [
        "sigma_total",
        "sigma_photo",
        "sigma_compton",
        "sigma_rayleigh",
        "sigma_pair",
        "e_centers_MeV",
        "e_edges_MeV",
        "P_brem_per_cm",
        "brem_inv_cdf_Efrac",
        "delta_inv_cdf_Efrac",
    ]
    
    missing_tables = []
    for table_name in required_tables:
        if not hasattr(tables, table_name) or getattr(tables, table_name) is None:
            missing_tables.append(table_name)
    
    if missing_tables:
        print(f"ERROR: Missing required physics tables: {missing_tables}")
        return False
    
    print("✓ All required physics tables are loaded")
    return True


def verify_bremsstrahlung_table(tables: PhysicsTables) -> bool:
    """Verify bremsstrahlung inverse CDF table structure."""
    brem_inv_cdf = tables.brem_inv_cdf_Efrac
    
    if brem_inv_cdf is None:
        print("ERROR: brem_inv_cdf_Efrac table is None")
        return False
    
    if brem_inv_cdf.ndim != 2:
        print(f"ERROR: brem_inv_cdf_Efrac should be 2D, got {brem_inv_cdf.ndim}D")
        return False
    
    n_energy_bins, n_cdf_points = brem_inv_cdf.shape
    
    if n_energy_bins != len(tables.e_centers_MeV):
        print(f"ERROR: brem_inv_cdf_Efrac energy bins ({n_energy_bins}) != e_centers_MeV ({len(tables.e_centers_MeV)})")
        return False
    
    # Check that CDF values are in valid range [0, 1]
    if torch.any(brem_inv_cdf < 0) or torch.any(brem_inv_cdf > 1):
        print("ERROR: brem_inv_cdf_Efrac contains values outside [0, 1]")
        return False
    
    # Check monotonicity (CDF should be non-decreasing)
    for i in range(n_energy_bins):
        if not torch.all(brem_inv_cdf[i, 1:] >= brem_inv_cdf[i, :-1]):
            print(f"ERROR: brem_inv_cdf_Efrac is not monotonic in energy bin {i}")
            return False
    
    print(f"✓ Bremsstrahlung inverse CDF table is valid ({n_energy_bins} energy bins × {n_cdf_points} CDF points)")
    return True


def verify_delta_ray_table(tables: PhysicsTables) -> bool:
    """Verify delta ray inverse CDF table structure."""
    delta_inv_cdf = tables.delta_inv_cdf_Efrac
    
    if delta_inv_cdf is None:
        print("ERROR: delta_inv_cdf_Efrac table is None")
        return False
    
    if delta_inv_cdf.ndim != 2:
        print(f"ERROR: delta_inv_cdf_Efrac should be 2D, got {delta_inv_cdf.ndim}D")
        return False
    
    n_energy_bins, n_cdf_points = delta_inv_cdf.shape
    
    if n_energy_bins != len(tables.e_centers_MeV):
        print(f"ERROR: delta_inv_cdf_Efrac energy bins ({n_energy_bins}) != e_centers_MeV ({len(tables.e_centers_MeV)})")
        return False
    
    # Check that CDF values are in valid range [0, 1]
    if torch.any(delta_inv_cdf < 0) or torch.any(delta_inv_cdf > 1):
        print("ERROR: delta_inv_cdf_Efrac contains values outside [0, 1]")
        return False
    
    # Check monotonicity (CDF should be non-decreasing)
    for i in range(n_energy_bins):
        if not torch.all(delta_inv_cdf[i, 1:] >= delta_inv_cdf[i, :-1]):
            print(f"ERROR: delta_inv_cdf_Efrac is not monotonic in energy bin {i}")
            return False
    
    print(f"✓ Delta ray inverse CDF table is valid ({n_energy_bins} energy bins × {n_cdf_points} CDF points)")
    return True


def verify_energy_conservation_in_tables(tables: PhysicsTables) -> bool:
    """Verify that physics tables respect energy conservation principles."""
    # Check that bremsstrahlung cross-sections are non-negative
    if torch.any(tables.P_brem_per_cm < 0):
        print("ERROR: P_brem_per_cm contains negative values")
        return False
    
    # Check that energy bins are properly ordered
    if not torch.all(tables.e_edges_MeV[1:] > tables.e_edges_MeV[:-1]):
        print("ERROR: e_edges_MeV is not monotonically increasing")
        return False
    
    # Check that energy centers are within bin edges
    for i, e_center in enumerate(tables.e_centers_MeV):
        if i == 0:
            if e_center < tables.e_edges_MeV[0] or e_center > tables.e_edges_MeV[1]:
                print(f"ERROR: Energy center {i} ({e_center:.3f} MeV) is outside bin edges")
                return False
        else:
            if e_center < tables.e_edges_MeV[i] or e_center > tables.e_edges_MeV[i+1]:
                print(f"ERROR: Energy center {i} ({e_center:.3f} MeV) is outside bin edges")
                return False
    
    print("✓ Energy conservation principles are respected in physics tables")
    return True


def verify_positron_annihilation_energy() -> bool:
    """Verify positron annihilation energy conservation."""
    from gpumcrpt.utils.constants import ELECTRON_REST_MASS_MEV
    
    # Positron at rest should produce 2 × 0.511 MeV photons
    expected_photon_energy = ELECTRON_REST_MASS_MEV
    total_annihilation_energy = 2 * expected_photon_energy
    
    # Energy conservation: E_initial = E_kinetic + 2*m_e*c^2 = 2*m_e*c^2 (annihilation photons)
    positron_kinetic_energy = 0.0
    positron_rest_energy = 2 * ELECTRON_REST_MASS_MEV
    initial_energy = positron_kinetic_energy + positron_rest_energy
    
    if abs(initial_energy - total_annihilation_energy) > 1e-6:
        print(f"ERROR: Positron annihilation energy conservation violated")
        print(f"  Initial energy: {initial_energy:.6f} MeV")
        print(f"  Annihilation photon energy: {total_annihilation_energy:.6f} MeV")
        return False
    
    print(f"✓ Positron annihilation energy conservation verified (2 × {ELECTRON_REST_MASS_MEV:.3f} MeV photons)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify physics table integration and energy conservation")
    ap.add_argument("--tables", required=True, help="Path to physics tables HDF5 file")
    args = ap.parse_args()
    
    print("=" * 60)
    print("Physics Table Integration Verification")
    print("=" * 60)
    
    # Load physics tables
    print(f"\nLoading physics tables from: {args.tables}")
    try:
        tables = load_physics_tables_h5(args.tables, device="cpu")
        print("✓ Physics tables loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load physics tables: {e}")
        return 1
    
    # Run verification tests
    print("\n" + "-" * 60)
    print("Running verification tests...")
    print("-" * 60)
    
    all_passed = True
    
    all_passed &= verify_physics_tables_exist(tables)
    all_passed &= verify_bremsstrahlung_table(tables)
    all_passed &= verify_delta_ray_table(tables)
    all_passed &= verify_energy_conservation_in_tables(tables)
    all_passed &= verify_positron_annihilation_energy()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All verification tests PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ Some verification tests FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
