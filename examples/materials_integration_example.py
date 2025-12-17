"""
Integration example demonstrating how to use the Materials Management System
with the existing GPU-accelerated Monte Carlo dosimetry pipeline.

This example shows:
1. Using the MaterialsManager to handle multiple HU materials tables
2. Generating pre-computed physics data for different materials configurations
3. Integrating with the existing pipeline for dosimetry calculations
4. Custom materials table creation and usage
"""

import os
import torch
import yaml
from pathlib import Path

# Import the materials management system
from gpumcrpt.materials.materials_manager import MaterialsManager
from gpumcrpt.materials.materials_registry import get_default_registry

# Import existing pipeline components
from gpumcrpt.pipeline import run_dosimetry
from gpumcrpt.io.nifti import load_nifti


def example_basic_integration():
    """Basic integration example showing how to use MaterialsManager with the pipeline."""
    
    print("=== Basic Materials Management Integration Example ===\n")
    
    # Initialize the materials manager
    manager = MaterialsManager()
    
    # Show available materials tables
    tables = manager.get_available_tables()
    print(f"Available materials tables: {tables}")
    
    # Get information about each table
    for table_name in tables:
        info = manager.get_table_info(table_name)
        print(f"\nTable '{table_name}':")
        print(f"  Description: {info['description']}")
        print(f"  Source: {info['source']}")
        print(f"  Default: {info['is_default']}")
        print(f"  Materials: {info['num_materials']}")
        print(f"  Elements: {info['num_elements']}")
    
    # Generate pre-computed physics data for the default table
    print("\n=== Generating Pre-computed Physics Data ===")
    
    # Generate basic materials data
    basic_data = manager.generate_basic_materials_data("default_icru44")
    print(f"✓ Basic data generated for {len(basic_data.material_names)} materials")
    
    # Generate comprehensive physics tables
    physics_data = manager.generate_physics_tables("default_icru44")
    print(f"✓ Physics tables generated with {len(physics_data.energy_bins)} energy bins")
    
    # Save pre-computed data for later use
    print("\n=== Saving Pre-computed Data ===")
    
    basic_file = manager.save_precomputed_data("default_icru44", "default_icru44_basic.h5", include_physics=False)
    physics_file = manager.save_precomputed_data("default_icru44", "default_icru44_physics.h5", include_physics=True)
    
    print(f"✓ Basic data saved to: {basic_file}")
    print(f"✓ Physics data saved to: {physics_file}")
    
    return manager, basic_file, physics_file


def example_pipeline_integration():
    """Example showing integration with the main dosimetry pipeline."""
    
    print("\n=== Pipeline Integration Example ===\n")
    
    # Create a configuration that uses the materials management system
    config = {
        "device": "cuda",
        "seed": 123,
        
        "io": {
            "resample_ct_to_activity": True,
            "output_uncertainty": "relative"
        },
        
        "cutoffs": {
            "photon_keV": 3.0,
            "electron_keV": 20.0
        },
        
        "materials": {
            # Use materials management system instead of direct configuration
            "use_materials_manager": True,
            "materials_table": "default_icru44",
            
            # Fallback to traditional configuration if needed
            "hu_to_density": [
                [-1000, 0.0012],   # Air
                [-700, 0.30],      # Lung
                [0, 1.00],         # Soft Tissue
                [1000, 1.85]       # Bone
            ],
            "hu_to_class": [
                [-1000, -700, 0],  # air
                [-700, -200, 1],   # lung
                [-200, 300, 2],    # soft tissue
                [300, 3000, 3]     # bone
            ]
        },
        
        "physics_tables": {
            "h5_path": "default_icru44_physics.h5"  # Use pre-computed data
        },
        
        "decaydb": {
            "type": "icrp107_json",
            "path": "icrp107_json_db"
        },
        
        "nuclide": {
            "name": "Lu-177"
        },
        
        "monte_carlo": {
            "n_histories": 100000,
            "n_batches": 10,
            "max_wavefront_iters": 1000,
            
            "triton": {
                "engine": "photon_only",
                "use_prefixsum_compaction": True,
                "enable_cuda_graphs": False
            }
        }
    }
    
    print("Configuration created for pipeline integration")
    print(f"  Materials table: {config['materials']['materials_table']}")
    print(f"  Physics tables: {config['physics_tables']['h5_path']}")
    
    return config


def example_custom_materials():
    """Example showing custom materials table creation and usage."""
    
    print("\n=== Custom Materials Table Example ===\n")
    
    # Create a custom materials configuration
    custom_config = {
        "name": "custom_patient_specific",
        "description": "Patient-specific materials based on clinical measurements",
        "source": "Clinical Study",
        "is_default": False,
        
        "hu_to_density": [
            [-1000, 0.0012],   # Air
            [-800, 0.25],      # Low-density lung
            [-600, 0.45],      # High-density lung
            [-100, 0.92],      # Fat
            [0, 1.02],         # Soft tissue (slightly denser than water)
            [50, 1.08],        # Muscle
            [200, 1.25],       # Low-density bone
            [800, 1.65],       # Medium-density bone
            [2000, 1.95]       # High-density bone
        ],
        
        "hu_to_class": [
            [-1000, -800, 0],  # air
            [-800, -600, 1],   # low_density_lung
            [-600, -100, 2],   # high_density_lung
            [-100, 0, 3],      # fat
            [0, 50, 4],        # soft_tissue
            [50, 200, 5],      # muscle
            [200, 800, 6],     # low_density_bone
            [800, 2000, 7],    # medium_density_bone
            [2000, 3000, 8]    # high_density_bone
        ],
        
        "material_library": {
            "elements": ["H", "C", "N", "O", "P", "Ca", "Na", "Cl"],
            "materials": [
                {
                    "name": "air",
                    "ref_density_g_cm3": 0.0012,
                    "wfrac": {"N": 0.79, "O": 0.21}
                },
                {
                    "name": "low_density_lung",
                    "ref_density_g_cm3": 0.25,
                    "wfrac": {"H": 0.103, "C": 0.105, "N": 0.031, "O": 0.761}
                },
                {
                    "name": "high_density_lung",
                    "ref_density_g_cm3": 0.45,
                    "wfrac": {"H": 0.103, "C": 0.105, "N": 0.031, "O": 0.761}
                },
                {
                    "name": "fat",
                    "ref_density_g_cm3": 0.92,
                    "wfrac": {"H": 0.114, "C": 0.598, "O": 0.288}
                },
                {
                    "name": "soft_tissue",
                    "ref_density_g_cm3": 1.02,
                    "wfrac": {"H": 0.101, "C": 0.111, "N": 0.026, "O": 0.762}
                },
                {
                    "name": "muscle",
                    "ref_density_g_cm3": 1.08,
                    "wfrac": {"H": 0.102, "C": 0.143, "N": 0.034, "O": 0.721}
                },
                {
                    "name": "low_density_bone",
                    "ref_density_g_cm3": 1.25,
                    "wfrac": {"H": 0.064, "C": 0.278, "N": 0.027, "O": 0.410, "P": 0.070, "Ca": 0.151}
                },
                {
                    "name": "medium_density_bone",
                    "ref_density_g_cm3": 1.65,
                    "wfrac": {"H": 0.034, "C": 0.155, "N": 0.042, "O": 0.435, "P": 0.103, "Ca": 0.231}
                },
                {
                    "name": "high_density_bone",
                    "ref_density_g_cm3": 1.95,
                    "wfrac": {"H": 0.034, "C": 0.155, "N": 0.042, "O": 0.435, "P": 0.103, "Ca": 0.231}
                }
            ]
        }
    }
    
    # Register the custom table
    registry = get_default_registry()
    registry.register_table(custom_config)
    
    # Create manager with updated registry
    manager = MaterialsManager(registry)
    
    # Verify the custom table is available
    tables = manager.get_available_tables()
    print(f"Available tables after custom registration: {tables}")
    
    # Generate pre-computed data for the custom table
    custom_file = manager.save_precomputed_data("custom_patient_specific", "custom_patient_physics.h5", include_physics=True)
    print(f"✓ Custom physics data saved to: {custom_file}")
    
    return manager, custom_config


def example_hu_conversion():
    """Example showing HU to materials conversion using different tables."""
    
    print("\n=== HU Conversion Example ===\n")
    
    manager = MaterialsManager()
    
    # Create a test HU volume
    hu_volume = torch.tensor([
        [-1000, -500, -100],  # Air, Lung, Fat
        [0, 40, 100],         # Water, Muscle, Trabecular bone
        [600, 1200, 3000]     # Cortical bone, Dense bone, Very dense bone
    ], dtype=torch.float32, device="cuda")
    
    print(f"Test HU volume shape: {hu_volume.shape}")
    print(f"HU values range: {hu_volume.min():.0f} to {hu_volume.max():.0f}")
    
    # Convert using different materials tables
    tables_to_test = ["default_icru44", "schneider_parameterization", "simple_test"]
    
    for table_name in tables_to_test:
        print(f"\n--- Using table '{table_name}' ---")
        
        # Convert HU to materials
        materials_volume = manager.get_materials_volume(hu_volume, table_name)
        
        print(f"Material IDs shape: {materials_volume.material_id.shape}")
        print(f"Density shape: {materials_volume.rho.shape}")
        print(f"Material IDs:\n{materials_volume.material_id.cpu().numpy()}")
        print(f"Densities (g/cm³):\n{materials_volume.rho.cpu().numpy()}")
    
    return hu_volume


def main():
    """Run all integration examples."""
    
    print("=" * 60)
    print("Materials Management System Integration Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic integration
        manager, basic_file, physics_file = example_basic_integration()
        
        # Example 2: Pipeline integration
        pipeline_config = example_pipeline_integration()
        
        # Example 3: Custom materials
        custom_manager, custom_config = example_custom_materials()
        
        # Example 4: HU conversion
        test_hu_volume = example_hu_conversion()
        
        print("\n" + "=" * 60)
        print("✅ All integration examples completed successfully!")
        print("=" * 60)
        
        print("\nSummary of generated files:")
        print(f"  • Basic materials data: {basic_file}")
        print(f"  • Physics tables: {physics_file}")
        print(f"  • Custom physics tables: custom_patient_physics.h5")
        
        print("\nNext steps:")
        print("  1. Use the generated physics tables in your dosimetry pipeline")
        print("  2. Modify configs/example_simulation.yaml to use materials_manager")
        print("  3. Create custom materials tables for specific patient populations")
        print("  4. Generate pre-computed data for different energy ranges")
        
    except Exception as e:
        print(f"\n❌ Error in integration example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()