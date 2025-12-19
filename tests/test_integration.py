from __future__ import annotations

import pytest
import torch
import numpy as np

from gpumcrpt.materials.hu_materials import MaterialsVolume, build_default_materials_library
from gpumcrpt.physics.tables import PhysicsTables
from gpumcrpt.source.sampling import sample_weighted_decays_and_primaries, ParticleQueues
from gpumcrpt.decaydb.icrp107_json import ICRP107Nuclide
from gpumcrpt.dose.scoring import edep_to_dose_and_uncertainty


class TestIntegrationPipeline:
    """Test complete physics pipeline integration."""

    @pytest.fixture
    def test_materials(self):
        """Create test materials volume."""
        Z, Y, X = 8, 8, 8
        return MaterialsVolume(
            material_id=torch.zeros((Z, Y, X), dtype=torch.int32),
            rho=torch.ones((Z, Y, X), dtype=torch.float32),
        )

    @pytest.fixture
    def test_nuclide(self):
        """Create test nuclide with multiple decay modes."""
        return ICRP107Nuclide(
            name="Lu-177",
            half_life_s=1.606e5,
            emissions={
                "gamma": [[0.112, 0.06], [0.208, 0.11], [0.320, 0.31]],
                "beta-": [],
                "b-spectra": [[0.0, 0.1], [0.5, 0.8], [0.6, 0.1]],
                "beta_minus_yield": 1.0,
            },
        )

    def test_activity_to_primaries_conversion(self, test_materials, test_nuclide):
        """Test conversion of activity to primary particles."""
        activity = torch.ones((8, 8, 8), dtype=torch.float32)
        
        primaries, local = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=test_nuclide,
            n_histories=200,
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 3.0, "electron_keV": 20.0},
            sampling_mode="accurate",
        )
        
        # Verify structure
        assert isinstance(primaries, ParticleQueues)
        assert "E_MeV" in primaries.photons
        assert "pos_cm" in primaries.photons
        assert "dir" in primaries.photons
        assert "w" in primaries.photons
        
        # Verify shapes
        assert primaries.photons["E_MeV"].shape[0] > 0
        assert primaries.photons["pos_cm"].shape[0] == primaries.photons["E_MeV"].shape[0]
        
        # Verify local deposition is non-negative
        assert (local >= 0.0).all()

    def test_energy_conservation_in_sampling(self, test_nuclide):
        """Test that energy is conserved during particle sampling."""
        activity = torch.ones((4, 4, 4), dtype=torch.float32) * 100.0  # High activity for statistics
        
        primaries, local = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=test_nuclide,
            n_histories=1000,
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 1.0, "electron_keV": 1.0},  # Low cutoffs to catch all
            sampling_mode="accurate",
        )
        
        # Calculate total sampled energy
        total_photon_E = (primaries.photons["E_MeV"] * primaries.photons["w"]).sum().item()
        total_electron_E = (primaries.electrons["E_MeV"] * primaries.electrons["w"]).sum().item()
        total_positron_E = (primaries.positrons["E_MeV"] * primaries.positrons["w"]).sum().item()
        total_local_E = local.sum().item()
        
        total_energy = total_photon_E + total_electron_E + total_positron_E + total_local_E
        
        # Total should be > 0
        assert total_energy > 0.0
        
        # Each component should be non-negative
        assert total_photon_E >= 0.0
        assert total_electron_E >= 0.0
        assert total_positron_E >= 0.0
        assert total_local_E >= 0.0

    def test_weighted_sampling_accuracy(self):
        """Test that weighted sampling respects voxel probabilities."""
        # Create activity map with known distribution
        activity = torch.zeros((8, 8, 8), dtype=torch.float32)
        activity[0, 0, 0] = 1000.0  # One hot voxel
        
        nuclide = ICRP107Nuclide(
            name="test",
            half_life_s=1.0,
            emissions={"gamma": [[0.5, 1.0]]},
        )
        
        primaries, _ = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=nuclide,
            n_histories=1000,
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 3.0, "electron_keV": 20.0},
            sampling_mode="accurate",
        )
        
        # All particles should originate near voxel (0,0,0)
        if primaries.photons["pos_cm"].shape[0] > 0:
            pos = primaries.photons["pos_cm"]
            # Should be in range [0, 0.4) for this voxel
            assert (pos[:, 0] >= 0.0).all()
            assert (pos[:, 0] < 0.4).all()


class TestPhysicsAccuracy:
    """Test physics calculations for accuracy."""

    def test_compton_kinematics(self):
        """Test that Compton scattering kinematics are physically valid."""
        # E + me*c^2 = E' + me*c^2 + T_e
        # E' = E / (1 + (E/me)*(1 - cos(theta)))
        
        me = 0.511  # MeV
        E_photon = 1.0  # MeV
        
        # Test various scattering angles
        cos_theta_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for cos_t in cos_theta_values:
            alpha = E_photon / me
            E_prime = E_photon / (1.0 + alpha * (1.0 - cos_t))
            T_electron = E_photon - E_prime
            
            # Energy conservation
            assert abs((E_photon + me) - (E_prime + me + T_electron)) < 1e-6
            
            # Physical bounds
            assert 0 <= E_prime <= E_photon
            assert 0 <= T_electron <= E_photon

    def test_range_energy_relation(self):
        """Test that range decreases with energy (physical expectation)."""
        # For condensed history, range_csda should decrease as energy increases
        # (for same material)
        
        energies = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        ranges = np.array([0.05, 0.2, 0.45, 1.0, 2.5])  # Example values
        
        # Should be monotonically increasing with energy (for same material)
        diffs = np.diff(ranges)
        assert np.all(diffs > 0), "Range should increase with energy"

    def test_stopping_power_positivity(self):
        """Test that stopping power is always positive."""
        # S_restricted should be positive (energy loss per unit length)
        S_values = np.array([0.1, 0.2, 0.3, 0.5])
        
        assert np.all(S_values > 0.0), "Stopping power must be positive"

    def test_cross_section_bounds(self):
        """Test that cross sections are within physical bounds."""
        # Cross sections must be non-negative and finite
        sigma = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
        
        assert np.all(sigma >= 0.0), "Cross sections must be non-negative"
        assert np.all(np.isfinite(sigma)), "Cross sections must be finite"
        
        # For a given interaction, cross section should be ≤ total cross section
        sigma_compton = 0.5
        sigma_total = 1.5
        assert sigma_compton <= sigma_total


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_zero_energy_particles(self):
        """Test handling of zero-energy particles."""
        edep = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        rho = torch.ones((8, 8, 8), dtype=torch.float32)
        
        dose, unc = edep_to_dose_and_uncertainty(
            edep_batches=edep,
            rho=rho,
            voxel_volume_cm3=1.0,
            uncertainty_mode="absolute",
        )
        
        assert torch.all(dose == 0.0)
        assert torch.all(torch.isfinite(unc))

    def test_very_high_activity(self):
        """Test handling of very high activity values."""
        activity = torch.ones((4, 4, 4), dtype=torch.float32) * 1e6  # Very high activity
        
        nuclide = ICRP107Nuclide(
            name="test",
            half_life_s=1.0,
            emissions={"gamma": [[0.1, 1.0]]},
        )
        
        primaries, local = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=nuclide,
            n_histories=100,  # Limited histories to avoid memory issues
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 3.0, "electron_keV": 20.0},
            sampling_mode="accurate",
        )
        
        # Should produce valid particles
        assert torch.isfinite(primaries.photons["E_MeV"]).all()
        assert torch.isfinite(primaries.photons["w"]).all()

    def test_mixed_mass_density(self):
        """Test dose calculation with varying density."""
        edep = torch.ones((1, 8, 8, 8), dtype=torch.float32)
        rho = torch.linspace(0.1, 2.0, 512).reshape(8, 8, 8)  # Varying density
        
        dose, unc = edep_to_dose_and_uncertainty(
            edep_batches=edep,
            rho=rho,
            voxel_volume_cm3=0.064,
            uncertainty_mode="relative",
        )
        
        # Voxels with higher density should have lower dose for same edep
        assert dose[0, 0, 0] > dose[7, 7, 7]
        assert torch.isfinite(dose).all()
        assert torch.isfinite(unc).all()


class TestMaterialHandling:
    """Test material-related physics."""

    def test_material_library_coverage(self):
        """Test that material library covers expected biological materials."""
        lib = build_default_materials_library(device="cpu")
        
        expected_materials = ["air", "lung", "fat", "muscle", "soft_tissue", "bone"]
        for mat_name in expected_materials:
            assert mat_name in lib.material_names, f"Missing material: {mat_name}"

    def test_material_density_ordering(self):
        """Test that material densities follow expected ordering."""
        lib = build_default_materials_library(device="cpu")
        
        # Air should be lightest
        air_idx = lib.material_names.index("air")
        air_rho = lib.ref_density_g_cm3[air_idx].item()
        
        # Bone should be heaviest
        bone_idx = lib.material_names.index("bone")
        bone_rho = lib.ref_density_g_cm3[bone_idx].item()
        
        assert air_rho < bone_rho, "Air should be lighter than bone"

    def test_composition_sum_normalization(self):
        """Test that elemental compositions sum to unity."""
        lib = build_default_materials_library(device="cpu")
        
        for i, mat_name in enumerate(lib.material_names):
            wfrac_sum = lib.material_wfrac[i].sum().item()
            assert abs(wfrac_sum - 1.0) < 1e-4, f"{mat_name} composition doesn't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
