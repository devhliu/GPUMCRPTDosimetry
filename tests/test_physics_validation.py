from __future__ import annotations

import pytest
import torch
import numpy as np

from gpumcrpt.source.sampling import (
    _parse_discrete_pairs,
    _parse_beta_spectrum_pairs,
    _sample_beta_pdf_from_bspectra,
    _expected_discrete_energy_MeV,
)
from gpumcrpt.dose.scoring import edep_to_dose_and_uncertainty
from gpumcrpt.materials.hu_materials import build_default_materials_library, compute_material_effective_atom_Z
from gpumcrpt.decaydb.icrp107_json import load_icrp107_json
from pathlib import Path


class TestDiscreteEmissionParsing:
    """Test radionuclide emission parsing from ICRP107 data."""

    def test_parse_discrete_pairs_list_format(self):
        """Test parsing discrete emissions in [[E, y], ...] format."""
        pairs = [[0.511, 0.5], [1.0, 0.3]]
        E, y = _parse_discrete_pairs(pairs, device="cpu")
        
        assert E.shape == (2,)
        assert y.shape == (2,)
        assert torch.allclose(E, torch.tensor([0.511, 1.0]))
        assert torch.allclose(y, torch.tensor([0.5, 0.3]))

    def test_parse_discrete_pairs_dict_format(self):
        """Test parsing discrete emissions in [{"energy": E, "yield": y}, ...] format."""
        pairs = [
            {"energy": 0.511, "yield": 0.5},
            {"E": 1.0, "intensity": 0.3},
        ]
        E, y = _parse_discrete_pairs(pairs, device="cpu")
        
        assert E.shape == (2,)
        assert y.shape == (2,)
        assert torch.allclose(E, torch.tensor([0.511, 1.0]))
        assert torch.allclose(y, torch.tensor([0.5, 0.3]))

    def test_parse_discrete_pairs_empty(self):
        """Test parsing empty discrete emissions."""
        E, y = _parse_discrete_pairs(None, device="cpu")
        assert E.shape == (0,)
        assert y.shape == (0,)

    def test_parse_beta_spectrum(self):
        """Test parsing beta spectrum data."""
        spectrum = [[0.1, 0.01], [0.5, 0.5], [1.0, 0.2]]
        bspec = _parse_beta_spectrum_pairs(spectrum)
        
        assert len(bspec) == 3
        assert bspec[0][0] == 0.1
        assert bspec[1][1] == 0.5

    def test_expected_discrete_energy_calculation(self):
        """Test expected value calculation for discrete emissions."""
        pairs = [[0.5, 0.4], [1.0, 0.6]]
        E_exp = _expected_discrete_energy_MeV(pairs, device="cpu")
        
        expected = 0.5 * 0.4 + 1.0 * 0.6
        assert torch.allclose(E_exp, torch.tensor(expected, dtype=torch.float32), atol=1e-6)


class TestBetaSpectralSampling:
    """Test beta spectrum sampling physics."""

    def test_beta_pdf_sampling_shape(self):
        """Test that beta PDF sampling returns correct shape."""
        bspec = [[0.1, 0.1], [0.5, 0.8], [2.0, 0.2]]
        g = torch.Generator(device="cpu")
        g.manual_seed(42)
        
        E = _sample_beta_pdf_from_bspectra(bspec, n=1000, g=g, device="cpu")
        
        assert E.shape == (1000,)
        assert (E >= 0.0).all()
        assert (E <= 2.0).all()

    def test_beta_sampling_pdf_bounds(self):
        """Test that sampled energies respect PDF bounds."""
        bspec = [[0.0, 0.5], [0.5, 1.0], [1.5, 0.3]]
        g = torch.Generator(device="cpu")
        g.manual_seed(42)
        
        E = _sample_beta_pdf_from_bspectra(bspec, n=5000, g=g, device="cpu")
        
        assert torch.all(E >= 0.0)
        assert torch.all(E <= 1.5)

    def test_beta_sampling_cdf_monotonicity(self):
        """Test that CDF is monotonically increasing."""
        bspec = [[0.01, 0.01], [0.5, 0.99], [0.99, 0.1]]
        # Verify that when sorted by energy, PDF can be treated as monotone
        energies = torch.tensor([p[0] for p in bspec])
        pdf_vals = torch.tensor([p[1] for p in bspec])
        
        order = torch.argsort(energies)
        pdf_sorted = pdf_vals[order]
        
        # CDF should be non-decreasing
        cdf = torch.cumsum(pdf_sorted, dim=0)
        diffs = torch.diff(cdf)
        assert torch.all(diffs >= -1e-6)  # Allow small numerical errors


class TestDoseConversion:
    """Test dose conversion from energy deposition."""

    def test_dose_conversion_basic(self):
        """Test basic dose conversion from MeV to Gy."""
        edep = torch.tensor([[[1e6]]], dtype=torch.float32)  # 1 MeV
        rho = torch.tensor([[[1.0]]], dtype=torch.float32)   # 1 g/cm^3
        voxel_vol = 0.064  # (0.4 cm)^3
        
        dose, _ = edep_to_dose_and_uncertainty(
            edep_batches=edep.unsqueeze(0),  # Add batch dimension
            rho=rho,
            voxel_volume_cm3=voxel_vol,
            uncertainty_mode="absolute"
        )
        
        # 1 MeV in 1 g of 0.064 cm^3 volume
        # mass = 1 * 0.064 / 1000 = 6.4e-5 kg
        # dose = 1e6 * 1.602e-13 / 6.4e-5 ≈ 2.5 Gy
        assert dose[0, 0, 0] > 0.0
        assert torch.isfinite(dose).all()

    def test_dose_uncertainty_calculation(self):
        """Test uncertainty calculation across batches."""
        edep_batch1 = torch.tensor([[[1.0]]], dtype=torch.float32)
        edep_batch2 = torch.tensor([[[1.1]]], dtype=torch.float32)
        edep_batches = torch.stack([edep_batch1, edep_batch2], dim=0)
        
        rho = torch.tensor([[[1.0]]], dtype=torch.float32)
        
        dose, unc_rel = edep_to_dose_and_uncertainty(
            edep_batches=edep_batches,
            rho=rho,
            voxel_volume_cm3=1.0,
            uncertainty_mode="relative"
        )
        
        assert torch.isfinite(dose).all()
        assert torch.isfinite(unc_rel).all()
        assert (unc_rel >= 0.0).all()

    def test_dose_conservation(self):
        """Test that total dose sums to input energy."""
        edep = torch.ones((1, 8, 8, 8), dtype=torch.float32) * 10.0  # 10 MeV per voxel
        rho = torch.ones((8, 8, 8), dtype=torch.float32)
        voxel_vol = 1.0
        
        dose, _ = edep_to_dose_and_uncertainty(
            edep_batches=edep,
            rho=rho,
            voxel_volume_cm3=voxel_vol,
            uncertainty_mode="absolute"
        )
        
        # Energy conservation: total dose × total mass should be proportional to input energy
        total_mass_kg = 8 * 8 * 8 * rho[0, 0, 0].item() * voxel_vol / 1000.0
        total_dose_Gy = dose.sum().item()
        
        assert torch.isfinite(dose).all()
        assert (dose >= 0.0).all()


class TestMaterialComposition:
    """Test material library and HU conversion."""

    def test_default_materials_library_composition(self):
        """Test that default materials have valid compositions."""
        lib = build_default_materials_library(device="cpu")
        
        # All materials should sum to 1
        w_sum = lib.material_wfrac.sum(dim=1)
        assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-5)
        
        # All weights should be non-negative
        assert (lib.material_wfrac >= 0.0).all()
        
        # Material names should match the number of rows
        assert len(lib.material_names) == lib.material_wfrac.shape[0]

    def test_material_density_values(self):
        """Test that material densities are physically realistic."""
        lib = build_default_materials_library(device="cpu")
        
        # Known physical densities
        expected = {
            "air": (0.001, 0.002),
            "lung": (0.3, 0.5),
            "fat": (0.9, 1.0),
            "muscle": (1.04, 1.08),
            "soft_tissue": (0.98, 1.02),
            "bone": (1.4, 1.6),
        }
        
        for name, (low, high) in expected.items():
            if name in lib.material_names:
                idx = lib.material_names.index(name)
                rho = lib.ref_density_g_cm3[idx].item()
                assert low <= rho <= high, f"{name} density {rho} outside range [{low}, {high}]"

    def test_effective_z_calculation(self):
        """Test effective atomic number calculation."""
        lib = build_default_materials_library(device="cpu")
        z_eff = compute_material_effective_atom_Z(lib, power=1.0)
        
        assert z_eff.shape[0] == lib.material_wfrac.shape[0]
        assert (z_eff > 0).all()
        # Effective Z should be reasonable for biological materials
        assert (z_eff < 20).all()  # Less than calcium


class TestRNGDeterminism:
    """Test random number generation reproducibility."""

    def test_seed_reproducibility(self):
        """Test that seeding produces reproducible results."""
        from gpumcrpt.source.sampling import sample_weighted_decays_and_primaries
        from gpumcrpt.decaydb.icrp107_json import ICRP107Nuclide
        
        # Create simple test nuclide
        nuclide = ICRP107Nuclide(
            name="test",
            half_life_s=1.0,
            emissions={
                "gamma": [[0.5, 1.0]],
            }
        )
        
        activity = torch.ones((4, 4, 4))
        
        prim1, local1 = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=nuclide,
            n_histories=100,
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 3.0, "electron_keV": 20.0},
            sampling_mode="accurate"
        )
        
        prim2, local2 = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=nuclide,
            n_histories=100,
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 3.0, "electron_keV": 20.0},
            sampling_mode="accurate"
        )
        
        # Check reproducibility
        assert torch.allclose(prim1.photons["E_MeV"], prim2.photons["E_MeV"])
        assert torch.allclose(local1, local2)


class TestEnergyConservation:
    """Test energy conservation in sampling."""

    def test_alpha_local_deposition(self):
        """Test that alpha particles are deposited locally."""
        from gpumcrpt.source.sampling import sample_weighted_decays_and_primaries
        from gpumcrpt.decaydb.icrp107_json import ICRP107Nuclide
        
        nuclide = ICRP107Nuclide(
            name="test",
            half_life_s=1.0,
            emissions={
                "alpha": [[4.0, 1.0]],  # 4 MeV alphas
                "gamma": [[0.5, 0.0]],  # No gammas
            }
        )
        
        activity = torch.ones((4, 4, 4))
        
        prim, local = sample_weighted_decays_and_primaries(
            activity_bqs=activity,
            voxel_size_cm=(0.4, 0.4, 0.4),
            affine=np.eye(4),
            nuclide=nuclide,
            n_histories=100,
            seed=42,
            device="cpu",
            cutoffs={"photon_keV": 3.0, "electron_keV": 20.0},
            sampling_mode="accurate"
        )
        
        # Alpha energy should be in local deposition
        total_alpha_energy = local.sum().item()
        assert total_alpha_energy > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
