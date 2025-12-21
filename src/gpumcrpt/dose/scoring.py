from __future__ import annotations

import torch


_MEV_TO_J = 1.602176634e-13


def edep_to_dose_and_uncertainty(
    *,
    edep_batches: torch.Tensor,   # [B,Z,Y,X] in MeV
    rho: torch.Tensor,           # [Z,Y,X] in g/cm^3
    voxel_volume_cm3: float,
    uncertainty_mode: str = "relative",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert deposited energy to absorbed dose (Gy) and batch-based uncertainty.

    This treats each batch as an independent estimate; uncertainty is computed across batches.

    uncertainty_mode:
      - "relative": returns relative standard error (unitless)
      - "absolute": returns absolute standard error in Gy
    """
    if edep_batches.ndim != 4:
        raise ValueError(f"edep_batches must be [B,Z,Y,X], got shape={tuple(edep_batches.shape)}")

    B = int(edep_batches.shape[0])
    edep_mean_MeV = edep_batches.mean(dim=0)

    # Convert to dose
    mass_kg = (rho.to(edep_mean_MeV.dtype) * float(voxel_volume_cm3)) / 1000.0
    mass_kg = torch.clamp(mass_kg, min=5e-5)

    dose_Gy = (edep_mean_MeV * _MEV_TO_J) / mass_kg

    if B <= 1:
        if uncertainty_mode == "relative":
            unc = torch.zeros_like(dose_Gy)
        elif uncertainty_mode == "absolute":
            unc = torch.zeros_like(dose_Gy)
        else:
            raise ValueError(f"Unknown uncertainty_mode={uncertainty_mode}")
        return dose_Gy, unc

    # Sample variance across batches -> standard error of mean
    edep_var = edep_batches.var(dim=0, unbiased=True)
    edep_sem_MeV = torch.sqrt(edep_var / float(B))
    dose_sem_Gy = (edep_sem_MeV * _MEV_TO_J) / mass_kg

    if uncertainty_mode == "relative":
        unc = dose_sem_Gy / torch.clamp(dose_Gy.abs(), min=1e-12)
    elif uncertainty_mode == "absolute":
        unc = dose_sem_Gy
    else:
        raise ValueError(f"Unknown uncertainty_mode={uncertainty_mode}")

    return dose_Gy, unc
