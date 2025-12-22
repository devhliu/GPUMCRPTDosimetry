from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch


@dataclass
class PhysicsTables:
    e_edges_MeV: torch.Tensor
    e_centers_MeV: torch.Tensor

    material_names: list[str]
    ref_density_g_cm3: torch.Tensor

    sigma_photo: torch.Tensor
    sigma_compton: torch.Tensor
    sigma_rayleigh: torch.Tensor
    sigma_pair: torch.Tensor
    sigma_total: torch.Tensor
    p_cum: torch.Tensor
    sigma_max: torch.Tensor

    S_restricted: torch.Tensor
    range_csda_cm: torch.Tensor
    P_brem_per_cm: Optional[torch.Tensor] = None
    P_delta_per_cm: Optional[torch.Tensor] = None

    # samplers
    compton_inv_cdf: Optional[torch.Tensor] = None  # [E,K] -> cos(theta)
    compton_convention: str = "cos_theta"           # enforce cos(theta)
    brem_inv_cdf_Efrac: Optional[torch.Tensor] = None
    delta_inv_cdf_Efrac: Optional[torch.Tensor] = None

    def as_cpu(self) -> "PhysicsTables":
        return self.to("cpu")

    def to(self, device: str) -> "PhysicsTables":
        def mv(x):
            return x.detach().to(device) if isinstance(x, torch.Tensor) else x

        return PhysicsTables(
            e_edges_MeV=mv(self.e_edges_MeV),
            e_centers_MeV=mv(self.e_centers_MeV),
            material_names=list(self.material_names),
            ref_density_g_cm3=mv(self.ref_density_g_cm3),
            sigma_photo=mv(self.sigma_photo),
            sigma_compton=mv(self.sigma_compton),
            sigma_rayleigh=mv(self.sigma_rayleigh),
            sigma_pair=mv(self.sigma_pair),
            sigma_total=mv(self.sigma_total),
            p_cum=mv(self.p_cum),
            sigma_max=mv(self.sigma_max),
            S_restricted=mv(self.S_restricted),
            range_csda_cm=mv(self.range_csda_cm),
            P_brem_per_cm=mv(self.P_brem_per_cm) if self.P_brem_per_cm is not None else None,
            P_delta_per_cm=mv(self.P_delta_per_cm) if self.P_delta_per_cm is not None else None,
            compton_inv_cdf=mv(self.compton_inv_cdf) if self.compton_inv_cdf is not None else None,
            compton_convention=self.compton_convention,
            brem_inv_cdf_Efrac=mv(self.brem_inv_cdf_Efrac) if self.brem_inv_cdf_Efrac is not None else None,
            delta_inv_cdf_Efrac=mv(self.delta_inv_cdf_Efrac) if self.delta_inv_cdf_Efrac is not None else None,
        )


def load_physics_tables_h5(path: str, device: str = "cuda") -> PhysicsTables:
    """Load physics tables from H5 file."""
    with h5py.File(path, "r") as f:
        schema_version = f["/meta"].attrs.get("schema_version", "NA")
        if isinstance(schema_version, bytes):
            schema_version = schema_version.decode("utf-8")
        if str(schema_version) != "1.0":
            raise ValueError(f"Unsupported schema_version={schema_version}, expected 1.0")

        mat_names = [x.decode("utf-8") for x in np.asarray(f["/meta/materials/names"])]
        ref_rho = np.asarray(f["/meta/materials/ref_density_g_cm3"], dtype=np.float32)

        edges = np.asarray(f["/energy/edges_MeV"], dtype=np.float32)
        centers = np.asarray(f["/energy/centers_MeV"], dtype=np.float32)

        physics_mode = f.attrs.get('physics_mode', 'unknown')
        
        # Initialize with zeros, then fill based on physics_mode
        dummy_shape = (len(mat_names), len(centers))
        
        sigma_photo = np.zeros(dummy_shape, dtype=np.float32)
        sigma_compton = np.zeros(dummy_shape, dtype=np.float32)
        sigma_rayleigh = np.zeros(dummy_shape, dtype=np.float32)
        sigma_pair = np.zeros(dummy_shape, dtype=np.float32)
        sigma_total = np.zeros(dummy_shape, dtype=np.float32)
        p_cum = np.zeros(dummy_shape, dtype=np.float32)
        sigma_max = np.zeros(len(centers), dtype=np.float32)

        S_restricted = np.zeros(dummy_shape, dtype=np.float32)
        range_csda_cm = np.zeros(dummy_shape, dtype=np.float32)
        P_brem = None
        P_delta = None
        
        compton_inv_cdf = None
        brem_inv = None
        delta_inv = None

        if physics_mode in ['photon_only', 'photon_em_condensedhistory']:
            ph = "/photons"
            sigma_photo = np.asarray(f[f"{ph}/sigma_photo"], dtype=np.float32)
            sigma_compton = np.asarray(f[f"{ph}/sigma_compton"], dtype=np.float32)
            sigma_rayleigh = np.asarray(f[f"{ph}/sigma_rayleigh"], dtype=np.float32)
            sigma_pair = np.asarray(f[f"{ph}/sigma_pair"], dtype=np.float32)
            sigma_total = np.asarray(f[f"{ph}/sigma_total"], dtype=np.float32)
            p_cum = np.asarray(f[f"{ph}/p_cum"], dtype=np.float32)
            sigma_max = np.asarray(f[f"{ph}/sigma_max"], dtype=np.float32)

        if physics_mode == 'photon_em_condensedhistory':
            el = "/electrons"
            S_restricted = np.asarray(f[f"{el}/S_restricted"], dtype=np.float32)
            range_csda_cm = np.asarray(f[f"{el}/range_csda_cm"], dtype=np.float32)
            if f"{el}/P_brem_per_cm" in f:
                P_brem = np.asarray(f[f"{el}/P_brem_per_cm"], dtype=np.float32)
            if f"{el}/P_delta_per_cm" in f:
                P_delta = np.asarray(f[f"{el}/P_delta_per_cm"], dtype=np.float32)

        if "/samplers/photon/compton/inv_cdf" in f:
            compton_inv_cdf = np.asarray(f["/samplers/photon/compton/inv_cdf"], dtype=np.float32)
            compton_convention = f["/samplers/photon/compton"].attrs.get("convention", "cos_theta")
            if isinstance(compton_convention, bytes):
                compton_convention = compton_convention.decode("utf-8")
        else:
            compton_convention = "cos_theta"

    def _to_torch(x, dev):
        if x is None:
            return None
        return torch.from_numpy(np.asarray(x)).to(device=dev)

    tables = PhysicsTables(
        e_edges_MeV=_to_torch(edges, device),
        e_centers_MeV=_to_torch(centers, device),
        material_names=mat_names,
        ref_density_g_cm3=_to_torch(ref_rho, device),
        sigma_photo=_to_torch(sigma_photo, device),
        sigma_compton=_to_torch(sigma_compton, device),
        sigma_rayleigh=_to_torch(sigma_rayleigh, device),
        sigma_pair=_to_torch(sigma_pair, device),
        sigma_total=_to_torch(sigma_total, device),
        p_cum=_to_torch(p_cum, device),
        sigma_max=_to_torch(sigma_max, device),
        S_restricted=_to_torch(S_restricted, device),
        range_csda_cm=_to_torch(range_csda_cm, device),
        P_brem_per_cm=_to_torch(P_brem, device),
        P_delta_per_cm=_to_torch(P_delta, device),
        compton_inv_cdf=_to_torch(compton_inv_cdf, device),
        compton_convention=str(compton_convention),
        brem_inv_cdf_Efrac=_to_torch(brem_inv, device),
        delta_inv_cdf_Efrac=_to_torch(delta_inv, device),
    )

    return tables