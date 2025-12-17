from __future__ import annotations

from dataclasses import dataclass
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
    with h5py.File(path, "r") as f:
        schema_version = f["/meta"].attrs.get("schema_version", "NA")
        if isinstance(schema_version, bytes):
            schema_version = schema_version.decode("utf-8")
        if str(schema_version) != "1.0":
            raise ValueError(f"Unsupported schema_version={schema_version}, expected 1.0")

        mat_names = [x.decode("utf-8") for x in np.asarray(f["/meta/materials/names"])]
        ref_rho = np.asarray(f["/meta/materials/ref_density_g_cm3"], dtype=np.float32)

        edges = np.asarray(f["/energy/edges_MeV"], dtype=np.float32)
        centers = np.asarray(
            f.get("/energy/centers_MeV", (edges[:-1] + edges[1:]) * 0.5), dtype=np.float32
        )

        ph = "/photons"
        sigma_photo = np.asarray(f[f"{ph}/sigma_photo"], dtype=np.float32)
        sigma_compton = np.asarray(f[f"{ph}/sigma_compton"], dtype=np.float32)
        sigma_rayleigh = np.asarray(f[f"{ph}/sigma_rayleigh"], dtype=np.float32)
        sigma_pair = np.asarray(f[f"{ph}/sigma_pair"], dtype=np.float32)
        sigma_total = np.asarray(f[f"{ph}/sigma_total"], dtype=np.float32)
        p_cum = np.asarray(f[f"{ph}/p_cum"], dtype=np.float32)
        sigma_max = np.asarray(f[f"{ph}/sigma_max"], dtype=np.float32)

        el = "/electrons"
        S_restricted = np.asarray(f[f"{el}/S_restricted"], dtype=np.float32)
        range_csda_cm = np.asarray(f[f"{el}/range_csda_cm"], dtype=np.float32)

        P_brem = np.asarray(f[f"{el}/P_brem_per_cm"], dtype=np.float32) if f.get(f"{el}/P_brem_per_cm", None) is not None else None
        P_delta = np.asarray(f[f"{el}/P_delta_per_cm"], dtype=np.float32) if f.get(f"{el}/P_delta_per_cm", None) is not None else None

        compton_inv_cdf = None
        compton_convention = "cos_theta"
        if f.get("/samplers/photon/compton/inv_cdf", None) is not None:
            compton_inv_cdf = np.asarray(f["/samplers/photon/compton/inv_cdf"], dtype=np.float32)
            compton_convention = f["/samplers/photon/compton"].attrs.get("convention", "cos_theta")
            if isinstance(compton_convention, bytes):
                compton_convention = compton_convention.decode("utf-8")
            if str(compton_convention) != "cos_theta":
                raise ValueError(f"Compton inv_cdf convention must be 'cos_theta', got {compton_convention}")

        brem_inv = np.asarray(f["/samplers/electron/brems/inv_cdf_Efrac"], dtype=np.float32) if f.get("/samplers/electron/brems/inv_cdf_Efrac", None) is not None else None
        delta_inv = np.asarray(f["/samplers/electron/delta/inv_cdf_Efrac"], dtype=np.float32) if f.get("/samplers/electron/delta/inv_cdf_Efrac", None) is not None else None

        # ---- Validate optional inverse-CDF sampler tables (when present)
        ECOUNT = int(centers.shape[0])

        def _check_inv_cdf_monotone(name: str, inv: np.ndarray, *, lo: float, hi: float) -> None:
            if inv.ndim != 2 or int(inv.shape[0]) != ECOUNT or int(inv.shape[1]) < 2:
                raise ValueError(f"{name} must have shape [ECOUNT,K>=2]; got {inv.shape} (ECOUNT={ECOUNT})")
            if not np.isfinite(inv).all():
                raise ValueError(f"{name} contains non-finite values")
            # inv-cdf must be nondecreasing in u.
            if (np.diff(inv, axis=1) < -1e-6).any():
                raise ValueError(f"{name} is not monotone nondecreasing along K")
            if inv.min() < lo - 1e-3 or inv.max() > hi + 1e-3:
                raise ValueError(f"{name} out of expected range [{lo},{hi}]: min={inv.min()}, max={inv.max()}")

        if compton_inv_cdf is not None:
            _check_inv_cdf_monotone("/samplers/photon/compton/inv_cdf", compton_inv_cdf, lo=-1.0, hi=1.0)

        if brem_inv is not None:
            _check_inv_cdf_monotone("/samplers/electron/brems/inv_cdf_Efrac", brem_inv, lo=0.0, hi=1.0)

        if delta_inv is not None:
            _check_inv_cdf_monotone("/samplers/electron/delta/inv_cdf_Efrac", delta_inv, lo=0.0, hi=1.0)

    def _to_torch(x, dev):
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
        P_brem_per_cm=_to_torch(P_brem, device) if P_brem is not None else None,
        P_delta_per_cm=_to_torch(P_delta, device) if P_delta is not None else None,
        compton_inv_cdf=_to_torch(compton_inv_cdf, device) if compton_inv_cdf is not None else None,
        compton_convention=str(compton_convention),
        brem_inv_cdf_Efrac=_to_torch(brem_inv, device) if brem_inv is not None else None,
        delta_inv_cdf_Efrac=_to_torch(delta_inv, device) if delta_inv is not None else None,
    )

    return tables