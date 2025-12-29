#!/usr/bin/env python3
"""Generate pre-computed physics tables for GPUMCRPTDosimetry.

This script intentionally avoids relying on ambiguous third-party packages named
"xcom" or "star" (which can resolve to unrelated PyPI projects).

Instead, it generates deterministic, self-contained tables using standard
physics models suitable for the current transport engines:

- Photon cross-sections (cm^2/g):
  - Compton: Klein–Nishina total cross section per electron × electrons/g
  - Photoelectric / Rayleigh / Pair: simple parametric models (research prototype)
- Compton angular sampling: inverse-CDF of Klein–Nishina dσ/dcosθ
- Electron CSDA range and stopping power: simple empirical CSDA range model
- Brems/delta samplers: analytic inverse-CDF in energy fraction (Efrac)

Schema matches `gpumcrpt.physics_tables.tables.load_physics_tables_h5` (schema_version=1.0).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def _project_src_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "src"


sys.path.insert(0, str(_project_src_dir()))

from gpumcrpt.materials.hu_materials import (
    build_default_materials_library,
    build_materials_library_from_config,
    compute_material_effective_atom_Z,
)
from gpumcrpt.materials.materials_registry import get_default_registry


AVOGADRO = 6.02214076e23
RE_CM = 2.8179403227e-13
ME_MEV = 0.51099895


def _atomic_mass_g_mol(*, symbol: str | None, Z: int) -> float:
    mapping = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "P": 30.974,
        "Ca": 40.078,
        "Ar": 39.948,
    }
    if symbol is not None and symbol in mapping:
        return float(mapping[symbol])
    if Z <= 0:
        raise ValueError(f"Invalid atomic number Z={Z}")
    if Z == 1:
        return 1.008
    if Z == 6:
        return 12.011
    if Z == 7:
        return 14.007
    if Z == 8:
        return 15.999
    return 2.0 * float(Z)


def _electrons_per_gram(material_wfrac: np.ndarray, element_Z: np.ndarray, element_symbol: list[str]) -> np.ndarray:
    Z_i = element_Z.astype(np.int32)
    A = np.asarray(
        [_atomic_mass_g_mol(symbol=sym, Z=int(z)) for sym, z in zip(element_symbol, Z_i, strict=False)],
        dtype=np.float64,
    )
    Z = Z_i.astype(np.float64)
    z_over_a = Z / A
    return AVOGADRO * (material_wfrac.astype(np.float64) * z_over_a[None, :]).sum(axis=1)


def _sigma_kn_total_per_electron_cm2(E_MeV: np.ndarray) -> np.ndarray:
    E = np.asarray(E_MeV, dtype=np.float64)
    alpha = np.maximum(E / ME_MEV, 1e-12)
    t1 = (1.0 + alpha) / (alpha**2)
    t2 = (2.0 * (1.0 + alpha) / (1.0 + 2.0 * alpha)) - (np.log1p(2.0 * alpha) / alpha)
    t3 = np.log1p(2.0 * alpha) / (2.0 * alpha)
    t4 = (1.0 + 3.0 * alpha) / ((1.0 + 2.0 * alpha) ** 2)
    sigma = 2.0 * math.pi * (RE_CM**2) * (t1 * t2 + t3 - t4)
    return sigma.astype(np.float64)


def _kn_dsigma_dcos_unnorm(E_MeV: np.ndarray, cos_t: np.ndarray) -> np.ndarray:
    E = np.asarray(E_MeV, dtype=np.float64)[:, None]
    mu = np.asarray(cos_t, dtype=np.float64)[None, :]
    alpha = np.maximum(E / ME_MEV, 1e-12)
    er = 1.0 / (1.0 + alpha * (1.0 - mu))
    sin2 = np.maximum(0.0, 1.0 - mu * mu)
    dsdo = 0.5 * (RE_CM**2) * (er**2) * (er + (1.0 / np.maximum(er, 1e-30)) - sin2)
    return (2.0 * math.pi) * np.maximum(dsdo, 0.0)


def _inv_cdf_from_pdf(pdf: np.ndarray, x: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    cdf = np.cumsum((pdf[:, 1:] + pdf[:, :-1]) * 0.5 * dx[None, :], axis=1)
    cdf = np.concatenate([np.zeros((pdf.shape[0], 1), dtype=np.float64), cdf], axis=1)
    norm = np.maximum(cdf[:, -1:], 1e-30)
    cdf = cdf / norm

    inv = np.empty((pdf.shape[0], u_grid.size), dtype=np.float64)
    for i in range(pdf.shape[0]):
        inv[i, :] = np.interp(u_grid, cdf[i, :], x)
    return inv


def build_kn_compton_inv_cdf(E_centers_MeV: np.ndarray, K: int = 256, n_mu: int = 4096) -> np.ndarray:
    mu = np.linspace(-1.0, 1.0, n_mu, dtype=np.float64)
    pdf = _kn_dsigma_dcos_unnorm(E_centers_MeV, mu)
    u_grid = np.linspace(0.0, 1.0, K, dtype=np.float64)
    inv = _inv_cdf_from_pdf(pdf=pdf, x=mu, u_grid=u_grid)
    return inv.astype(np.float32)


def _csda_range_g_cm2(E_MeV: np.ndarray) -> np.ndarray:
    E = np.asarray(E_MeV, dtype=np.float64)
    E = np.maximum(E, 1e-4)
    r = 0.412 * (E**1.265) - 0.0954 * np.log(E)
    return np.maximum(r, 1e-6)


def _make_inv_cdf_brems(ECOUNT: int, K: int = 256, xmin: float = 1e-4, xmax: float = 0.3) -> np.ndarray:
    u = np.linspace(0.0, 1.0, K, dtype=np.float64)
    inv_1d = xmin * ((xmax / xmin) ** u)
    return np.tile(inv_1d[None, :], (ECOUNT, 1)).astype(np.float32)


def _make_inv_cdf_delta(ECOUNT: int, K: int = 256, xmin: float = 1e-4, xmax: float = 0.5) -> np.ndarray:
    u = np.linspace(0.0, 1.0, K, dtype=np.float64)
    inv_1d = 1.0 / (1.0 / xmin - u * (1.0 / xmin - 1.0 / xmax))
    return np.tile(inv_1d[None, :], (ECOUNT, 1)).astype(np.float32)


def main() -> None:
    registry = get_default_registry()
    available_material_libraries = ["default_materials", *registry.list_tables()]

    parser = argparse.ArgumentParser(description="Generate pre-computed physics tables for GPUMCRPTDosimetry.")
    parser.add_argument(
        "--material_library",
        required=True,
        choices=available_material_libraries,
        help="Name of the material library to use.",
    )
    parser.add_argument(
        "--physics_mode",
        required=True,
        choices=["local_deposit", "photon_electron_local", "photon_electron_condensed"],
        help="The physics mode for which to generate the tables.",
    )
    parser.add_argument(
        "--output_dir",
        default="src/gpumcrpt/physics_tables/precomputed_tables",
        help="Directory to save the HDF5 files.",
    )
    parser.add_argument("--num_energy_bins", type=int, default=500, help="Number of energy bins.")
    parser.add_argument("--min_energy_mev", type=float, default=0.01, help="Minimum energy in MeV.")
    parser.add_argument("--max_energy_mev", type=float, default=10.0, help="Maximum energy in MeV.")
    parser.add_argument("--compton_inv_cdf_bins", type=int, default=256, help="# inverse-CDF samples per energy bin.")
    args = parser.parse_args()

    print(f"Generating physics tables: material_library={args.material_library!r}, physics_mode={args.physics_mode!r}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.material_library}-{args.physics_mode}.h5"

    if args.material_library == "default_materials":
        materials_library = build_default_materials_library(device="cpu")
    else:
        mat_table_config = registry.get_table(args.material_library)
        materials_library = build_materials_library_from_config(
            cfg={"material_library": mat_table_config.material_library},
            device="cpu",
        )
    mat_names = list(materials_library.material_names)
    num_materials = len(mat_names)

    e_edges = np.logspace(
        np.log10(float(args.min_energy_mev)),
        np.log10(float(args.max_energy_mev)),
        int(args.num_energy_bins) + 1,
        dtype=np.float32,
    )
    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:]).astype(np.float32)

    element_Z = materials_library.element_Z.detach().cpu().numpy().astype(np.int32)
    element_symbol = list(materials_library.element_symbol)
    wfrac = materials_library.material_wfrac.detach().cpu().numpy().astype(np.float32)
    if hasattr(materials_library, "ref_density_g_cm3"):
        rho = materials_library.ref_density_g_cm3.detach().cpu().numpy().astype(np.float32)
    else:
        rho = materials_library.density_g_cm3.detach().cpu().numpy().astype(np.float32)

    Z_eff = compute_material_effective_atom_Z(materials_library).detach().cpu().numpy().astype(np.float32)

    shape = (num_materials, e_centers.size)
    sigma_photo = np.zeros(shape, dtype=np.float32)
    sigma_compton = np.zeros(shape, dtype=np.float32)
    sigma_rayleigh = np.zeros(shape, dtype=np.float32)
    sigma_pair = np.zeros(shape, dtype=np.float32)
    S_restricted = np.zeros(shape, dtype=np.float32)
    range_csda_cm = np.zeros(shape, dtype=np.float32)

    P_brem_per_cm = None
    P_delta_per_cm = None

    if args.physics_mode in {"photon_electron_local", "photon_electron_condensed"}:
        ne_g = _electrons_per_gram(wfrac, element_Z, element_symbol).astype(np.float64)
        sigma_kn = _sigma_kn_total_per_electron_cm2(e_centers.astype(np.float64))
        sigma_compton[:, :] = (ne_g[:, None] * sigma_kn[None, :]).astype(np.float32)

        Z_water = 7.42
        E_ref = 0.1
        sigma_pe_water = 0.03
        E = e_centers.astype(np.float64)
        for m in range(num_materials):
            z = float(Z_eff[m])
            z4 = (z / Z_water) ** 4
            z2 = (z / Z_water) ** 2
            sigma_photo[m, :] = (sigma_pe_water * z4 * (E_ref / np.maximum(E, 1e-6)) ** 3).astype(np.float32)
            sigma_rayleigh[m, :] = (0.02 * z2 * sigma_compton[m, :] * (E_ref / np.maximum(E, 1e-6)) ** 2).astype(np.float32)
            thr = 1.022
            sigma_pair[m, :] = (1e-3 * z2 * np.maximum(0.0, np.log(np.maximum(E, thr) / thr))).astype(np.float32)

    sigma_total = sigma_photo + sigma_compton + sigma_rayleigh + sigma_pair
    sigma_max = np.max(sigma_total, axis=0).astype(np.float32)
    p_cum = np.zeros_like(sigma_total, dtype=np.float32)

    if args.physics_mode == "photon_electron_condensed":
        rg = _csda_range_g_cm2(e_centers.astype(np.float64))
        for m in range(num_materials):
            range_csda_cm[m, :] = (rg / float(rho[m])).astype(np.float32)
            S_restricted[m, :] = (e_centers.astype(np.float64) / rg).astype(np.float32)

        E2 = e_centers.astype(np.float32)
        P_brem_per_cm = (0.002 * rho[:, None] * (E2[None, :] / 1.0) * (Z_eff[:, None] / 7.42)).astype(np.float32)
        P_delta_per_cm = (0.01 * rho[:, None] * (E2[None, :] / 1.0)).astype(np.float32)

    compton_inv_cdf = None
    if args.physics_mode in {"photon_electron_local", "photon_electron_condensed"}:
        print("Building Compton inverse-CDF (Klein–Nishina)...")
        compton_inv_cdf = build_kn_compton_inv_cdf(e_centers, K=int(args.compton_inv_cdf_bins))

    brem_inv = None
    delta_inv = None
    if args.physics_mode == "photon_electron_condensed":
        brem_inv = _make_inv_cdf_brems(ECOUNT=e_centers.size)
        delta_inv = _make_inv_cdf_delta(ECOUNT=e_centers.size)

    with h5py.File(output_path, "w") as f:
        f.attrs["material_library"] = args.material_library
        f.attrs["physics_mode"] = args.physics_mode

        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "1.0"

        mg = meta.create_group("materials")
        mg.create_dataset("names", data=[m.encode("utf-8") for m in mat_names])
        mg.create_dataset("ref_density_g_cm3", data=rho)

        eg = f.create_group("energy")
        eg.create_dataset("edges_MeV", data=e_edges)
        eg.create_dataset("centers_MeV", data=e_centers)

        if args.physics_mode in {"photon_electron_local", "photon_electron_condensed"}:
            ph = f.create_group("photons")
            ph.create_dataset("sigma_photo", data=sigma_photo)
            ph.create_dataset("sigma_compton", data=sigma_compton)
            ph.create_dataset("sigma_rayleigh", data=sigma_rayleigh)
            ph.create_dataset("sigma_pair", data=sigma_pair)
            ph.create_dataset("sigma_total", data=sigma_total)
            ph.create_dataset("p_cum", data=p_cum)
            ph.create_dataset("sigma_max", data=sigma_max)

        if args.physics_mode == "photon_electron_condensed":
            el = f.create_group("electrons")
            el.create_dataset("S_restricted", data=S_restricted)
            el.create_dataset("range_csda_cm", data=range_csda_cm)
            if P_brem_per_cm is not None:
                el.create_dataset("P_brem_per_cm", data=P_brem_per_cm)
            if P_delta_per_cm is not None:
                el.create_dataset("P_delta_per_cm", data=P_delta_per_cm)

        if compton_inv_cdf is not None or brem_inv is not None or delta_inv is not None:
            sg = f.create_group("samplers")

            if compton_inv_cdf is not None:
                pg = sg.create_group("photon")
                cg = pg.create_group("compton")
                cg.attrs["convention"] = "cos_theta"
                cg.create_dataset("inv_cdf", data=compton_inv_cdf)

            if brem_inv is not None or delta_inv is not None:
                eg2 = sg.create_group("electron")
                if brem_inv is not None:
                    bg = eg2.create_group("brems")
                    bg.create_dataset("inv_cdf_Efrac", data=brem_inv)
                if delta_inv is not None:
                    dg = eg2.create_group("delta")
                    dg.create_dataset("inv_cdf_Efrac", data=delta_inv)

    print(f"Successfully generated physics tables at: {output_path}")


if __name__ == "__main__":
    main()
