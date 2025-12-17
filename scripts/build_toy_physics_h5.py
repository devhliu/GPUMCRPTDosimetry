"""
Build a minimal schema-v1.0 physics tables .h5 for pipeline bring-up and CI.

NOTE: Toy tables only; not clinically accurate.

Updated:
- ensure photons/sigma_max has length E (not E+1)
- provide sigma_total shape [M,E]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    materials = ["air", "lung", "soft", "bone", "muscle", "fat"]
    ref_rho = np.array([0.0012, 0.30, 1.00, 1.85, 1.06, 0.92], dtype=np.float32)
    M = len(materials)

    edges = np.array([0.01, 0.03, 0.06, 0.1, 0.2, 0.5, 1.0, 3.0], dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    E = centers.shape[0]

    base = 0.2 / centers
    sigma_photo = np.stack([base * s for s in [0.005, 0.01, 0.02, 0.06, 0.02, 0.015]], axis=0).astype(np.float32)
    sigma_compton = np.stack([0.02 * np.ones_like(base) * s for s in [0.6, 0.8, 1.0, 1.3, 1.05, 0.9]], axis=0).astype(np.float32)
    sigma_rayleigh = np.zeros((M, E), dtype=np.float32)
    sigma_pair = np.zeros((M, E), dtype=np.float32)

    sigma_total = sigma_photo + sigma_compton + sigma_rayleigh + sigma_pair

    p_photo = sigma_photo / np.maximum(sigma_total, 1e-12)
    p_compton = sigma_compton / np.maximum(sigma_total, 1e-12)

    p_cum = np.zeros((M, E, 4), dtype=np.float32)
    p_cum[..., 0] = p_photo
    p_cum[..., 1] = p_photo + p_compton
    p_cum[..., 2] = p_cum[..., 1]  # rayleigh disabled
    p_cum[..., 3] = p_cum[..., 2]  # pair disabled
    # For disabled channels, final cumulative should still be 1.0
    p_cum[..., 3] = 1.0

    sigma_max = np.max(sigma_total, axis=0).astype(np.float32)  # [E]

    S_restricted = np.stack(
        [1.2 * np.ones(E), 1.5 * np.ones(E), 2.0 * np.ones(E), 3.0 * np.ones(E), 2.2 * np.ones(E), 1.8 * np.ones(E)],
        axis=0,
    ).astype(np.float32)
    range_csda_cm = np.stack(
        [0.25 * centers, 0.18 * centers, 0.12 * centers, 0.08 * centers, 0.11 * centers, 0.14 * centers],
        axis=0,
    ).astype(np.float32)

    # Toy secondary emission rates (per cm). These are not physically accurate.
    # Kept modest to avoid runaway emission in toy runs.
    P_brem_per_cm = (0.15 * np.ones((M, E), dtype=np.float32))
    P_delta_per_cm = (0.08 * np.ones((M, E), dtype=np.float32))

    # Toy inverse-CDF samplers for E-fraction (shape [E, K]).
    # Brems photons: favor lower fractions; delta electrons: allow larger fractions.
    K = 256
    u = np.linspace(0.0, 1.0, K, dtype=np.float32)
    brem_inv_cdf_Efrac = np.clip((u ** 2) * 0.3, 0.0, 0.3).astype(np.float32)
    delta_inv_cdf_Efrac = np.clip((u ** 1.0) * 0.5, 0.0, 0.5).astype(np.float32)
    brem_inv_cdf_Efrac = np.repeat(brem_inv_cdf_Efrac[None, :], E, axis=0)
    delta_inv_cdf_Efrac = np.repeat(delta_inv_cdf_Efrac[None, :], E, axis=0)

    # Optional hard-event (per-cm) rates for Milestone-3 secondary spawning bring-up.
    # Keep values small by default; tests can override these after loading.
    P_brem_per_cm = (0.02 * np.ones((M, E), dtype=np.float32))
    P_delta_per_cm = (0.01 * np.ones((M, E), dtype=np.float32))

    # Simple inverse-CDF samplers for energy fractions (Efrac = E_secondary / E_parent).
    # Shape must be [ECOUNT, K] to match Triton samplers.
    K = 256
    brem_inv_cdf_Efrac = np.tile(np.linspace(0.0, 0.3, K, dtype=np.float32), (E, 1))
    delta_inv_cdf_Efrac = np.tile(np.linspace(0.0, 0.5, K, dtype=np.float32), (E, 1))

    if os.path.exists(args.out):
        os.remove(args.out)

    with h5py.File(args.out, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "1.0"
        meta.attrs["energy_min_MeV"] = float(edges[0])
        meta.attrs["energy_max_MeV"] = float(edges[-1])
        meta.create_dataset("materials/names", data=np.array(materials, dtype="S"))
        meta.create_dataset("materials/count", data=np.array([M], dtype=np.int32))
        meta.create_dataset("materials/ref_density_g_cm3", data=ref_rho)

        eg = f.create_group("energy")
        eg.create_dataset("edges_MeV", data=edges)
        eg.create_dataset("centers_MeV", data=centers)

        ph = f.create_group("photons")
        ph.create_dataset("sigma_photo", data=sigma_photo)
        ph.create_dataset("sigma_compton", data=sigma_compton)
        ph.create_dataset("sigma_rayleigh", data=sigma_rayleigh)
        ph.create_dataset("sigma_pair", data=sigma_pair)
        ph.create_dataset("sigma_total", data=sigma_total)
        ph.create_dataset("p_cum", data=p_cum)
        ph.create_dataset("sigma_max", data=sigma_max)

        el = f.create_group("electrons")
        el.create_dataset("S_restricted", data=S_restricted)
        el.create_dataset("range_csda_cm", data=range_csda_cm)
        samplers = f.create_group("samplers")
        el_s = samplers.create_group("electron")
        brem = el_s.create_group("brems")
        brem.create_dataset("inv_cdf_Efrac", data=brem_inv_cdf_Efrac)
        delt = el_s.create_group("delta")
        delt.create_dataset("inv_cdf_Efrac", data=delta_inv_cdf_Efrac)

    print(f"Wrote toy physics table to: {args.out}")


if __name__ == "__main__":
    main()
