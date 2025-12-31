from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import nibabel as nib
import numpy as np
import torch

from nibabel.processing import resample_from_to
from gpumcrpt.materials.hu_materials import (
    build_materials_from_hu,
    build_materials_library_from_config,
)
from gpumcrpt.materials.materials_manager import get_default_materials_manager
from gpumcrpt.physics_tables.tables import PhysicsTables, load_physics_tables_h5
from gpumcrpt.decaydb import load_icrp107_nuclide
from gpumcrpt.source.sampling import sample_weighted_decays_and_primaries
from gpumcrpt.transport.engine_main import TransportEngine
from gpumcrpt.dose.scoring import edep_to_dose_and_uncertainty


@dataclass
class RunInputs:
    activity_nifti_path: str
    ct_nifti_path: str
    sim_yaml_path: str


def run_dosimetry(
    activity_nifti_path: str,
    ct_nifti_path: str,
    sim_config: dict,
    output_dose_path: str,
    output_unc_path: str,
    device: Optional[str] = None,
) -> float:
    device = device or sim_config.get("device", "cuda")
    start_time = time.time()

    act_img = nib.load(activity_nifti_path)
    ct_img = nib.load(ct_nifti_path)

    if sim_config["io"].get("resample_ct_to_activity", True):
        ct_img = resample_from_to(ct_img, act_img)

    act_data = act_img.get_fdata(dtype=np.float32)
    ct_data = ct_img.get_fdata(dtype=np.float32)
    
    act = torch.from_numpy(act_data).to(device=device, dtype=torch.float32)
    hu = torch.from_numpy(ct_data).to(device=device, dtype=torch.float32)

    materials_cfg = sim_config.get("materials", {})
    materials_manager = get_default_materials_manager()
    
    mat_lib = None
    if materials_cfg.get("material_library", None) is not None:
        mat_lib = build_materials_library_from_config(materials_cfg, device=device)
        mats = build_materials_from_hu(
            hu=hu,
            hu_to_density=sim_config["materials"]["hu_to_density"],
            hu_to_class=sim_config["materials"]["hu_to_class"],
            material_library=mat_lib,
            device=device,
        )
    else:
        table_name = materials_cfg.get("name", None)
        mats = materials_manager.get_materials_volume(
            hu_volume=hu,
            table_name=table_name,
            device=device
        )
        mat_lib = mats.lib

    def _engine_to_physics_mode(engine_name: str) -> str:
        e = str(engine_name).lower()
        if e in {"localdepositonly", "local_deposit", "local-deposit"}:
            return "local_deposit"
        if e == "photon_electron_local":
            return "photon_electron_local"
        if e == "photon_electron_condensed":
            return "photon_electron_condensed"
        if e in {"em_energybucketed", "photon-em-energybucketed", "photon_em_energybucketed", "energy_bucketed", "energybucketed"}:
            return "photon_em_energybucketed"
        return e

    # Load physics tables (or create dummy tables for local_deposit engine)
    triton_engine = sim_config.get("monte_carlo", {}).get("triton", {}).get("engine", "local_deposit")
    physics_mode = _engine_to_physics_mode(triton_engine)

    if physics_mode == "local_deposit":
        # LocalDepositOnlyTransportEngine does not use physics tables.
        # Provide a minimal placeholder to satisfy TransportEngine.
        n_mat = len(mat_lib.material_names)
        ref_rho = (
            mat_lib.ref_density_g_cm3
            if hasattr(mat_lib, "ref_density_g_cm3")
            else mat_lib.density_g_cm3
        ).to(device=device, dtype=torch.float32)
        z = torch.zeros((n_mat, 1), device=device, dtype=torch.float32)
        tables = PhysicsTables(
            e_edges_MeV=torch.tensor([0.0, 1.0], device=device, dtype=torch.float32),
            e_centers_MeV=torch.tensor([0.5], device=device, dtype=torch.float32),
            material_names=list(mat_lib.material_names),
            ref_density_g_cm3=ref_rho,
            sigma_photo=z,
            sigma_compton=z,
            sigma_rayleigh=z,
            sigma_pair=z,
            sigma_total=z,
            p_cum=z,
            sigma_max=torch.zeros((1,), device=device, dtype=torch.float32),
            S_restricted=z,
            range_csda_cm=z,
        )
    else:
        physics_tables_cfg = sim_config["physics_tables"]
        
        if materials_cfg.get("material_library", None) is not None:
            material_library_name = "custom"
            h5_filename = f"{material_library_name}-{physics_mode}.h5"
        else:
            material_library_name = sim_config.get("materials", {}).get("name", "default_materials")
            h5_filename = f"{material_library_name}-{physics_mode}.h5"
        
        h5_path = Path(physics_tables_cfg.get("directory", "src/gpumcrpt/physics_tables/precomputed_tables")) / h5_filename

        if not h5_path.exists():
            raise FileNotFoundError(
                f"Physics table not found at {h5_path}. "
                "Generate it first using scripts/generate_physics_tables.py "
                f"with --material_library {material_library_name} --physics_mode {physics_mode}."
            )

        tables = load_physics_tables_h5(str(h5_path), device=device)

    # Decay DB (ICRP107 JSON)
    db = sim_config["decaydb"]
    assert db["type"] == "icrp107_json"
    nuclide_name = sim_config["nuclide"]["name"]
    nuclide = load_icrp107_nuclide(db_dir=db["path"], nuclide_name=nuclide_name)

    # Get voxel size from header
    zooms_mm = act_img.header.get_zooms()[:3]
    voxel_size_cm = (float(zooms_mm[0]) / 10.0, float(zooms_mm[1]) / 10.0, float(zooms_mm[2]) / 10.0)
    
    primaries, alpha_local_edep = sample_weighted_decays_and_primaries(
        activity_bqs=act,
        voxel_size_cm=voxel_size_cm,
        affine=act_img.affine,
        nuclide=nuclide,
        n_histories=int(sim_config["monte_carlo"]["n_histories"]),
        seed=int(sim_config.get("seed", 0)),
        device=device,
        cutoffs=sim_config["cutoffs"],
    )

    engine = TransportEngine(
        mats=mats,
        tables=tables,
        sim_config=sim_config,
        voxel_size_cm=voxel_size_cm,
        device=device,
    )

    edep_batches = engine.run_batches(
        primaries=primaries,
        alpha_local_edep=alpha_local_edep,
        n_batches=int(sim_config["monte_carlo"]["n_batches"]),
    )

    # Calculate voxel volume
    voxel_volume_cm3 = voxel_size_cm[0] * voxel_size_cm[1] * voxel_size_cm[2]
    
    dose, unc = edep_to_dose_and_uncertainty(
        edep_batches=edep_batches,
        rho=mats.rho,
        voxel_volume_cm3=voxel_volume_cm3,
        uncertainty_mode=sim_config["io"].get("output_uncertainty", "relative"),
    )

    # Save dose and uncertainty using nibabel
    dose_img = nib.Nifti1Image(dose.detach().cpu().numpy(), affine=act_img.affine)
    unc_img = nib.Nifti1Image(unc.detach().cpu().numpy(), affine=act_img.affine)
    nib.save(dose_img, output_dose_path)
    nib.save(unc_img, output_unc_path)
    nib.save(unc_img, output_unc_path)
    
    execution_time = time.time() - start_time
    return execution_time