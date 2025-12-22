from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from nibabel.processing import resample_from_to
from .materials.hu_materials import (
    build_default_materials_library,
    build_materials_from_hu,
    build_materials_library_from_config,
)
from .physics.tables import load_physics_tables_h5
from .decaydb import load_icrp107_nuclide
from .source.sampling import sample_weighted_decays_and_primaries
from .transport.engine import TransportEngine
from .dose.scoring import edep_to_dose_and_uncertainty


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
) -> None:
    device = device or sim_config.get("device", "cuda")

    act_img = nib.load(activity_nifti_path)
    ct_img = nib.load(ct_nifti_path)

    if sim_config["io"].get("resample_ct_to_activity", True):
        ct_img = resample_from_to(ct_img, act_img)

    act_data = act_img.get_fdata(dtype=np.float32)
    ct_data = ct_img.get_fdata(dtype=np.float32)
    
    act = torch.from_numpy(act_data).to(device=device, dtype=torch.float32)
    hu = torch.from_numpy(ct_data).to(device=device, dtype=torch.float32)

    materials_cfg = sim_config.get("materials", {})
    if materials_cfg.get("material_library", None) is not None:
        mat_lib = build_materials_library_from_config(materials_cfg, device=device)
    else:
        mat_lib = build_default_materials_library(device=device)

    mats = build_materials_from_hu(
        hu=hu,
        hu_to_density=sim_config["materials"]["hu_to_density"],
        hu_to_class=sim_config["materials"]["hu_to_class"],
        material_library=mat_lib,
        device=device,
    )

    # Determine physics table path dynamically
    physics_tables_cfg = sim_config["physics_tables"]
    material_library_name = sim_config["materials"].get("name", "default_materials")
    physics_mode = sim_config["monte_carlo"]["triton"]["engine"]
    
    h5_filename = f"{material_library_name}-{physics_mode}.h5"
    h5_path = Path(physics_tables_cfg.get("directory", "src/gpumcrpt/physics_tables/precomputed_tables")) / h5_filename
    
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Physics table not found at {h5_path}. "
            f"Please generate it first using the 'scripts/generate_physics_tables.py' script."
        )
        
    tables = load_physics_tables_h5(h5_path, device=device)

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