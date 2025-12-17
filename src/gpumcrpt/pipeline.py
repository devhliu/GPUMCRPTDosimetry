from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .io.nifti import load_nifti, save_nifti_like, resample_to_reference
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

    act_img = load_nifti(activity_nifti_path)
    ct_img = load_nifti(ct_nifti_path)

    if sim_config["io"].get("resample_ct_to_activity", True):
        ct_img = resample_to_reference(moving=ct_img, reference=act_img, mode="linear")

    act = torch.from_numpy(act_img.data).to(device=device, dtype=torch.float32)
    hu = torch.from_numpy(ct_img.data).to(device=device, dtype=torch.float32)

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

    tables = load_physics_tables_h5(sim_config["physics_tables"]["h5_path"], device=device)

    # Decay DB (ICRP107 JSON)
    db = sim_config["decaydb"]
    assert db["type"] == "icrp107_json"
    nuclide_name = sim_config["nuclide"]["name"]
    nuclide = load_icrp107_nuclide(db_dir=db["path"], nuclide_name=nuclide_name)

    primaries, alpha_local_edep = sample_weighted_decays_and_primaries(
        activity_bqs=act,
        voxel_size_cm=act_img.voxel_size_cm,
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
        voxel_size_cm=act_img.voxel_size_cm,
        device=device,
    )

    edep_batches = engine.run_batches(
        primaries=primaries,
        alpha_local_edep=alpha_local_edep,
        n_batches=int(sim_config["monte_carlo"]["n_batches"]),
    )

    dose, unc = edep_to_dose_and_uncertainty(
        edep_batches=edep_batches,
        rho=mats.rho,
        voxel_volume_cm3=act_img.voxel_volume_cm3,
        uncertainty_mode=sim_config["io"].get("output_uncertainty", "relative"),
    )

    save_nifti_like(act_img, dose.detach().cpu().numpy(), output_dose_path)
    save_nifti_like(act_img, unc.detach().cpu().numpy(), output_unc_path)