from __future__ import annotations

import argparse

import torch

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics.tables import load_physics_tables_h5
from gpumcrpt.source.sampling import ParticleQueues
from gpumcrpt.transport.engine import TransportEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="toy_edep.pt")
    ap.add_argument("--tables", default="toy_physics.h5")
    ap.add_argument("--histories", type=int, default=100_000)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tables = load_physics_tables_h5(args.tables, device=device)

    Z, Y, X = 32, 32, 32
    mats = MaterialsVolume(
        material_id=torch.full((Z, Y, X), 2, dtype=torch.int32, device=device),
        rho=torch.full((Z, Y, X), 1.0, dtype=torch.float32, device=device),
    )

    N = int(args.histories)
    vx, vy, vz = (0.2, 0.2, 0.2)  # 2 mm voxels in cm

    g = torch.Generator(device=device)
    g.manual_seed(123)

    # Random birth positions inside volume in cm, pos_cm = [z_cm, y_cm, x_cm]
    pos_cm = torch.empty((N, 3), device=device, dtype=torch.float32)
    pos_cm[:, 0] = torch.rand((N,), generator=g, device=device) * (Z * vz)
    pos_cm[:, 1] = torch.rand((N,), generator=g, device=device) * (Y * vy)
    pos_cm[:, 2] = torch.rand((N,), generator=g, device=device) * (X * vx)

    # Random directions (unused in MVP engine)
    direction = torch.randn((N, 3), generator=g, device=device, dtype=torch.float32)
    direction = direction / torch.clamp(torch.linalg.norm(direction, dim=1, keepdim=True), min=1e-12)

    E = torch.full((N,), 0.2, dtype=torch.float32, device=device)  # 200 keV photons
    w = torch.ones((N,), dtype=torch.float32, device=device)

    empty = {
        "pos_cm": torch.empty((0, 3), device=device),
        "dir": torch.empty((0, 3), device=device),
        "E_MeV": torch.empty((0,), device=device),
        "w": torch.empty((0,), device=device),
    }

    prim = ParticleQueues(
        photons={"pos_cm": pos_cm, "dir": direction, "E_MeV": E, "w": w},
        electrons=empty,
        positrons=empty,
    )

    sim_cfg = {
        "cutoffs": {"photon_keV": 3.0, "electron_keV": 20.0},
        "monte_carlo": {"n_batches": 4},
        "seed": 123,
    }

    engine = TransportEngine(
        mats=mats,
        tables=tables,
        sim_config=sim_cfg,
        voxel_size_cm=(vx, vy, vz),
        device=device,
    )

    edep_batches = engine.run_batches(prim, alpha_local_edep=torch.zeros((Z, Y, X), device=device), n_batches=4)
    torch.save(edep_batches.detach().cpu(), args.out)
    print(f"Wrote {args.out} with shape {tuple(edep_batches.shape)}")


if __name__ == "__main__":
    main()
