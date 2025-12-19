from __future__ import annotations

import argparse

import torch

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics.tables import load_physics_tables_h5
from gpumcrpt.source.sampling import ParticleQueues
from gpumcrpt.transport.engine_gpu_triton_photon_only import PhotonOnlyTransportEngine


def _rand_unit_dirs(n: int, *, device: str) -> torch.Tensor:
    mu = 2.0 * torch.rand((n,), device=device) - 1.0
    phi = 2.0 * torch.pi * torch.rand((n,), device=device)
    sin = torch.sqrt(torch.clamp(1.0 - mu * mu, min=0.0))
    return torch.stack([mu, sin * torch.cos(phi), sin * torch.sin(phi)], dim=1).to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="toy_physics.h5")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--out", default="toy_edep_photon_only.pt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for photon-only Triton engine")

    device = "cuda"
    voxel_size_cm = (0.4, 0.4, 0.4)

    Z, Y, X = 32, 32, 32
    mats = MaterialsVolume(
        material_id=torch.zeros((Z, Y, X), device=device, dtype=torch.int32),
        rho=torch.ones((Z, Y, X), device=device, dtype=torch.float32),
    )

    tables = load_physics_tables_h5(args.tables, device=device)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    pos_cm = (torch.rand((args.n, 3), generator=g, device=device) * torch.tensor([Z, Y, X], device=device) * torch.tensor([voxel_size_cm[2], voxel_size_cm[1], voxel_size_cm[0]], device=device)).to(torch.float32)
    prim = ParticleQueues(
        photons={
            "pos_cm": pos_cm,
            "dir": _rand_unit_dirs(args.n, device=device).contiguous(),
            "E_MeV": torch.full((args.n,), 0.2, device=device, dtype=torch.float32),
            "w": torch.ones((args.n,), device=device, dtype=torch.float32),
        },
        electrons={"pos_cm": pos_cm, "dir": _rand_unit_dirs(args.n, device=device), "E_MeV": torch.zeros((args.n,), device=device), "w": torch.ones((args.n,), device=device)},
        positrons={"pos_cm": pos_cm, "dir": _rand_unit_dirs(args.n, device=device), "E_MeV": torch.zeros((args.n,), device=device), "w": torch.ones((args.n,), device=device)},
    )

    sim_config = {
        "seed": args.seed,
        "cutoffs": {"photon_keV": 3.0, "electron_keV": 20.0},
        "monte_carlo": {"max_wavefront_iters": 4096},
    }

    engine = PhotonOnlyTransportEngine(
        mats=mats,
        tables=tables,
        sim_config=sim_config,
        voxel_size_cm=voxel_size_cm,
        device=device,
    )

    edep = engine.run_one_batch(prim, alpha_local_edep=torch.zeros((Z, Y, X), device=device, dtype=torch.float32))
    torch.save(edep.detach().cpu(), args.out)
    print(f"Saved {args.out}; sum(edep)={float(edep.sum().item()):.6g} MeV; escaped={engine.last_stats.escaped_energy_MeV:.6g} MeV")


if __name__ == "__main__":
    main()
