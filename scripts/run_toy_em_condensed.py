from __future__ import annotations

import argparse

import torch

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics.tables import load_physics_tables_h5
from gpumcrpt.source.sampling import ParticleQueues
from gpumcrpt.transport.engine_gpu_triton_em_condensed import TritonEMCondensedTransportEngine


def _rand_unit_dirs(n: int, *, device: str) -> torch.Tensor:
    mu = 2.0 * torch.rand((n,), device=device) - 1.0
    phi = 2.0 * torch.pi * torch.rand((n,), device=device)
    sin = torch.sqrt(torch.clamp(1.0 - mu * mu, min=0.0))
    return torch.stack([mu, sin * torch.cos(phi), sin * torch.sin(phi)], dim=1).to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="toy_physics.h5")
    ap.add_argument("--n_ph", type=int, default=100_000)
    ap.add_argument("--n_e", type=int, default=50_000)
    ap.add_argument("--n_pos", type=int, default=50_000)
    ap.add_argument("--out", default="toy_edep_em_condensed.pt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for em_condensed Triton engine")

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

    def rand_pos(n: int) -> torch.Tensor:
        scale = torch.tensor([Z, Y, X], device=device) * torch.tensor(
            [voxel_size_cm[2], voxel_size_cm[1], voxel_size_cm[0]], device=device
        )
        return (torch.rand((n, 3), generator=g, device=device) * scale).to(torch.float32)

    ph_pos = rand_pos(args.n_ph)
    e_pos = rand_pos(args.n_e)
    p_pos = rand_pos(args.n_pos)

    prim = ParticleQueues(
        photons={
            "pos_cm": ph_pos,
            "dir": _rand_unit_dirs(args.n_ph, device=device).contiguous(),
            "E_MeV": torch.full((args.n_ph,), 0.2, device=device, dtype=torch.float32),
            "w": torch.ones((args.n_ph,), device=device, dtype=torch.float32),
        },
        electrons={
            "pos_cm": e_pos,
            "dir": _rand_unit_dirs(args.n_e, device=device).contiguous(),
            "E_MeV": torch.full((args.n_e,), 0.2, device=device, dtype=torch.float32),
            "w": torch.ones((args.n_e,), device=device, dtype=torch.float32),
        },
        positrons={
            "pos_cm": p_pos,
            "dir": _rand_unit_dirs(args.n_pos, device=device).contiguous(),
            # 30 keV is just above the default 20 keV cutoff -> they stop quickly and annihilate
            "E_MeV": torch.full((args.n_pos,), 0.03, device=device, dtype=torch.float32),
            "w": torch.ones((args.n_pos,), device=device, dtype=torch.float32),
        },
    )

    sim_config = {
        "seed": args.seed,
        "cutoffs": {"photon_keV": 3.0, "electron_keV": 20.0},
        "monte_carlo": {"max_wavefront_iters": 4096},
        "electron_transport": {"f_voxel": 0.3, "f_range": 0.2, "max_dE_frac": 0.2, "max_steps": 4096},
    }

    engine = TritonEMCondensedTransportEngine(
        mats=mats,
        tables=tables,
        sim_config=sim_config,
        voxel_size_cm=voxel_size_cm,
        device=device,
    )

    edep = engine.run_one_batch(prim, alpha_local_edep=torch.zeros((Z, Y, X), device=device, dtype=torch.float32))
    torch.save(edep.detach().cpu(), args.out)

    print(
        f"Saved {args.out}; sum(edep)={float(edep.sum().item()):.6g} MeV; "
        f"escaped_ph={engine.last_stats.escaped_photon_energy_MeV:.6g} MeV; "
        f"annihilations={engine.last_stats.annihilations}"
    )


if __name__ == "__main__":
    main()
