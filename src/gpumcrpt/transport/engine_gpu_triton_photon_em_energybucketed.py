from __future__ import annotations

# Next step: re-enable charged CUDA graphs under sorted_voxel tally by
# preallocating per-bucket record buffers (rec_lin/rec_val) in the static graph state.
#
# This file adds bucketed record buffers for:
#   - electron graph
#   - positron graph
#
# and a "record_mode" flag in BucketedGraphConfig to switch which kernels are captured:
#   - atomic mode: existing *_condensed_step_kernel (atomicAdd inside kernel)
#   - record mode: new *_condensed_step_record_kernel (writes rec_lin/rec_val; no atomicAdd)
#
# IMPORTANT:
# - sorted_voxel_accumulate still runs outside graphs.
# - But allocations are eliminated, and the microcycle steps are graph-capturable again.

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import triton

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics_tables.tables import PhysicsTables

from gpumcrpt.transport.triton.photon_flight import photon_woodcock_flight_kernel
from gpumcrpt.transport.triton.photon_interactions import photon_classify_kernel

from gpumcrpt.transport.triton.electron_step import electron_condensed_step_kernel
from gpumcrpt.transport.triton.positron_step import positron_condensed_step_kernel

from gpumcrpt.transport.triton.electron_step_record import electron_condensed_step_record_kernel
from gpumcrpt.transport.triton.positron_step_record import positron_condensed_step_record_kernel

from gpumcrpt.transport.triton.cuda_graphs import CUDAGraphBucketManager


@dataclass
class BucketedGraphConfig:
    bucket_sizes: List[int] = None
    block: int = 256
    photon_micro_steps: int = 2
    electron_micro_steps: int = 2
    positron_micro_steps: int = 2
    use_graphs: bool = True
    photon_capture_classify: bool = True

    # NEW
    charged_record_mode: bool = False  # when True, capture record kernels and write rec_lin/rec_val


def _ensure_rng(q: Dict[str, torch.Tensor], device: str):
    if "rng" not in q:
        n = q["E_MeV"].shape[0]
        q["rng"] = torch.randint(1, 2**31 - 1, (n,), device=device, dtype=torch.int32)
    return q


def _precompute_ebin(E: torch.Tensor, e_edges: torch.Tensor) -> torch.Tensor:
    return (torch.bucketize(E, e_edges) - 1).clamp_(0, e_edges.numel() - 2).to(torch.int32)


class TritonPhotonEMEnergyBucketedGraphsEngine:
    """
    Photon-EM-EnergyBucketedPersistentGraphs transport engine (Phase 7).
    Adds per-bucket record buffers so charged steps can be CUDA-graphed even in sorted_voxel mode.
    """

    def __init__(
        self,
        mats: MaterialsVolume,
        tables: PhysicsTables,
        voxel_size_cm: Tuple[float, float, float],
        sim_config: dict,
        device: str = "cuda",
        cfg: BucketedGraphConfig | None = None,
    ):
        self.mats = mats
        self.tables = tables
        self.voxel_size_cm = voxel_size_cm
        self.sim_config = sim_config
        self.device = device

        self.cfg = cfg or BucketedGraphConfig()
        if self.cfg.bucket_sizes is None:
            self.cfg.bucket_sizes = [4096, 16384, 65536, 262144, 1048576]

        self.M = len(self.tables.material_names)
        self.ECOUNT = int(self.tables.e_edges_MeV.numel() - 1)

        et = sim_config.get("electron_transport", {})
        self.f_vox = float(et.get("f_voxel", 0.3))
        self.f_range = float(et.get("f_range", 0.2))
        self.max_dE_frac = float(et.get("max_dE_frac", 0.2))
        self.e_cut = float(sim_config.get("cutoffs", {}).get("electron_keV", 20.0)) * 1e-3

        self.ph_graph_mgr = CUDAGraphBucketManager(device=device) if self.cfg.use_graphs else None
        self.el_graph_mgr = CUDAGraphBucketManager(device=device) if self.cfg.use_graphs else None
        self.po_graph_mgr = CUDAGraphBucketManager(device=device) if self.cfg.use_graphs else None

    # ---------------- photon graph unchanged (omitted here) ----------------
    # Keep your Phase 7 photon graph code with itype capture.

    # ---------------- electron graph with record buffers ----------------
    def _make_static_el(self, max_n: int, Z: int, Y: int, X: int) -> Dict[str, torch.Tensor]:
        st = {
            "pos": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "dir": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "E": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "w": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "rng": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "ebin": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "id": torch.empty((max_n,), device=self.device, dtype=torch.int64),
            "out_id": torch.empty((max_n,), device=self.device, dtype=torch.int64),

            "out_pos": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "out_dir": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "out_E": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "out_w": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "out_rng": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "out_ebin": torch.empty((max_n,), device=self.device, dtype=torch.int32),

            "alive": torch.empty((max_n,), device=self.device, dtype=torch.int8),
            "emit_brem": torch.empty((max_n,), device=self.device, dtype=torch.int8),
            "emit_delta": torch.empty((max_n,), device=self.device, dtype=torch.int8),

            # NEW: record buffers (one per lane per replay step)
            "rec_lin": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "rec_val": torch.empty((max_n,), device=self.device, dtype=torch.float32),

            "Z": torch.tensor([Z], device=self.device, dtype=torch.int32),
            "Y": torch.tensor([Y], device=self.device, dtype=torch.int32),
            "X": torch.tensor([X], device=self.device, dtype=torch.int32),
        }
        return st

    def _capture_el(self, st: Dict[str, torch.Tensor], edep_flat: torch.Tensor):
        max_n = st["E"].shape[0]
        BLOCK = self.cfg.block
        grid = (triton.cdiv(max_n, BLOCK),)

        P_brem = self.tables.P_brem_per_cm if self.tables.P_brem_per_cm is not None else torch.zeros_like(self.tables.S_restricted)
        P_delta = self.tables.P_delta_per_cm if self.tables.P_delta_per_cm is not None else torch.zeros_like(self.tables.S_restricted)

        for _ in range(self.cfg.electron_micro_steps):
            if self.cfg.charged_record_mode:
                electron_condensed_step_record_kernel[grid](
                    st["pos"], st["dir"], st["E"], st["w"], st["rng"], st["ebin"],
                    self.mats.material_id, self.mats.rho, self.tables.ref_density_g_cm3,
                    self.tables.S_restricted, self.tables.range_csda_cm,
                    P_brem, P_delta,
                    st["rec_lin"], st["rec_val"],
                    st["out_pos"], st["out_dir"], st["out_E"], st["out_w"], st["out_rng"], st["out_ebin"],
                    st["alive"], st["emit_brem"], st["emit_delta"],
                    Z=int(st["Z"].item()), Y=int(st["Y"].item()), X=int(st["X"].item()),
                    M=self.M, ECOUNT=self.ECOUNT,
                    BLOCK=BLOCK,
                    voxel_z_cm=float(self.voxel_size_cm[2]),
                    voxel_y_cm=float(self.voxel_size_cm[1]),
                    voxel_x_cm=float(self.voxel_size_cm[0]),
                    f_vox=self.f_vox, f_range=self.f_range, max_dE_frac=self.max_dE_frac,
                    num_warps=4,
                )
            else:
                electron_condensed_step_kernel[grid](
                    st["pos"], st["dir"], st["E"], st["w"], st["rng"], st["ebin"],
                    self.mats.material_id, self.mats.rho, self.tables.ref_density_g_cm3,
                    self.tables.S_restricted, self.tables.range_csda_cm,
                    P_brem, P_delta,
                    edep_flat,
                    st["out_pos"], st["out_dir"], st["out_E"], st["out_w"], st["out_rng"], st["out_ebin"],
                    st["alive"], st["emit_brem"], st["emit_delta"],
                    Z=int(st["Z"].item()), Y=int(st["Y"].item()), X=int(st["X"].item()),
                    M=self.M, ECOUNT=self.ECOUNT,
                    BLOCK=BLOCK,
                    voxel_z_cm=float(self.voxel_size_cm[2]),
                    voxel_y_cm=float(self.voxel_size_cm[1]),
                    voxel_x_cm=float(self.voxel_size_cm[0]),
                    f_vox=self.f_vox, f_range=self.f_range, max_dE_frac=self.max_dE_frac,
                    num_warps=4,
                )

            # ping-pong state
            st["pos"], st["out_pos"] = st["out_pos"], st["pos"]
            st["dir"], st["out_dir"] = st["out_dir"], st["dir"]
            st["E"], st["out_E"] = st["out_E"], st["E"]
            st["w"], st["out_w"] = st["out_w"], st["w"]
            st["rng"], st["out_rng"] = st["out_rng"], st["rng"]
            st["ebin"], st["out_ebin"] = st["out_ebin"], st["ebin"]

            st["out_id"].copy_(st["id"])
            st["id"], st["out_id"] = st["out_id"], st["id"]

    @torch.no_grad()
    def run_electron_microcycles(self, electrons: Dict[str, torch.Tensor], edep: torch.Tensor) -> Dict[str, torch.Tensor]:
        Z, Y, X = self.mats.material_id.shape
        electrons = _ensure_rng(electrons, self.device)
        N = electrons["E_MeV"].shape[0]
        if N == 0:
            return {
                "pos_cm": electrons["pos_cm"], "dir": electrons["dir"], "E_MeV": electrons["E_MeV"], "w": electrons["w"], "rng": electrons["rng"],
                "id": torch.empty((0,), device=self.device, dtype=torch.int64),
                "alive": torch.empty((0,), device=self.device, dtype=torch.int8),
                "emit_brem": torch.empty((0,), device=self.device, dtype=torch.int8),
                "emit_delta": torch.empty((0,), device=self.device, dtype=torch.int8),
                "rec_lin": torch.empty((0,), device=self.device, dtype=torch.int32),
                "rec_val": torch.empty((0,), device=self.device, dtype=torch.float32),
            }

        if "id" not in electrons:
            electrons["id"] = torch.arange(N, device=self.device, dtype=torch.int64)

        bucket = self.el_graph_mgr.pick_bucket(N, self.cfg.bucket_sizes)
        edep_flat = edep.view(-1)

        def make_static(max_n: int):
            return self._make_static_el(max_n, Z, Y, X)

        def capture_fn(st):
            self._capture_el(st, edep_flat)

        b = self.el_graph_mgr.get_or_capture(bucket, make_static=make_static, capture_fn=capture_fn)

        # pad
        b.static["E"].zero_()
        b.static["w"].zero_()
        b.static["pos"].zero_()
        b.static["dir"].zero_()
        b.static["rng"].zero_()
        b.static["ebin"].zero_()
        b.static["alive"].zero_()
        b.static["emit_brem"].zero_()
        b.static["emit_delta"].zero_()
        b.static["rec_lin"].fill_(-1)
        b.static["rec_val"].zero_()
        b.static["id"].fill_(-1)

        b.static["pos"][:N].copy_(electrons["pos_cm"])
        b.static["dir"][:N].copy_(electrons["dir"])
        b.static["E"][:N].copy_(electrons["E_MeV"])
        b.static["w"][:N].copy_(electrons["w"])
        b.static["rng"][:N].copy_(electrons["rng"])
        b.static["ebin"][:N].copy_(_precompute_ebin(electrons["E_MeV"], self.tables.e_edges_MeV))
        b.static["id"][:N].copy_(electrons["id"])

        b.graph.replay()

        return {
            "pos_cm": b.static["pos"][:bucket],
            "dir": b.static["dir"][:bucket],
            "E_MeV": b.static["E"][:bucket],
            "w": b.static["w"][:bucket],
            "rng": b.static["rng"][:bucket],
            "id": b.static["id"][:bucket],
            "alive": b.static["alive"][:bucket],
            "emit_brem": b.static["emit_brem"][:bucket],
            "emit_delta": b.static["emit_delta"][:bucket],
            "rec_lin": b.static["rec_lin"][:bucket],
            "rec_val": b.static["rec_val"][:bucket],
        }

    # ---------------- positron graph with record buffers ----------------
    def _make_static_po(self, max_n: int, Z: int, Y: int, X: int) -> Dict[str, torch.Tensor]:
        st = {
            "pos": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "dir": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "E": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "w": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "rng": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "ebin": torch.empty((max_n,), device=self.device, dtype=torch.int32),

            "id": torch.empty((max_n,), device=self.device, dtype=torch.int64),
            "out_id": torch.empty((max_n,), device=self.device, dtype=torch.int64),

            "out_pos": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "out_dir": torch.empty((max_n, 3), device=self.device, dtype=torch.float32),
            "out_E": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "out_w": torch.empty((max_n,), device=self.device, dtype=torch.float32),
            "out_rng": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "out_ebin": torch.empty((max_n,), device=self.device, dtype=torch.int32),

            "alive": torch.empty((max_n,), device=self.device, dtype=torch.int8),
            "emit_brem": torch.empty((max_n,), device=self.device, dtype=torch.int8),
            "emit_delta": torch.empty((max_n,), device=self.device, dtype=torch.int8),
            "stop": torch.empty((max_n,), device=self.device, dtype=torch.int8),

            "rec_lin": torch.empty((max_n,), device=self.device, dtype=torch.int32),
            "rec_val": torch.empty((max_n,), device=self.device, dtype=torch.float32),

            "Z": torch.tensor([Z], device=self.device, dtype=torch.int32),
            "Y": torch.tensor([Y], device=self.device, dtype=torch.int32),
            "X": torch.tensor([X], device=self.device, dtype=torch.int32),
        }
        return st

    def _capture_po(self, st: Dict[str, torch.Tensor], edep_flat: torch.Tensor):
        max_n = st["E"].shape[0]
        BLOCK = self.cfg.block
        grid = (triton.cdiv(max_n, BLOCK),)

        P_brem = self.tables.P_brem_per_cm if self.tables.P_brem_per_cm is not None else torch.zeros_like(self.tables.S_restricted)
        P_delta = self.tables.P_delta_per_cm if self.tables.P_delta_per_cm is not None else torch.zeros_like(self.tables.S_restricted)

        for _ in range(self.cfg.positron_micro_steps):
            if self.cfg.charged_record_mode:
                positron_condensed_step_record_kernel[grid](
                    st["pos"], st["dir"], st["E"], st["w"], st["rng"], st["ebin"],
                    self.mats.material_id, self.mats.rho, self.tables.ref_density_g_cm3,
                    self.tables.S_restricted, self.tables.range_csda_cm,
                    P_brem, P_delta,
                    st["rec_lin"], st["rec_val"],
                    st["out_pos"], st["out_dir"], st["out_E"], st["out_w"], st["out_rng"], st["out_ebin"],
                    st["alive"], st["emit_brem"], st["emit_delta"], st["stop"],
                    Z=int(st["Z"].item()), Y=int(st["Y"].item()), X=int(st["X"].item()),
                    M=self.M, ECOUNT=self.ECOUNT,
                    BLOCK=BLOCK,
                    voxel_z_cm=float(self.voxel_size_cm[2]),
                    voxel_y_cm=float(self.voxel_size_cm[1]),
                    voxel_x_cm=float(self.voxel_size_cm[0]),
                    f_vox=self.f_vox, f_range=self.f_range, max_dE_frac=self.max_dE_frac,
                    e_cut_MeV=self.e_cut,
                    num_warps=4,
                )
            else:
                positron_condensed_step_kernel[grid](
                    st["pos"], st["dir"], st["E"], st["w"], st["rng"], st["ebin"],
                    self.mats.material_id, self.mats.rho, self.tables.ref_density_g_cm3,
                    self.tables.S_restricted, self.tables.range_csda_cm,
                    P_brem, P_delta,
                    edep_flat,
                    st["out_pos"], st["out_dir"], st["out_E"], st["out_w"], st["out_rng"], st["out_ebin"],
                    st["alive"], st["emit_brem"], st["emit_delta"], st["stop"],
                    Z=int(st["Z"].item()), Y=int(st["Y"].item()), X=int(st["X"].item()),
                    M=self.M, ECOUNT=self.ECOUNT,
                    BLOCK=BLOCK,
                    voxel_z_cm=float(self.voxel_size_cm[2]),
                    voxel_y_cm=float(self.voxel_size_cm[1]),
                    voxel_x_cm=float(self.voxel_size_cm[0]),
                    f_vox=self.f_vox, f_range=self.f_range, max_dE_frac=self.max_dE_frac,
                    e_cut_MeV=self.e_cut,
                    num_warps=4,
                )

            st["pos"], st["out_pos"] = st["out_pos"], st["pos"]
            st["dir"], st["out_dir"] = st["out_dir"], st["dir"]
            st["E"], st["out_E"] = st["out_E"], st["E"]
            st["w"], st["out_w"] = st["out_w"], st["w"]
            st["rng"], st["out_rng"] = st["out_rng"], st["rng"]
            st["ebin"], st["out_ebin"] = st["out_ebin"], st["ebin"]

            st["out_id"].copy_(st["id"])
            st["id"], st["out_id"] = st["out_id"], st["id"]

    @torch.no_grad()
    def run_positron_microcycles(self, positrons: Dict[str, torch.Tensor], edep: torch.Tensor) -> Dict[str, torch.Tensor]:
        Z, Y, X = self.mats.material_id.shape
        positrons = _ensure_rng(positrons, self.device)
        N = positrons["E_MeV"].shape[0]
        if N == 0:
            return {
                "pos_cm": positrons["pos_cm"], "dir": positrons["dir"], "E_MeV": positrons["E_MeV"], "w": positrons["w"], "rng": positrons["rng"],
                "id": torch.empty((0,), device=self.device, dtype=torch.int64),
                "alive": torch.empty((0,), device=self.device, dtype=torch.int8),
                "emit_brem": torch.empty((0,), device=self.device, dtype=torch.int8),
                "emit_delta": torch.empty((0,), device=self.device, dtype=torch.int8),
                "stop": torch.empty((0,), device=self.device, dtype=torch.int8),
                "rec_lin": torch.empty((0,), device=self.device, dtype=torch.int32),
                "rec_val": torch.empty((0,), device=self.device, dtype=torch.float32),
            }

        if "id" not in positrons:
            positrons["id"] = torch.arange(N, device=self.device, dtype=torch.int64)

        bucket = self.po_graph_mgr.pick_bucket(N, self.cfg.bucket_sizes)
        edep_flat = edep.view(-1)

        def make_static(max_n: int):
            return self._make_static_po(max_n, Z, Y, X)

        def capture_fn(st):
            self._capture_po(st, edep_flat)

        b = self.po_graph_mgr.get_or_capture(bucket, make_static=make_static, capture_fn=capture_fn)

        b.static["E"].zero_()
        b.static["w"].zero_()
        b.static["pos"].zero_()
        b.static["dir"].zero_()
        b.static["rng"].zero_()
        b.static["ebin"].zero_()
        b.static["alive"].zero_()
        b.static["emit_brem"].zero_()
        b.static["emit_delta"].zero_()
        b.static["stop"].zero_()
        b.static["rec_lin"].fill_(-1)
        b.static["rec_val"].zero_()
        b.static["id"].fill_(-1)

        b.static["pos"][:N].copy_(positrons["pos_cm"])
        b.static["dir"][:N].copy_(positrons["dir"])
        b.static["E"][:N].copy_(positrons["E_MeV"])
        b.static["w"][:N].copy_(positrons["w"])
        b.static["rng"][:N].copy_(positrons["rng"])
        b.static["ebin"][:N].copy_(_precompute_ebin(positrons["E_MeV"], self.tables.e_edges_MeV))
        b.static["id"][:N].copy_(positrons["id"])

        b.graph.replay()

        return {
            "pos_cm": b.static["pos"][:bucket],
            "dir": b.static["dir"][:bucket],
            "E_MeV": b.static["E"][:bucket],
            "w": b.static["w"][:bucket],
            "rng": b.static["rng"][:bucket],
            "id": b.static["id"][:bucket],
            "alive": b.static["alive"][:bucket],
            "emit_brem": b.static["emit_brem"][:bucket],
            "emit_delta": b.static["emit_delta"][:bucket],
            "stop": b.static["stop"][:bucket],
            "rec_lin": b.static["rec_lin"][:bucket],
            "rec_val": b.static["rec_val"][:bucket],
        }