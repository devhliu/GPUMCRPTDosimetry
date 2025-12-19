"""
Final Phase 10 wiring blocks matching your *actual* initialization schema.

You provided:
- ParticleBank stores rng_offset:int64 (single), not Philox SoA.
- You confirmed you want Option A (Philox SoA) for kernels.
- VacancyBank currently has rng_offset:int64 as well.

Therefore this "final wiring" does TWO things:

(A) Minimal invasive: keep your existing rng_offset storage, but expose Philox SoA views to kernels:
    key0/key1 are engine constants, ctr0 derived from rng_offset low 32 bits, ctr1..3 = 0.
    After a stage consumes RNG, we advance rng_offset by a fixed increment (deterministic).

(B) Complete Phase 10: implement PE -> append electrons/vacancies to banks + relax vacancies -> append photons/electrons.

IMPORTANT: To truly eliminate the bridge and be perfectly aligned with §4.2 in physics_rpt_design4GPUMC.md,
I still recommend upgrading ParticleBank/VacancyBank to store rng_key0/rng_key1/rng_ctr0..3 int32 directly.
But the bridge lets you integrate now without rewriting your Bank classes.

This file assumes you will add missing bank fields:
- ParticleBank: ebin + rng_key0/rng_key1/rng_ctr0..3 (or keep bridge and NOT store these)
- VacancyBank: w + rng fields already exist as rng_offset but Phase 10 needs SoA rng for relaxation.

Since your banks do NOT currently have rng_key*/rng_ctr* attributes, the append kernels below
write rng_key/rng_ctr into *staging* buffers only, then we convert back to rng_offset when storing into banks.
"""

from __future__ import annotations
from typing import Tuple
import torch
import triton

from gpumcrpt.kernels.triton.photon_pe_soa import photon_photoelectric_pe_soa_kernel
from gpumcrpt.kernels.triton.atomic_relaxation_soa import atomic_relaxation_soa_kernel
from gpumcrpt.kernels.triton.bank_append_soa import (
    append_photons_bank_soa_kernel,
    append_electrons_bank_soa_kernel,
)
from gpumcrpt.kernels.triton.vacancy_append_full_soa import append_vacancies_full_bank_soa_kernel
from gpumcrpt.kernels.triton.energy_bin import compute_ebin_log_uniform_kernel


# ---- interaction codes ----
PE_CODE = 0  # set to your classifier encoding

# ---- Philox key constants (per run) ----
PHILOX_KEY0 = 0x12345678
PHILOX_KEY1 = 0x9ABCDEF0


def _ctr0_from_rng_offset(rng_offset_i64: torch.Tensor) -> torch.Tensor:
    return (rng_offset_i64.to(torch.int64) & 0xFFFFFFFF).to(torch.int32)


def _rng_offset_from_ctr0(ctr0_i32: torch.Tensor) -> torch.Tensor:
    # store ctr0 into low bits; high bits 0 for now
    return ctr0_i32.to(torch.int64)


@torch.no_grad()
def compute_ebin_log_uniform(E: torch.Tensor, *, common_log_E_min: float, common_log_step_inv: float, NB: int) -> torch.Tensor:
    out = torch.empty((E.numel(),), device=E.device, dtype=torch.int32)
    if E.numel() == 0:
        return out
    grid = (triton.cdiv(E.numel(), 256),)
    compute_ebin_log_uniform_kernel[grid](
        E, out,
        n=E.numel(),
        common_log_E_min=float(common_log_E_min),
        common_log_step_inv=float(common_log_step_inv),
        NB=int(NB),
        BLOCK=256,
        num_warps=4,
    )
    return out


class PhotonEMBankSoAVacancyRelaxationMixin:
    """
    Mixin-style: include these methods in your TritonEngine class (Phase 10).

    Photon-EM-BankSoAVacancyRelaxation architecture with:
      - SoA particle banks (x,y,z,dx,dy,dz,E,w,ebin,rng,status)
      - Vacancy bank (x,y,z,atom_Z,shell_idx,w,rng,status)
      - Atomic appends to bank tails
      - Vacancy relaxation cascade (X-rays, Auger electrons)
      - Philox 4x32 counter-based RNG

    Requires you define/own:
      self.photons, self.electrons, self.vacancies
      self.global_counters (int32[>=3])
      self.material_id (int32[Z*Y*X])  [voxel material indices]
      self.tables (physics tables)
      self.mat_table.atom_Z (int32[M])  [material effective atom Z]
      self.Z,self.Y,self.X and self.voxel_size_cm
      self.edep_flat (float32 energy deposition)
      self.photon_cut_MeV, self.e_cut_MeV (cutoffs)

    And requires your existing infrastructure:
      self.compact_photons() -> photon_active_indices (int32)
      self.compact_vacancies() -> vacancy_active_indices (int32)
      self.get_photon_itype_for_active(photon_active_indices) -> photon_itype (int32/int8)
    """

    @torch.no_grad()
    def phase10_step(self):
        # 0) active lists (you already do compaction; reuse your functions)
        photon_active = self.compact_photons()          # int32 [Nph_active]
        photon_itype = self.get_photon_itype_for_active(photon_active)  # aligned with active
        self._phase10_dispatch_photoelectric(photon_active, photon_itype)

        # Relax vacancies created this step as well (single-queue model)
        vacancy_active = self.compact_vacancies()       # int32 [Nvac_active]
        self._phase10_dispatch_relaxation(vacancy_active)

        # ... then continue with other transport (Compton, etc.) ...
        # finally end-of-step compaction already done by your engine


    @torch.no_grad()
    def _phase10_dispatch_photoelectric(self, photon_active: torch.Tensor, photon_itype: torch.Tensor):
        pe_mask = (photon_itype == PE_CODE)
        if not torch.any(pe_mask):
            return

        pe_idx = photon_active[pe_mask]
        Npe = int(pe_idx.numel())
        if Npe == 0:
            return
        idx64 = pe_idx.to(torch.int64)
        dev = self.device

        # ---- gather photon bank SoA into staging ----
        in_x = self.photons.x[idx64]
        in_y = self.photons.y[idx64]
        in_z = self.photons.z[idx64]
        in_dx = self.photons.dx[idx64]
        in_dy = self.photons.dy[idx64]
        in_dz = self.photons.dz[idx64]
        in_E = self.photons.E[idx64]
        in_w = self.photons.w[idx64]

        NB = int(self.tables.NB)
        in_ebin = compute_ebin_log_uniform(
            in_E, common_log_E_min=float(self.tables.common_log_E_min),
            common_log_step_inv=float(self.tables.common_log_step_inv),
            NB=NB,
        )

        # RNG bridge: rng_offset -> Philox ctr0; other ctrs 0, keys constant
        in_key0 = torch.full((Npe,), PHILOX_KEY0, device=dev, dtype=torch.int32)
        in_key1 = torch.full((Npe,), PHILOX_KEY1, device=dev, dtype=torch.int32)
        in_ctr0 = _ctr0_from_rng_offset(self.photons.rng_offset[idx64])
        in_ctr1 = torch.zeros((Npe,), device=dev, dtype=torch.int32)
        in_ctr2 = torch.zeros((Npe,), device=dev, dtype=torch.int32)
        in_ctr3 = torch.zeros((Npe,), device=dev, dtype=torch.int32)

        # ---- staging outputs (electron) ----
        e_x = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_y = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_z = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_dx = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_dy = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_dz = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_E = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_w = torch.empty((Npe,), device=dev, dtype=torch.float32)
        e_key0 = torch.empty((Npe,), device=dev, dtype=torch.int32)
        e_key1 = torch.empty((Npe,), device=dev, dtype=torch.int32)
        e_ctr0 = torch.empty((Npe,), device=dev, dtype=torch.int32)
        e_ctr1 = torch.empty((Npe,), device=dev, dtype=torch.int32)
        e_ctr2 = torch.empty((Npe,), device=dev, dtype=torch.int32)
        e_ctr3 = torch.empty((Npe,), device=dev, dtype=torch.int32)
        e_has = torch.empty((Npe,), device=dev, dtype=torch.int8)

        # ---- staging outputs (vacancy base) ----
        v_x = torch.empty((Npe,), device=dev, dtype=torch.float32)
        v_y = torch.empty((Npe,), device=dev, dtype=torch.float32)
        v_z = torch.empty((Npe,), device=dev, dtype=torch.float32)
        v_Z = torch.empty((Npe,), device=dev, dtype=torch.int32)
        v_shell = torch.empty((Npe,), device=dev, dtype=torch.int32)
        v_has = torch.empty((Npe,), device=dev, dtype=torch.int8)

        # geometry + tables
        vx, vy, vz = map(float, self.voxel_size_cm)
        M = int(self.tables.relax_shell_cdf.shape[0])
        S = int(self.tables.relax_shell_cdf.shape[1])

        grid = (triton.cdiv(Npe, 256),)
        photon_photoelectric_pe_soa_kernel[grid](
            in_x, in_y, in_z,
            in_dx, in_dy, in_dz,
            in_E, in_w,
            in_ebin,
            in_key0, in_key1,
            in_ctr0, in_ctr1, in_ctr2, in_ctr3,
            self.material_id,
            self.mat_table.atom_Z,
            self.tables.relax_shell_cdf,
            self.tables.relax_E_bind_MeV,
            # electron staging
            e_x, e_y, e_z,
            e_dx, e_dy, e_dz,
            e_E, e_w,
            e_key0, e_key1,
            e_ctr0, e_ctr1, e_ctr2, e_ctr3,
            e_has,
            # vacancy staging
            v_x, v_y, v_z,
            v_Z, v_shell,
            v_has,
            self.edep_flat,
            Npe=Npe,
            Zdim=int(self.Z), Ydim=int(self.Y), Xdim=int(self.X),
            M=M, S=S,
            voxel_x_cm=vx, voxel_y_cm=vy, voxel_z_cm=vz,
            BLOCK=256,
            num_warps=4,
        )

        # Append electrons to electron bank:
        # Your current ParticleBank lacks ebin + rng SoA fields, so append_... kernels can't target it directly.
        # Therefore: we do a *direct append* into your bank using Python-side masked scatter to reserved indices.
        # This keeps the Integration correct; you can later replace with a Triton bank-append tailored to your exact bank.
        e_has_b = e_has.bool()
        if torch.any(e_has_b):
            n_new = int(e_has_b.sum().item())  # if you want to avoid sync, replace with a GPU prefix-sum allocator
            base = int(self.global_counters[1].item())  # same note
            self.global_counters[1] += n_new

            dst = torch.arange(base, base + n_new, device=dev, dtype=torch.int64)
            src = torch.nonzero(e_has_b, as_tuple=False).view(-1).to(torch.int64)

            self.electrons.x[dst] = e_x[src]
            self.electrons.y[dst] = e_y[src]
            self.electrons.z[dst] = e_z[src]
            self.electrons.dx[dst] = e_dx[src]
            self.electrons.dy[dst] = e_dy[src]
            self.electrons.dz[dst] = e_dz[src]
            self.electrons.E[dst] = e_E[src]
            self.electrons.w[dst] = e_w[src]
            self.electrons.status[dst] = 1

            # Store RNG back into rng_offset as ctr0 (bridge)
            self.electrons.rng_offset[dst] = _rng_offset_from_ctr0(e_ctr0[src])

        # Append vacancies to vacancy bank (full fields)
        v_has_b = v_has.bool()
        if torch.any(v_has_b):
            n_new = int(v_has_b.sum().item())
            base = int(self.global_counters[2].item())
            self.global_counters[2] += n_new

            dst = torch.arange(base, base + n_new, device=dev, dtype=torch.int64)
            src = torch.nonzero(v_has_b, as_tuple=False).view(-1).to(torch.int64)

            self.vacancies.x[dst] = v_x[src]
            self.vacancies.y[dst] = v_y[src]
            self.vacancies.z[dst] = v_z[src]
            self.vacancies.atom_Z[dst] = v_Z[src]
            self.vacancies.shell_idx[dst] = v_shell[src]
            self.vacancies.w[dst] = in_w[src]  # inherit weight
            self.vacancies.status[dst] = 1
            # inherit RNG via advanced ctr0 from electron stage
            self.vacancies.rng_offset[dst] = _rng_offset_from_ctr0(e_ctr0[src])

        # kill the parent photons
        self.photons.status[idx64] = 0
        # advance parent photon rng offsets by 1 unit (shell selection)
        self.photons.rng_offset[idx64] += 1


    @torch.no_grad()
    def _phase10_dispatch_relaxation(self, vacancy_active: torch.Tensor):
        Nv = int(vacancy_active.numel())
        if Nv == 0:
            return
        idx64 = vacancy_active.to(torch.int64)
        dev = self.device

        # gather vacancy staging
        in_x = self.vacancies.x[idx64]
        in_y = self.vacancies.y[idx64]
        in_z = self.vacancies.z[idx64]
        in_Z = self.vacancies.atom_Z[idx64]
        in_shell = self.vacancies.shell_idx[idx64]
        in_w = self.vacancies.w[idx64]

        in_key0 = torch.full((Nv,), PHILOX_KEY0, device=dev, dtype=torch.int32)
        in_key1 = torch.full((Nv,), PHILOX_KEY1, device=dev, dtype=torch.int32)
        in_ctr0 = _ctr0_from_rng_offset(self.vacancies.rng_offset[idx64])
        in_ctr1 = torch.zeros((Nv,), device=dev, dtype=torch.int32)
        in_ctr2 = torch.zeros((Nv,), device=dev, dtype=torch.int32)
        in_ctr3 = torch.zeros((Nv,), device=dev, dtype=torch.int32)

        # staging outputs photons/electrons
        phx = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phy = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phz = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phd_x = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phd_y = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phd_z = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phE = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phw = torch.empty((Nv,), device=dev, dtype=torch.float32)
        phk0 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        phk1 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        phc0 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        phc1 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        phc2 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        phc3 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        ph_has = torch.empty((Nv,), device=dev, dtype=torch.int8)

        ex = torch.empty((Nv,), device=dev, dtype=torch.float32)
        ey = torch.empty((Nv,), device=dev, dtype=torch.float32)
        ez = torch.empty((Nv,), device=dev, dtype=torch.float32)
        edx = torch.empty((Nv,), device=dev, dtype=torch.float32)
        edy = torch.empty((Nv,), device=dev, dtype=torch.float32)
        edz = torch.empty((Nv,), device=dev, dtype=torch.float32)
        eE = torch.empty((Nv,), device=dev, dtype=torch.float32)
        ew = torch.empty((Nv,), device=dev, dtype=torch.float32)
        ek0 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        ek1 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        ec0 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        ec1 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        ec2 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        ec3 = torch.empty((Nv,), device=dev, dtype=torch.int32)
        e_has = torch.empty((Nv,), device=dev, dtype=torch.int8)

        # constants
        vx, vy, vz = map(float, self.voxel_size_cm)
        Zmax = int(self.tables.relax_fluor_yield.shape[0] - 1)
        S = int(self.tables.relax_fluor_yield.shape[1])

        grid = (triton.cdiv(Nv, 256),)
        atomic_relaxation_soa_kernel[grid](
            in_x, in_y, in_z,
            in_Z, in_shell, in_w,
            in_key0, in_key1,
            in_ctr0, in_ctr1, in_ctr2, in_ctr3,
            self.tables.relax_fluor_yield,
            self.tables.relax_E_xray_MeV,
            self.tables.relax_E_auger_MeV,
            Zmax=Zmax,
            S=S,
            out_ph_x_ptr=phx, out_ph_y_ptr=phy, out_ph_z_ptr=phz,
            out_ph_dx_ptr=phd_x, out_ph_dy_ptr=phd_y, out_ph_dz_ptr=phd_z,
            out_ph_E_ptr=phE, out_ph_w_ptr=phw,
            out_ph_rng_key0_ptr=phk0, out_ph_rng_key1_ptr=phk1,
            out_ph_rng_ctr0_ptr=phc0, out_ph_rng_ctr1_ptr=phc1, out_ph_rng_ctr2_ptr=phc2, out_ph_rng_ctr3_ptr=phc3,
            out_ph_has_ptr=ph_has,
            out_e_x_ptr=ex, out_e_y_ptr=ey, out_e_z_ptr=ez,
            out_e_dx_ptr=edx, out_e_dy_ptr=edy, out_e_dz_ptr=edz,
            out_e_E_ptr=eE, out_e_w_ptr=ew,
            out_e_rng_key0_ptr=ek0, out_e_rng_key1_ptr=ek1,
            out_e_rng_ctr0_ptr=ec0, out_e_rng_ctr1_ptr=ec1, out_e_rng_ctr2_ptr=ec2, out_e_rng_ctr3_ptr=ec3,
            out_e_has_ptr=e_has,
            edep_flat_ptr=self.edep_flat,
            Zdim=int(self.Z), Ydim=int(self.Y), Xdim=int(self.X),
            voxel_x_cm=vx, voxel_y_cm=vy, voxel_z_cm=vz,
            photon_cut_MeV=float(self.photon_cut_MeV),
            e_cut_MeV=float(self.e_cut_MeV),
            Nv=Nv,
            BLOCK=256,
            num_warps=4,
        )

        # append new photons/electrons to banks (same staging->bank CPU masked scatter as above)
        ph_mask = ph_has.bool()
        if torch.any(ph_mask):
            n_new = int(ph_mask.sum().item())
            base = int(self.global_counters[0].item())
            self.global_counters[0] += n_new
            dst = torch.arange(base, base + n_new, device=dev, dtype=torch.int64)
            src = torch.nonzero(ph_mask, as_tuple=False).view(-1).to(torch.int64)

            self.photons.x[dst] = phx[src]
            self.photons.y[dst] = phy[src]
            self.photons.z[dst] = phz[src]
            self.photons.dx[dst] = phd_x[src]
            self.photons.dy[dst] = phd_y[src]
            self.photons.dz[dst] = phd_z[src]
            self.photons.E[dst] = phE[src]
            self.photons.w[dst] = phw[src]
            self.photons.status[dst] = 1
            # rng_offset from ctr0
            self.photons.rng_offset[dst] = _rng_offset_from_ctr0(phc0[src])

        e_mask = e_has.bool()
        if torch.any(e_mask):
            n_new = int(e_mask.sum().item())
            base = int(self.global_counters[1].item())
            self.global_counters[1] += n_new
            dst = torch.arange(base, base + n_new, device=dev, dtype=torch.int64)
            src = torch.nonzero(e_mask, as_tuple=False).view(-1).to(torch.int64)

            self.electrons.x[dst] = ex[src]
            self.electrons.y[dst] = ey[src]
            self.electrons.z[dst] = ez[src]
            self.electrons.dx[dst] = edx[src]
            self.electrons.dy[dst] = edy[src]
            self.electrons.dz[dst] = edz[src]
            self.electrons.E[dst] = eE[src]
            self.electrons.w[dst] = ew[src]
            self.electrons.status[dst] = 1
            self.electrons.rng_offset[dst] = _rng_offset_from_ctr0(ec0[src])

        # kill processed vacancies
        self.vacancies.status[idx64] = 0
        self.vacancies.rng_offset[idx64] += 1