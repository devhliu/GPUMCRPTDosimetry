from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from ..decaydb.icrp107_json import ICRP107Nuclide


@dataclass
class ParticleQueues:
    photons: Dict[str, torch.Tensor]
    electrons: Dict[str, torch.Tensor]
    positrons: Dict[str, torch.Tensor]


def sample_weighted_decays_and_primaries(
    activity_bqs: torch.Tensor,  # [Z,Y,X] Bq*s per voxel
    voxel_size_cm: Tuple[float, float, float],
    affine,
    nuclide: ICRP107Nuclide,
    n_histories: int,
    seed: int,
    device: str,
    cutoffs: dict,
    sampling_mode: str = "accurate",
) -> tuple[ParticleQueues, torch.Tensor]:
    """Sample primary particles for radioactive decays.

    sampling_mode:
    - "accurate": variable-length particle queues by sampling multiplicities per decay.
      Discrete emissions are sampled from a discrete PDF (line yields) via CDF/inversion;
      continuous beta spectra are sampled via inverse-CDF from a tabulated PDF.

      This aligns with docs/CDF_PDF_MC_suggestion.md: PDFs define physics; sampling uses
      CDF/inversion (no energy summation across categories).

    - "yield_weighted_single": fixed-length queues (one particle per category per history)
      with weights scaled by total yield. This is faster but approximate.

    Notes
    - activity_bqs is treated as Bq*s per voxel (Time-Integrated Activity).
    - Returned particle weights are scaled so tallies are proportional to activity.
    - Positions are in cm with ordering (z, y, x). Directions use the same ordering.
    """

    if n_histories <= 0:
        raise ValueError("n_histories must be > 0")

    a = torch.clamp(activity_bqs.to(device=device, dtype=torch.float32), min=0.0)
    total_decays = torch.sum(a)
    probs = (a / torch.clamp(total_decays, min=1e-30)).reshape(-1)

    # Seed both a per-call generator and the global RNG.
    # torch.poisson does not accept a generator, so we seed the global RNG for determinism.
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    idx = torch.multinomial(probs, num_samples=n_histories, replacement=True, generator=g)
    w_hist = (total_decays / float(n_histories)).to(torch.float32).expand(n_histories)

    Z, Y, X = activity_bqs.shape
    z = idx // (Y * X)
    y = (idx % (Y * X)) // X
    x = idx % X

    u = torch.rand((n_histories, 3), generator=g, device=device, dtype=torch.float32)
    pos_vox = torch.stack([z.to(torch.float32), y.to(torch.float32), x.to(torch.float32)], dim=1) + u
    vx, vy, vz = float(voxel_size_cm[0]), float(voxel_size_cm[1]), float(voxel_size_cm[2])
    pos_cm = pos_vox * torch.tensor([vz, vy, vx], device=device, dtype=torch.float32)

    def sample_isotropic_dirs(n: int) -> torch.Tensor:
        mu = 2.0 * torch.rand((n,), generator=g, device=device, dtype=torch.float32) - 1.0
        phi = 2.0 * torch.pi * torch.rand((n,), generator=g, device=device, dtype=torch.float32)
        sin = torch.sqrt(torch.clamp(1.0 - mu * mu, min=0.0))
        return torch.stack([mu, sin * torch.cos(phi), sin * torch.sin(phi)], dim=1)

    ecut_MeV = float(cutoffs.get("electron_keV", 20.0)) * 1e-3
    gcut_MeV = float(cutoffs.get("photon_keV", 3.0)) * 1e-3

    em = nuclide.emissions or {}

    if sampling_mode not in ("accurate", "yield_weighted_single"):
        raise ValueError(f"Unknown sampling_mode={sampling_mode!r}")

    # ---------------------------------------------------------------------------------
    # Accurate mode: variable-length queues (multiple emissions per decay)
    # ---------------------------------------------------------------------------------
    if sampling_mode == "accurate":
        local_edep = torch.zeros_like(a, dtype=torch.float32, device=device)
        lin = (z * (Y * X) + y * X + x).to(torch.int64)

        def deposit_at_histories(E_MeV: torch.Tensor, hist_idx: torch.Tensor) -> None:
            if E_MeV.numel() == 0:
                return
            local_edep.reshape(-1).index_add_(0, lin[hist_idx], (E_MeV * w_hist[hist_idx]).to(torch.float32))

        def _empty_queue() -> Dict[str, torch.Tensor]:
            return {
                "pos_cm": pos_cm[:0],
                "dir": pos_cm[:0],
                "E_MeV": w_hist[:0],
                "w": w_hist[:0],
            }

        def emit_discrete(*, pairs_obj: Any, cut_MeV: float) -> Dict[str, torch.Tensor]:
            energies, yields = _parse_discrete_pairs(pairs_obj, device=device)
            if energies.numel() == 0:
                return _empty_queue()

            total_yield = torch.sum(yields).to(torch.float32)
            if float(total_yield.item()) <= 0.0:
                return _empty_queue()

            probs_lines = yields / torch.clamp(total_yield, min=1e-12)

            # Total multiplicity per decay ~ Poisson(total_yield)
            rate = torch.full((n_histories,), float(total_yield.item()), device=device, dtype=torch.float32)
            counts = torch.poisson(rate).to(torch.int64)
            total = int(counts.sum().item())
            if total == 0:
                return _empty_queue()

            hist_idx = torch.repeat_interleave(
                torch.arange(n_histories, device=device, dtype=torch.int64), counts
            )
            line_idx = torch.multinomial(probs_lines, num_samples=total, replacement=True, generator=g)

            E = energies[line_idx]
            ww = w_hist[hist_idx]
            dirs = sample_isotropic_dirs(total)
            poss = pos_cm[hist_idx]

            below = (E > 0) & (E < cut_MeV)
            if below.any():
                deposit_at_histories(E[below], hist_idx[below])

            keep = E >= cut_MeV
            return {
                "pos_cm": poss[keep],
                "dir": dirs[keep],
                "E_MeV": E[keep],
                "w": ww[keep],
            }

        def emit_beta(*, spectrum_obj: Any, yield_per_decay: float, cut_MeV: float) -> Dict[str, torch.Tensor]:
            bspec = _parse_beta_spectrum_pairs(spectrum_obj)
            if not bspec or yield_per_decay <= 0.0:
                return _empty_queue()

            rate = torch.full((n_histories,), float(yield_per_decay), device=device, dtype=torch.float32)
            counts = torch.poisson(rate).to(torch.int64)
            total = int(counts.sum().item())
            if total == 0:
                return _empty_queue()

            hist_idx = torch.repeat_interleave(
                torch.arange(n_histories, device=device, dtype=torch.int64), counts
            )
            E = _sample_beta_pdf_from_bspectra(bspec, total, g, device)
            ww = w_hist[hist_idx]
            dirs = sample_isotropic_dirs(total)
            poss = pos_cm[hist_idx]

            below = (E > 0) & (E < cut_MeV)
            if below.any():
                deposit_at_histories(E[below], hist_idx[below])

            keep = E >= cut_MeV
            return {
                "pos_cm": poss[keep],
                "dir": dirs[keep],
                "E_MeV": E[keep],
                "w": ww[keep],
            }

        # Photons (gamma + X)
        photons = emit_discrete(pairs_obj=list(em.get("gamma", [])) + list(em.get("X", [])), cut_MeV=gcut_MeV)

        # Electrons (beta- + discrete conversion/Auger)
        beta_minus_obj = em.get("b-spectra", em.get("beta-", em.get("beta_minus", [])))
        beta_minus_present = bool(beta_minus_obj)
        beta_minus_yield = float(em.get("beta_minus_yield", 1.0 if beta_minus_present else 0.0))
        beta_minus = emit_beta(spectrum_obj=beta_minus_obj, yield_per_decay=beta_minus_yield, cut_MeV=ecut_MeV)

        elec_disc = emit_discrete(
            pairs_obj=list(em.get("IE", [])) + list(em.get("auger", [])),
            cut_MeV=ecut_MeV,
        )

        if beta_minus["E_MeV"].numel() == 0:
            electrons = elec_disc
        elif elec_disc["E_MeV"].numel() == 0:
            electrons = beta_minus
        else:
            electrons = {
                "pos_cm": torch.cat([beta_minus["pos_cm"], elec_disc["pos_cm"]], dim=0),
                "dir": torch.cat([beta_minus["dir"], elec_disc["dir"]], dim=0),
                "E_MeV": torch.cat([beta_minus["E_MeV"], elec_disc["E_MeV"]], dim=0),
                "w": torch.cat([beta_minus["w"], elec_disc["w"]], dim=0),
            }

        # Positrons (beta+) if present
        beta_plus_obj = em.get("b+spectra", em.get("beta+", em.get("beta_plus", [])))
        beta_plus_present = bool(beta_plus_obj)
        beta_plus_yield = float(em.get("beta_plus_yield", 1.0 if beta_plus_present else 0.0))
        positrons = emit_beta(spectrum_obj=beta_plus_obj, yield_per_decay=beta_plus_yield, cut_MeV=ecut_MeV)

        # Alpha always local deposition (Milestone scope)
        alpha_E_expected = _expected_discrete_energy_MeV(em.get("alpha", []), device=device)
        if float(alpha_E_expected.item()) > 0.0:
            local_edep.view(-1).index_add_(0, lin, (alpha_E_expected * w_hist).to(torch.float32))

        return ParticleQueues(photons=photons, electrons=electrons, positrons=positrons), local_edep

    # ---------------------------------------------------------------------------------
    # Fast approximation: yield-weighted single-particle mode
    # ---------------------------------------------------------------------------------

    local_edep = torch.zeros_like(a, dtype=torch.float32, device=device)

    def deposit_local(E: torch.Tensor, ww: torch.Tensor) -> None:
        if E.numel() == 0:
            return
        lin = (z * (Y * X) + y * X + x).to(torch.int64)
        local_edep.view(-1).index_add_(0, lin, (E * ww).to(torch.float32))

    def deposit_local_if_below_cut(E: torch.Tensor, ww: torch.Tensor, cut: float) -> None:
        below = (E > 0) & (E < cut)
        if below.any():
            lin = (z * (Y * X) + y * X + x).to(torch.int64)
            local_edep.view(-1).index_add_(0, lin[below], (E[below] * ww[below]).to(torch.float32))

    # Photons: combine gamma + X into one discrete sampler (one photon per history)
    photon_pairs = list(em.get("gamma", [])) + list(em.get("X", []))
    photon_E, photon_yield = _sample_discrete_pairs_MeV_yieldweighted(photon_pairs, n_histories, g, device)
    photon_w = w_hist * photon_yield

    # Electrons: mixture of beta- (continuous) and discrete conversion/Auger
    beta_minus_obj = em.get("b-spectra", em.get("beta-", em.get("beta_minus", [])))
    beta_minus_present = bool(beta_minus_obj)
    beta_minus_yield = torch.full(
        (n_histories,), 1.0 if beta_minus_present else 0.0, device=device, dtype=torch.float32
    )

    elec_pairs = list(em.get("IE", [])) + list(em.get("auger", []))
    elec_disc_E, elec_disc_yield = _sample_discrete_pairs_MeV_yieldweighted(elec_pairs, n_histories, g, device)

    elec_total_yield = beta_minus_yield + elec_disc_yield
    u_mix = torch.rand((n_histories,), generator=g, device=device, dtype=torch.float32)
    p_beta = beta_minus_yield / torch.clamp(elec_total_yield, min=1e-12)
    pick_beta = (u_mix < p_beta) & (beta_minus_yield > 0)

    beta_minus_E = _sample_beta_pdf_from_bspectra(_parse_beta_spectrum_pairs(beta_minus_obj), n_histories, g, device)
    electron_E = torch.where(pick_beta, beta_minus_E, elec_disc_E)
    electron_w = w_hist * elec_total_yield

    # Positrons: beta+ if present
    beta_plus_obj = em.get("b+spectra", em.get("beta+", em.get("beta_plus", [])))
    beta_plus_present = bool(beta_plus_obj)
    positron_yield = torch.full(
        (n_histories,), 1.0 if beta_plus_present else 0.0, device=device, dtype=torch.float32
    )
    positron_E = (
        _sample_beta_pdf_from_bspectra(_parse_beta_spectrum_pairs(beta_plus_obj), n_histories, g, device)
        if beta_plus_present
        else torch.zeros((n_histories,), device=device, dtype=torch.float32)
    )
    positron_w = w_hist * positron_yield

    # Alpha always local deposition
    alpha_E_expected = _expected_discrete_energy_MeV(em.get("alpha", []), device=device)
    if float(alpha_E_expected.item()) > 0.0:
        deposit_local(alpha_E_expected.expand(n_histories), w_hist)

    # Below-cutoff deposit locally for energy conservation
    deposit_local_if_below_cut(photon_E, photon_w, gcut_MeV)
    deposit_local_if_below_cut(electron_E, electron_w, ecut_MeV)
    deposit_local_if_below_cut(positron_E, positron_w, ecut_MeV)

    # Keep particles above cut; fixed-length queues by zeroing below-cut
    photons_E = torch.where(photon_E >= gcut_MeV, photon_E, torch.zeros_like(photon_E))
    photons_w = torch.where(photon_E >= gcut_MeV, photon_w, torch.zeros_like(photon_w))

    electrons_E = torch.where(electron_E >= ecut_MeV, electron_E, torch.zeros_like(electron_E))
    electrons_w = torch.where(electron_E >= ecut_MeV, electron_w, torch.zeros_like(electron_w))

    positrons_E = torch.where(positron_E >= ecut_MeV, positron_E, torch.zeros_like(positron_E))
    positrons_w = torch.where(positron_E >= ecut_MeV, positron_w, torch.zeros_like(positron_w))

    eQ = {"pos_cm": pos_cm, "dir": sample_isotropic_dirs(n_histories), "E_MeV": electrons_E, "w": electrons_w}
    pQ = {"pos_cm": pos_cm, "dir": sample_isotropic_dirs(n_histories), "E_MeV": photons_E, "w": photons_w}
    posQ = {"pos_cm": pos_cm, "dir": sample_isotropic_dirs(n_histories), "E_MeV": positrons_E, "w": positrons_w}

    return ParticleQueues(photons=pQ, electrons=eQ, positrons=posQ), local_edep


def _parse_discrete_pairs(pairs_obj: Any, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a discrete emission list into (energies_MeV, yields).

    Supports common ICRP107/OpenGATE JSON encodings:
    - [[E, y], ...]
    - [{"energy": E, "yield": y}, ...] (also accepts keys: E/intensity/probability)
    """

    if pairs_obj is None:
        return (
            torch.zeros((0,), device=device, dtype=torch.float32),
            torch.zeros((0,), device=device, dtype=torch.float32),
        )

    if isinstance(pairs_obj, (list, tuple)):
        if len(pairs_obj) == 0:
            return (
                torch.zeros((0,), device=device, dtype=torch.float32),
                torch.zeros((0,), device=device, dtype=torch.float32),
            )

        first = pairs_obj[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            energies = torch.tensor([float(p[0]) for p in pairs_obj], device=device, dtype=torch.float32)
            yields = torch.tensor([float(p[1]) for p in pairs_obj], device=device, dtype=torch.float32)
            return energies, torch.clamp(yields, min=0.0)

        if isinstance(first, dict):
            energies_list = []
            yields_list = []
            for rec in pairs_obj:
                e = rec.get("energy", rec.get("E", rec.get("energy_MeV", None)))
                y = rec.get("yield", rec.get("intensity", rec.get("probability", rec.get("p", None))))
                if e is None or y is None:
                    continue
                energies_list.append(float(e))
                yields_list.append(float(y))

            if not energies_list:
                return (
                    torch.zeros((0,), device=device, dtype=torch.float32),
                    torch.zeros((0,), device=device, dtype=torch.float32),
                )

            energies = torch.tensor(energies_list, device=device, dtype=torch.float32)
            yields = torch.tensor(yields_list, device=device, dtype=torch.float32)
            return energies, torch.clamp(yields, min=0.0)

    return (
        torch.zeros((0,), device=device, dtype=torch.float32),
        torch.zeros((0,), device=device, dtype=torch.float32),
    )


def _expected_discrete_energy_MeV(pairs_obj: Any, *, device: str) -> torch.Tensor:
    energies, yields = _parse_discrete_pairs(pairs_obj, device=device)
    if energies.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    return torch.sum(energies * yields).to(torch.float32)


def _parse_beta_spectrum_pairs(spectrum_obj: Any) -> list[list[float]]:
    """Normalize a beta spectrum object into a list of [E_MeV, pdf] pairs."""

    if spectrum_obj is None:
        return []

    if isinstance(spectrum_obj, (list, tuple)):
        if len(spectrum_obj) == 0:
            return []
        if isinstance(spectrum_obj[0], (list, tuple)) and len(spectrum_obj[0]) >= 2:
            return [[float(p[0]), float(p[1])] for p in spectrum_obj]

    if isinstance(spectrum_obj, dict):
        if "spectrum" in spectrum_obj:
            sp = spectrum_obj.get("spectrum")
            if isinstance(sp, (list, tuple)):
                return [
                    [float(p[0]), float(p[1])]
                    for p in sp
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ]

        if "E" in spectrum_obj and "pdf" in spectrum_obj:
            Es = spectrum_obj.get("E")
            ps = spectrum_obj.get("pdf")
            if isinstance(Es, (list, tuple)) and isinstance(ps, (list, tuple)) and len(Es) == len(ps):
                return [[float(e), float(p)] for e, p in zip(Es, ps)]

    return []


def _sample_discrete_pairs_MeV_yieldweighted(
    pairs: Any, n: int, g: torch.Generator, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one discrete line energy and return the total yield.

    Returns:
    - E: sampled line energy in MeV (0 if total_yield == 0)
    - total_yield: sum of yields (per decay). Caller typically scales particle weights by this.

    This matches PDF/CDF guidance: yields define a discrete PDF; its CDF is used for sampling.
    """

    energies, yields = _parse_discrete_pairs(pairs, device=device)
    if energies.numel() == 0:
        return (
            torch.zeros((n,), device=device, dtype=torch.float32),
            torch.zeros((n,), device=device, dtype=torch.float32),
        )

    total = torch.sum(yields)
    if float(total.item()) <= 0.0:
        return (
            torch.zeros((n,), device=device, dtype=torch.float32),
            torch.zeros((n,), device=device, dtype=torch.float32),
        )

    probs = yields / torch.clamp(total, min=1e-12)
    idx = torch.multinomial(probs, num_samples=n, replacement=True, generator=g)
    E = energies[idx]
    total_yield = torch.full((n,), float(total.item()), device=device, dtype=torch.float32)
    return E, total_yield


def _sample_beta_pdf_from_bspectra(bspec: list[list[float]], n: int, g: torch.Generator, device: str) -> torch.Tensor:
    """Sample beta energy from a tabulated PDF via inverse-CDF.

    bspec: [[E_MeV, pdf_value], ...] treated as a PDF shape.
    """

    if not bspec:
        return torch.zeros((n,), device=device, dtype=torch.float32)

    E = torch.tensor([float(p[0]) for p in bspec], device=device, dtype=torch.float32)
    pdf = torch.tensor([float(p[1]) for p in bspec], device=device, dtype=torch.float32)
    pdf = torch.clamp(pdf, min=0.0)

    if E.numel() >= 2:
        order = torch.argsort(E)
        E = E[order]
        pdf = pdf[order]

    if E.numel() < 2:
        return E.expand(n).clone()

    dE = E[1:] - E[:-1]
    area = 0.5 * (pdf[1:] + pdf[:-1]) * torch.clamp(dE, min=1e-12)
    cdf = torch.cat([torch.zeros((1,), device=device, dtype=torch.float32), torch.cumsum(area, dim=0)], dim=0)
    cdf = cdf / torch.clamp(cdf[-1], min=1e-12)

    u = torch.rand((n,), generator=g, device=device, dtype=torch.float32)
    idx = torch.bucketize(u, cdf)
    idx = torch.clamp(idx, 1, len(cdf) - 1)

    c0 = cdf[idx - 1]
    c1 = cdf[idx]
    e0 = E[idx - 1]
    e1 = E[idx]
    t = (u - c0) / torch.clamp(c1 - c0, min=1e-8)
    return e0 + t * (e1 - e0)
