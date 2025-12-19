# Physics Implementation

This document details the implementation of the physics models used in the GPUMCRPTDosimetry simulation. The goal is to accurately model the transport of photons, electrons, and positrons through matter, and their interactions with it.

## Photon Physics

Photon transport is a cornerstone of the simulation. The following physics processes are included:

### Photon Transport: Woodcock Tracking

Instead of using traditional ray-tracing through a complex geometry, the simulation employs **Woodcock tracking** (also known as the delta-tracking or pseudo-collision method). This is a variance-reduction technique that is well-suited for GPU implementation.

*   The entire simulation volume is treated as having a single, maximum interaction cross-section (Σ_max).
*   The distance to the next potential interaction is sampled from an exponential distribution with mean free path 1/Σ_max.
*   At the potential interaction site, a "virtual" vs. "real" collision decision is made by comparing the actual material cross-section at that point (Σ(E, material)) to Σ_max.
*   If the collision is "real" (with probability Σ(E, material) / Σ_max), an interaction is simulated.
*   If it is "virtual", the photon continues along its path without changing energy or direction.

This method avoids the need for complex geometric tracking and boundary crossings, which are difficult to implement efficiently on the GPU.

### Photon Interactions

When a "real" collision occurs, the type of interaction is sampled based on the relative probabilities of the following processes:

1.  **Compton Scattering:** An inelastic scatter of a photon by an atomic electron.
    *   The energy of the scattered photon and the recoil electron are calculated using the Klein-Nishina formula.
    *   The scattering angle is sampled from pre-computed inverse CDF tables, which store `cos(theta)`. The convention is explicitly defined as:
        ```
        E' = E / (1 + (E / m_e*c^2) * (1 - cos(theta)))
        ```
    *   The recoil electron's kinetic energy (T = E - E') is either deposited locally or the electron is added to the electron queue for transport.

2.  **Photoelectric Effect:** The photon is absorbed by an atom, and an electron (a photoelectron) is ejected.
    *   In simpler models, the full photon energy is deposited locally.
    *   More advanced engines (`EM–BankSoAVacancyRelaxation (Phase 10)`) track the **atomic vacancy** left by the ejected electron. This vacancy then relaxes through a cascade of characteristic X-rays and Auger electrons, which are added to the particle queues for transport. This is crucial for accurate dosimetry, especially in high-Z materials.

3.  **Rayleigh Scattering:** A coherent scattering process where the photon's direction changes but its energy is conserved.

4.  **Pair Production:** A high-energy photon (E > 1.022 MeV) interacts with the electric field of a nucleus and is converted into an electron-positron pair.
    *   In simpler models, the photon energy is deposited locally.
    *   In more advanced models, the electron and positron are added to their respective queues for transport.

## Charged Particle Physics (Electrons and Positrons)

The transport of charged particles is simulated using a **condensed-history** approach. This is an approximation where the cumulative effect of many small interactions over a path segment is modeled as a single "step".

### Condensed-History Stepping

For each step, the following are calculated:

*   **Energy Loss:** The continuous energy loss due to ionization and excitation is calculated using restricted stopping power data.
*   **Multiple Scattering:** The angular deflection of the particle is sampled from a distribution that models the cumulative effect of many small-angle scatters (e.g., using Highland's formula).
*   **Hard Interactions:** Discrete, large-energy-transfer events are simulated explicitly:
    *   **Bremsstrahlung:** The emission of a photon by an electron or positron in the electric field of a nucleus.
    *   **Delta-ray production:** The production of a high-energy secondary electron.

A **secondary particle budget** is used to control the generation of these "hard" secondaries to prevent an uncontrolled explosion in the number of particles.

### Positron Annihilation

When a positron's energy drops below a certain cutoff, it is considered to have stopped. It then annihilates with an electron, producing two **511 keV photons** that are emitted back-to-back. These photons are then added to the photon queue for transport.
