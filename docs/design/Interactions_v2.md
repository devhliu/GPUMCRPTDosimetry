This report summarizes particle interactions with human tissues, separated into **General Medical Applications** (Diagnostic Radiology & External Beam Radiotherapy) and **Radionuclide Pharmacutical Therapy (RPT)**. The data reflects current medical physics consensus (ICRU/AAPM standards) for the year 2025.

---

# 1. All Types of Interactions with Human Tissue

## 1.1 Photon Interactions
*Photons are indirectly ionizing. Their interaction probabilities depend heavily on the tissue effective atomic number ($Z_{eff} \approx 7.4$) and photon energy.*

| Primary Particle | Interaction Type | Description of Interaction | Immediate Secondary Particles | Cascaded & Downstream Particles | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Photon ($\gamma$)** | **Photoelectric Effect** | Complete absorption of photon by inner-shell electron (K-shell). Energy overcomes binding; excess becomes kinetic. | **Photoelectron** (e⁻) | **Auger Electrons** (Relaxation)<br>**Characteristic X-rays**<br>**Bremsstrahlung** (from photoelectron) | Dominant at low energies (<30 keV) |
| **Photon ($\gamma$)** | **Compton Scattering** | Collision with outer-shell electron. Photon scatters with reduced energy; electron recoils. | **Compton Electron** (Recoil e⁻)<br>**Scattered Photon** ($\gamma'$) | **Delta Rays** (Secondary Ionizations)<br>**Bremsstrahlung**<br>**Scattered Photons** (Noise) | Dominant in therapy range (0.1-25 MeV) |
| **Photon ($\gamma$)** | **Pair Production** | Interaction with nuclear field converts photon into matter/antimatter ($E > 1.022$ MeV). | **Electron** (e⁻)<br>**Positron** (e⁺) | **Annihilation Photons** (2 $\gamma$)<br>**Bremsstrahlung** | Significant above 10 MeV |
| **Photon ($\gamma$)** | **Rayleigh Scattering** | Coherent scattering by the whole atom. No ionization or energy loss. | **Scattered Photon** ($\gamma$) | None | Very weak in human tissue |
| **Photon ($\gamma$)** | **Photonuclear** | Absorption by nucleus; nucleon ejection ($E > 8\text{-}10$ MeV). | **Neutron** (n)<br>**Proton** (p⁺) | **Radioactive Isotopes**<br>**Beta/Gamma** (from decay) | Very weak in human tissue (<1% dose) |
| **Photon ($\gamma$)** | **Thomson Scattering** | The low-energy limit of Rayleigh/Compton scattering involving free electrons (classical limit). The photon scatters without energy loss. | **Scattered Photon** | None | Very weak in human tissue - theoretical |
| **Photon ($\gamma$)** | **Delbrück Scattering** | The scattering of a photon by the Coulomb field of a nucleus via the creation and annihilation of virtual electron-positron pairs. | **Scattered Photon** | None | Very weak in human tissue - high energy physics |
| **Photon ($\gamma$)** | **Nuclear Resonance Fluorescence (NRF)** | The photon is absorbed by a nucleus, exciting it to a specific nuclear level, which then de-excites by emitting a photon. | **Re-emitted Photon** | None | Very weak in human tissue - isotope detection |
| **Photon ($\gamma$)** | **Double Compton Scattering** | A higher-order effect where a photon interacts with an electron and emits *two* photons instead of one. | **Recoil Electron**<br>**Two Scattered Photons** | None | Very weak in human tissue - negligible |

## 1.2 Electron and Positron Interactions
*Electrons and Positrons are directly ionizing.*

| Primary Particle | Interaction Type | Description of Interaction | Immediate Secondary Particles | Cascaded & Downstream Particles | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Electron (e⁻)** | **Inelastic Collisions** | Coulomb interaction with atomic electrons (Ionization/Excitation). | **Delta Rays** (Secondary e⁻)<br>**Ejected Electrons** | **Auger Electrons**<br>**Characteristic X-rays**<br>**Heat** | Primary dose deposition mechanism |
| **Electron (e⁻)** | **Bremsstrahlung** | Radiative braking near nucleus. Yield increases with Energy ($E$) and Atomic Number ($Z$). | **Bremsstrahlung Photons** (X-rays) | **Photoelectrons**<br>**Compton Electrons** | <1% yield in human tissue |
| **Electron (e⁻)** | **Cerenkov Radiation** | Occurs when the charged particle travels faster through a medium (tissue/water) than the phase velocity of light in that medium ($v > c/n$). | **Cerenkov Photons**<br>*(UV / Visible Blue Light)* | None | Very weak in human tissue - no dose contribution (<0.1%) |
| **Electron (e⁻)** | **Transition Radiation** | Emitted when a relativistic charged particle crosses the boundary between two materials with different dielectric constants (e.g., bone to soft tissue). | **Soft X-rays** | None | Very weak in human tissue - interface effect |
| **Positron (e⁺)** | **Annihilation** | Positron stops, combines with electron; mass $\to$ energy. | **Two 511 keV Photons** | **Compton/Photoelectrons** (from photon interaction) | Standard annihilation mode |
| **Positron (e⁺)** | **Positronium Formation (Para-Positronium)** | The positron forms a bound state with an electron (like a hydrogen atom) with spins *antiparallel* (Singlet state). Short lifetime (~125 ps). | **Two 511 keV Photons** | None | Special case - 25% probability |
| **Positron (e⁺)** | **Positronium Formation (Ortho-Positronium)** | The positron forms a bound state with an electron with spins *parallel* (Triplet state). Longer lifetime (~142 ns in vacuum). | **Three Photons**<br>*(Total energy = 1.022 MeV)* | None | Special case - 75% probability, PET artifacts |
| **Positron (e⁺)** | **In-flight Annihilation** | The positron annihilates with an electron *before* coming to rest. | **Two Photons**<br>*(Energies $\neq$ 511 keV)* | None | Special case - spectrum background |

## 1.3 Atomic Relaxation Interactions
*Applicable to all internal emitters where vacancies are created.*

| Primary Event | Interaction Type | Description | Immediate Secondary Particles | Cascaded & Downstream Particles | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vacancy Creation**<br>*(via EC, IC, or Photoelectric)* | **Auger / Coster-Kronig** | Non-radiative transition filling inner shells. | **Auger Electron** | **Vacancy Cascade**<br>**Coulomb Explosion** | Dominant for low-Z atoms |
| **Inner Shell Vacancy** | **Radiative Auger Effect (RAE)** | A "hybrid" transition where an electron fills a vacancy, but instead of emitting *either* a photon *or* an electron, it emits a photon *and* simultaneously excites another electron. | **Photon** (lower energy)<br>**Excited Electron** | None | Very weak in human tissue - spectroscopy |
| **Inner Shell Vacancy** | **Double Auger Effect** | One electron fills a vacancy, and the energy release ejects *two* electrons simultaneously (3-body interaction). | **Two Auger Electrons** | None | Very weak in human tissue - rare |
| **Orbital Electrons** | **Shake-off Effect** | Sudden change in nuclear charge (e.g., during Beta decay or Alpha emission) changes the atomic potential so rapidly that orbital electrons are "shaken" loose. | **Shake-off Electrons**<br>*(Low energy)* | None | Special case - increases local ionization |
| **Orbital Electrons** | **Shake-up Effect** | Similar to Shake-off, but the electron is not ejected; it is excited to a higher shell. | **Excited Atom** | **Subsequent Photons**<br>Relaxation emits low-energy optical/UV photons | Very weak in human tissue |

## 1.4 Alpha Particle Interactions
*High-LET particles (e.g., Ac-225, Ra-223, Pb-212).*

| Primary Particle | Interaction Type | Description of Interaction | Immediate Secondary Particles | Cascaded & Downstream Particles | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alpha ($\alpha$)** | **Inelastic Coulomb (High LET)** | Dense ionization along a straight track ($\sim 40\text{-}80 \mu m$). | **High-Density Delta Rays** | **Double-Strand DNA Breaks**<br>**Reactive Oxygen Species** | Primary dose deposition for alphas |
| **Recoil Nucleus**<br>*(e.g., Bi-213, Tl-208)* | **Nuclear Recoil** | Conservation of momentum kicks the daughter nucleus with ~100 keV energy. | **Ionized Daughter Atom** | **Local Ionization Spike**<br>**Chemical Bond Rupture** | Special case - daughter redistribution |
| **Alpha ($\alpha$)** | **Charge Exchange** | Capture of tissue electrons near end-of-track. | None (Alpha becomes He) | **Fluorescence** (minor) | Very weak in human tissue |

## 1.5 Nuclear Decay Interactions
*Advanced nuclear processes during decay.*

| Primary Event | Interaction Type | Description | Secondary / Cascaded Particles | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Nuclear Decay** | **Internal Pair Formation (IPF)** | An excited nucleus (with excitation energy > 1.022 MeV) de-excites not by emitting a Gamma ray, but by ejecting an e⁻/e⁺ pair directly. | **Electron**<br>**Positron** | Special case - alternative to gamma emission |

---

# 2. Interactions Considered for General Medical Applications
**Energy Range:** 10 keV – 25 MeV
**Context:** External Beam Radiotherapy (Linacs, Protons), Diagnostic Imaging (CT, Mammography, PET), and Radiation Protection.

## 2.1 Photon Interactions (External Beams & Diagnostics)

| Primary Particle | Interaction Type | Importance for Dosimetry & Imaging |
| :--- | :--- | :--- |
| **Photon ($\gamma$)** | **Photoelectric Effect** | **Dominant in Diagnostics (<30 keV)**<br>Produces high image contrast (bone vs. tissue). In therapy, irrelevant except for superficial kv X-rays. High local dose deposition. |
| **Photon ($\gamma$)** | **Compton Scattering** | **Dominant in Therapy (0.1 – 25 MeV)**<br>Primary mechanism of dose deposition in Linac therapy. Determines "skin sparing" and depth-dose profiles. Reduces image contrast in MV imaging. |
| **Photon ($\gamma$)** | **Pair Production** | **High Energy Therapy (>10 MV)**<br>Contributes to dose at depth. Positron range broadens the beam penumbra. |
| **Photon ($\gamma$)** | **Rayleigh Scattering** | **Imaging Artifacts**<br>No dose contribution. Degrades image quality (fog) in mammography and CT. |
| **Photon ($\gamma$)** | **Photonuclear** | **Radiation Protection**<br>Requires neutron shielding for Linac vaults (>10 MV). Minimal patient dose contribution (<1%). |

## 2.2 Charged Particle Interactions (External Beams)

| Primary Particle | Interaction Type | Importance for Dosimetry & Imaging |
| :--- | :--- | :--- |
| **Electron (e⁻)** | **Inelastic Collisions** | **The Definition of "Dose"**<br>Absorbed dose ($D$) is essentially the energy deposited by these collisions. |
| **Electron (e⁻)** | **Bremsstrahlung** | **Beam Generation**<br>Used to create the beam in the Linac target (Tungsten). Inside the patient (tissue), yield is <1%. |
| **Positron (e⁺)** | **Annihilation** | **PET Imaging**<br>Basis of functional imaging. In therapy beams (>10 MV), these photons transport energy away from the interaction site. |

---

# 3. Interactions Considered for External Beam Radiotherapy
**Energy Range:** 10 keV – 25 MeV
**Context:** External Beam Radiotherapy (Linacs, Protons)

## 3.1 Primary Interactions

| Primary Particle | Interaction Type | Role in External Beam Therapy |
| :--- | :--- | :--- |
| **Photon ($\gamma$)** | **Compton Scattering** | Primary mechanism of dose deposition in Linac therapy. Determines "skin sparing" and depth-dose profiles. |
| **Photon ($\gamma$)** | **Pair Production** | Contributes to dose at depth in high energy therapy (>10 MV). Positron range broadens the beam penumbra. |
| **Photon ($\gamma$)** | **Photoelectric Effect** | Relevant only for superficial kv X-rays. High local dose deposition. |
| **Electron (e⁻)** | **Inelastic Collisions** | The definition of "dose" - absorbed dose ($D$) is essentially the energy deposited by these collisions. |
| **Electron (e⁻)** | **Bremsstrahlung** | Used to create the beam in the Linac target (Tungsten). Inside the patient (tissue), yield is <1%. |
| **Positron (e⁺)** | **Annihilation** | In therapy beams (>10 MV), these photons transport energy away from the interaction site. |

## 3.2 Secondary and Specialized Interactions

| Primary Particle | Interaction Type | Role in External Beam Therapy |
| :--- | :--- | :--- |
| **Photon ($\gamma$)** | **Rayleigh Scattering** | No dose contribution. Degrades image quality in MV imaging. |
| **Photon ($\gamma$)** | **Photonuclear** | Requires neutron shielding for Linac vaults (>10 MV). Minimal patient dose contribution (<1%). |
| **Electron (e⁻)** | **Cerenkov Radiation** | Patients undergoing high-energy therapy often see "blue flashes" (Cerenkov in the eye). Used for dosimetry imaging (Cerenkov Luminescence Imaging). No dose contribution (energy loss is <0.1%). |
| **Electron (e⁻)** | **Transition Radiation** | Minor effect in medical energies. More relevant for particle detectors. |

---

# 4. Interactions Considered for Radionuclide Pharmacutical Therapy (RPT)
**Energy Range:** 10 keV – 10 MeV
**Context:** Targeted Alpha Therapy (TAT), Beta-emitters (Lu-177, Y-90), Theranostics, Internal Dosimetry (MIRD).

## 4.1 Photon Interactions (Internal Sources)
*In RPT, photons are often considered "wasted energy" regarding the tumor, but "critical" for verification.*

| Primary Particle | Interaction Type | Role for Radionuclide Dosimetry |
| :--- | :--- | :--- |
| **Gamma ($\gamma$)**<br>*(e.g., from I-131, Lu-177)* | **Compton Scattering** | **Cross-Fire & Toxicity**<br>Gammas travel far ($cm$ to $m$). They contribute minimal dose to the small tumor but irradiate healthy organs (marrow/kidneys). |
| **Gamma ($\gamma$)** | **Photoelectric Effect** | **Imaging Signal**<br>Unhindered photons leaving the body are detected by SPECT/gamma cameras to calculate the dose map. |

## 4.2 Beta and Auger Electron Interactions (Internal Sources)
*The primary drivers of tumor control in standard molecular radiotherapy.*

| Primary Particle | Interaction Type | Role for Radionuclide Dosimetry |
| :--- | :--- | :--- |
| **Beta ($\beta^-$)**<br>*(e.g., Y-90, Lu-177)* | **Inelastic Collisions** | **Cross-Fire Dose**<br>Betas travel mm to cm. They kill cells adjacent to the target cell (beneficial for heterogeneous tumor antigen expression). |
| **Beta ($\beta^-$)** | **Bremsstrahlung** | **Surrogate Imaging**<br>For pure beta emitters (Y-90), this is the *only* way to image the distribution (Bremsstrahlung SPECT). |
| **Auger Electrons**<br>*(from EC/IC decay)* | **Cascaded Ionization** | **Self-Dose (High RBE)**<br>Range is nm (sub-cellular). If isotope is in DNA (e.g., I-125), RBE is high (like Alphas). If in cytoplasm, effect is negligible. |

## 4.3 Alpha Particle Interactions (Targeted Alpha Therapy)
*High-LET particles (e.g., Ac-225, Ra-223, Pb-212).*

| Primary Particle | Interaction Type | Role for Radionuclide Dosimetry |
| :--- | :--- | :--- |
| **Alpha ($\alpha$)** | **Inelastic Coulomb (High LET)** | **Biological Dose Dominance**<br>High RBE (3–7+). Kills cells effectively with 1–2 hits. Overcomes hypoxia and radiation resistance. Dose is confined to the target micro-environment. |
| **Recoil Nucleus**<br>*(e.g., Bi-213, Tl-208)* | **Nuclear Recoil** | **Daughter Redistribution**<br>The recoil energy breaks the chelator bond. The radioactive daughter becomes "free," potentially migrating to healthy organs (e.g., kidney toxicity in Ac-225 generators). |
| **Alpha ($\alpha$)** | **Charge Exchange** | **Track End**<br>Determines the precise range (Bragg Peak) where maximum biological damage occurs. |

## 4.4 Atomic Relaxation (The Cascade Source)
*Applicable to all internal emitters where vacancies are created.*

| Primary Event | Interaction Type | Role for Radionuclide Dosimetry |
| :--- | :--- | :--- |
| **Vacancy Creation**<br>*(via EC, IC, or Photoelectric)* | **Auger / Coster-Kronig** | **Nanodosimetry**<br>Leaves the residual atom with a high positive charge, causing molecular fragmentation (Coulomb Explosion). Crucial for DNA-intercalating isotopes. |

## 4.5 Advanced Nuclear Effects in RPT

| Primary Event | Interaction Type | Role for Radionuclide Dosimetry |
| :--- | :--- | :--- |
| **Orbital Electrons** | **Shake-off Effect** | Adds extra low-energy electrons to the decay spectrum of isotopes like Lu-177 or Y-90. Increases local ionization density. |
| **Orbital Electrons** | **Shake-up Effect** | Subsequent relaxation emits low-energy optical/UV photons. |
| **Inner Shell Vacancy** | **Radiative Auger Effect (RAE)** | Manifests as "satellite lines" in X-ray spectra. Rare compared to standard Auger. |
| **Inner Shell Vacancy** | **Double Auger Effect** | Increases the ionization state of the residual atom faster than standard single Auger steps. |
| **Nuclear Decay** | **Internal Pair Formation (IPF)** | Alternative to Gamma emission (Internal Conversion is the alternative for low energy). Relevant for high energy beta dosimetry. |

## 4.6 Positron Effects in RPT

| Primary Particle | Interaction Type | Role for Radionuclide Dosimetry |
| :--- | :--- | :--- |
| **Positron (e⁺)** | **Positronium Formation (Para-Positronium)** | The standard annihilation mode. Basis of PET imaging. |
| **Positron (e⁺)** | **Positronium Formation (Ortho-Positronium)** | Decays into 3 photons instead of 2. Can reduce PET image quality if not filtered, as it violates the "back-to-back" coincidence logic. |
| **Positron (e⁺)** | **In-flight Annihilation** | Produces photons with energies >511 keV or <511 keV, contributing to background noise in PET. |

---

# 5. Solutions to Avoid Double Counting

This analysis addresses a critical challenge in computational dosimetry (Monte Carlo simulations and Deterministic algorithms): **Energy Conservation**.

Double counting occurs when a simulation accounts for energy deposition *implicitly* (via approximations or pre-calculated spectra) and then accounts for it again *explicitly* (by transporting the secondary particles). This is particularly prone to errors in **Radionuclide Therapy (RPT)** involving Auger cascades and in **Interface Dosimetry** (e.g., tissue-bone or tissue-air).

## 5.1 The "Source Definition" Problem (Radionuclide Therapy)

**The Trap:** Radioactive decay databases (e.g., ICRP 107, MIRD) often list *every* particle emitted, including the full cascade of Auger electrons. However, advanced Monte Carlo codes (Geant4, MCNP, EGSnrc) often have internal physics models that *generate* these cascades dynamically when a vacancy is created.

**The Error:** If you input a full MIRD spectrum *and* enable "Atomic De-excitation" in the physics list, the code will simulate the input Augers, AND then generate *new* Augers when the input particles interact, or if the initial vacancy was simulated twice.

**Solutions:**

### Method A: The "Physics-Driven" Approach (Recommended for Microdosimetry)
- **Input:** Define the source only as the parent radionuclide (ion) or the primary decay event (e.g., the initial electron capture).
- **Physics Setting:** Enable `Atomic De-excitation`, `Fluorescence`, and `Auger Cascade` in the simulation engine.
- **Result:** The code calculates the probability of vacancies and generates the secondary/cascaded particles on the fly.
- **Benefit:** Preserves correct angular correlations and spatial origin of the cascade.

### Method B: The "Spectrum-Driven" Approach (Recommended for Macrodosimetry)
- **Input:** Use the full detailed spectrum (e.g., ICRP 107) listing all photons and electrons.
- **Physics Setting:** **DISABLE** atomic de-excitation for the primary decay event generation (or ensure the source loader acts as a "phase space" source without recreating the decay shell vacancy).
- **Benefit:** Matches the standardized data exactly, but may lose spatial precision at the nanometer scale (all particles usually start at the same point).
- **Avoidance:** If you use a standard spectrum file (Method B), the code creates the electrons as independent particles. They may not be emitted from the exact same coordinate or time, and you lose the "high local charge density" correlation that causes the biological lethality (high RBE). **Always simulate the decay explicitly for Nanodosimetry.**

## 5.2 The "Production Threshold" Problem (General Dosimetry)

**The Trap:** In charged particle transport, energy loss is continuous. However, occasionally a "hard" collision creates a discrete secondary electron (Delta ray).

**The Error:** If the code subtracts "Stopping Power" energy (which mathematically accounts for all energy losses) *and* creates a discrete Delta ray without reducing the primary's energy, energy is created from nothing.

**Solutions:**

### Production Cut-offs ($E_{cut}$ or Range Cuts)
- Define a threshold (e.g., 1 keV or 0.1 mm).
- **Below $E_{cut}$:** The energy loss is treated as "Continuous Slowing Down" (CSDA). The energy is deposited locally along the step. **No secondary particle is tracked.**
- **Above $E_{cut}$:** The interaction is treated as discrete. A new particle (Delta ray) is created and tracked. **The primary particle loses exactly that amount of kinetic energy.**
- **Consistency Check:** Ensure the "Restricted Stopping Power" (used for continuous loss) matches the production cut-off used for discrete creation.

## 5.3 The "Kerma vs. Dose" Problem (Photon Dosimetry)

**The Trap:** For photons, energy is transferred to electrons (Kerma), which then deposit dose.

**The Error:** Scoring "Energy Released" (Kerma) *and* transporting the electrons to score "Energy Deposited" (Dose) in the same volume, then adding them together.

**Solutions:**

### Exclusive Scoring
- **Use Kerma ($K$):** If Charged Particle Equilibrium (CPE) exists (e.g., deep inside a liver tumor), $D \approx K$. You do not need to transport secondary electrons; assume energy is deposited at the point of interaction. (Fast, approximate).
- **Use Absorbed Dose ($D$):** If explicit electron transport is enabled, score *only* energy deposited by the charged particles. Ignore the photon energy loss events in the tallying.

### Terminal Energy Deposition
- When a particle drops below the tracking threshold ($E < E_{cut}$), the simulation must "kill" the particle and deposit its *remaining* kinetic energy into the current voxel.

## 5.4 The "Relaxation" Double Count (Diagnostic/Low Energy)

**The Trap:** During Photoelectric interactions.

**The Error:** Some approximate codes deposit the full photon energy ($E_\gamma$) locally (assuming the photoelectron and Augers stop instantly). If the user *also* tracks the photoelectron, they double-count the kinetic energy. If they *also* track fluorescence photons, they double-count the binding energy.

**Solutions:**

### Energy Budgeting
- $E_{deposited} = E_{\gamma} - E_{kinetic\_electron} - E_{binding}$.
- The simulation must explicitly track the $E_{binding}$.
