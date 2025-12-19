Here is the refined and physics-correct material table for dosimetry calculations. I have split the "Bone" category into **Trabecular (Spongy) Bone** and **Cortical (Compact) Bone**.

The data below is derived from **ICRU Report 44** and **ICRP Publication 110**, which are the standard references for tissue substitutes in radiation physics (Monte Carlo simulations and Treatment Planning Systems).

### Physics-Correct Material Table

| Compartment | HU Range (Approx.) | Density ($\rho$) (g/cm³) | RED ($\rho_e/\rho_{e,w}$) | Elemental Composition (by mass %) |
| :--- | :--- | :--- | :--- | :--- |
| **Air** | −1000 | 0.0012 | 0.001 | N 75.5, O 23.2, Ar 1.3 |
| **Lung** | −950 to −400 | 0.26 | 0.25 | H 10.3, C 10.5, N 3.1, O 74.9, Na 0.2, P 0.2, S 0.3, Cl 0.3, K 0.2 |
| **Adipose or Fat** | −120 to −50 | 0.95 | 0.95 | H 11.4, C 59.8, N 0.7, O 27.8, Na 0.1, S 0.1, Cl 0.1 |
| **Soft Tissue or Water** | −50 to +40 | 1.00 | 1.00 | H 11.2, O 88.8 (Approximated as Water for dosimetry) |
| **Muscle** | +40 to +80 | 1.05 | 1.04 | H 10.2, C 14.3, N 3.4, O 71.0, Na 0.1, P 0.2, S 0.3, Cl 0.1, K 0.4 |
| **Trabecular Bone** | **+100 to +600** | **1.10 – 1.20** | **1.08 – 1.14** | **H 8.5, C 40.4, N 2.8, O 36.7, Na 0.1, P 3.4, S 0.2, Ca 7.8** |
| **Cortical Bone** | **+600 to +3000** | **1.85 – 1.92** | **1.65 – 1.70** | **H 3.4, C 15.5, N 4.2, O 43.5, Na 0.1, Mg 0.2, P 10.3, S 0.3, Ca 22.5** |

---

### Physics & Dosimetry Notes

To ensure your calculation is "physics correct," you must understand the relationship between these values. Here is the breakdown:

#### 1. The Split: Trabecular vs. Cortical Bone
In radiation transport (especially for photons < 1 MeV), the distinction is critical due to the **Photoelectric Effect**.
*   **Trabecular Bone (Spongy):** Contains a lattice of bone matrix filled with bone marrow (which is essentially fat and blood). Therefore, it has a high Carbon content (~40%) and lower Calcium (~7-8%). Its density is only slightly higher than water.
*   **Cortical Bone (Compact):** The hard shell of the bone. It has very little fat and high mineral content (Hydroxyapatite). The Calcium content is high (~22%), which drastically increases the Effective Atomic Number ($Z_{eff}$), causing higher attenuation of low-energy x-rays.

#### 2. Relative Electron Density (RED) vs. Mass Density
Dosimetry algorithms (like Convolution Superposition) rely on **RED**, while Monte Carlo algorithms rely on **Mass Density** and **Elemental Composition**.

*   **Relationship:** $\text{RED} \approx \rho \times \frac{(Z/A)_{material}}{(Z/A)_{water}}$
*   Hydrogen has a $Z/A \approx 1$. Heavier elements (C, N, O, Ca) have $Z/A \approx 0.5$.
*   **Fat:** High Hydrogen content means its RED (0.95) is numerically closer to its mass density (0.95).
*   **Bone:** Low Hydrogen and high Calcium content means the electron density does not scale perfectly linearly with mass density compared to water.
    *   *Notice in the table:* Cortical bone Mass Density is **1.92**, but RED is only **~1.70**. If you assume RED = Mass Density for bone, you will **overestimate** the dose attenuation in standard algorithms.

#### 3. Hounsfield Unit (HU) Saturation
In the table above, the transition from Trabecular to Cortical is usually modeled bi-linearly.
*   **< 0 HU:** The curve is linear based on air and fat.
*   **0 to ~100 HU:** The curve accounts for water/muscle mixture.
*   **> 100 HU:** The curve slope changes because the increase in HU is driven by Calcium (high Z), not just density.

#### 4. The "Schneider" Technique
If you are implementing this for a high-accuracy Treatment Planning System or Monte Carlo simulation, you should look up the **Schneider parameterization** (Schneider et al., *Phys Med Biol* 2000).
*   It does not use fixed bins. Instead, it converts HU directly to elemental weights using a continuous ramp.
*   However, for segmented compartment calculations (like the table you requested), the binning provided above is the standard "Discrete Material" approach used in phantom dosimetry.