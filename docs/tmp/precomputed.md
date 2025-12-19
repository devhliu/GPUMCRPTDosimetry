For GPU-accelerated Monte Carlo (MC) dosimetry (e.g., using codes like compiled Geant4-DNA, GATE-GPU, or in-house CUDA/OpenCL codes), the primary bottleneck is memory access latency. To overcome this, specific physics and geometry data are **precomputed and stored in high-speed GPU memory** (Texture Memory or Constant Memory) to avoid complex on-the-fly calculations.

Here are the standard categories of precomputed data used in these calculations:

### 1. Physics Interaction Look-up Tables (LUTs)
Instead of calculating quantum mechanical cross-sections (using complex formulas like Klein-Nishina or Bethe-Bloch) at every step, the GPU looks up pre-calculated values.

*   **Total Cross-Sections:** Tables defining the probability of an interaction occurring (mean free path) for every particle type, energy level, and material type.
    *   *Photons:* Photoelectric, Compton scattering, Rayleigh scattering, Pair production.
    *   *Electrons/Positrons:* Elastic scattering, Ionization, Bremsstrahlung, Excitation.
*   **Restricted Stopping Power ($L_\Delta$):** Precomputed tables for the rate of energy loss per unit path length ($dE/dx$). This is critical for charged particle transport (Betas and Alphas).
*   **Inverse Cumulative Distribution Functions (iCDFs):** Used for **sampling**. When an interaction occurs (e.g., Compton scattering), the GPU must decide the scattering angle and output energy. Instead of integrating a function, the GPU picks a random number and looks up the corresponding angle/energy from a precomputed iCDF table.

### 2. Isotope Decay Spectra (Source Data)
Radionuclides (e.g., Lu-177, Y-90, I-131) do not emit particles at a single energy (except for specific gamma lines); beta emission is a continuous spectrum.

*   **Beta/Positron Spectra:** The Fermi distribution for beta decay is computationally expensive. It is precomputed into a **Probability Density Function (PDF)** or **CDF**.
    *   *Usage:* The GPU generates a random number [0,1] and queries this table to assign an initial energy to the beta particle.
*   **Auger & Conversion Electron Cascades:** For isotopes like I-125 or Tb-161, the complex atomic relaxation trees are often flattened into a precomputed list of emission probabilities and energies.

### 3. Voxelized Geometry & Material Map (3D Textures)
The CT scan and your material classification (from the previous prompt) are converted into 3D arrays stored in **GPU Texture Memory**. This takes advantage of the GPU's hardware caching.

*   **Density Map:** A 3D voxel map containing the mass density ($\rho$) for every voxel.
*   **Material Index Map:** A 3D voxel map containing an integer ID (e.g., 0=Air, 1=Lung, 2=Trabecular, 3=Cortical) corresponding to the material properties.
    *   *Why split them?* Often packed into a single RGBA texture where the Red channel = Density and Green channel = Material ID to minimize memory fetches.

### 4. Electron Scattering Power & Range Tables
Simulating every single electron interaction is impossible (there are too many). GPUs use "Condensed History" algorithms (like Class II condensed history).

*   **Scattering Power ($T_{scat}$):** Precomputed data describing how much the electron direction variance increases per unit step.
*   **CSDA Range:** The Continuous Slowing Down Approximation range table helps the GPU decide when a particle has effectively stopped (energy < cut-off) so the thread can be terminated to free up resources.


### Summary of Data Flow on GPU

1.  **Thread Start:** Look up **Decay Spectrum CDF** $\to$ Spawn Particle.
2.  **Navigation:** Look up **3D Texture Map** $\to$ Get Material ID & Density at current $(x,y,z)$.
3.  **Step Calculation:** Look up **Cross-Section LUT** (based on Material ID + Energy) $\to$ Determine distance to next interaction.
4.  **Interaction:** Look up **Interaction iCDF** $\to$ Determine energy loss and angle change.
5.  **Scoring:** Atomic add energy to **Dose Grid** (3D Array).

### Why this matters for your "Bone" split?
Because you split Trabecular and Cortical bone, the GPU needs **two distinct rows** in the **Cross-Section LUT** (Category 1).
*   Even if the density is similar, the **Photoelectric cross-section** for Cortical bone will be much higher due to the Calcium ($Z=20$).
*   If you had not split them, the precomputed table would average the Calcium content, causing errors in bone marrow dose calculations (crucial for toxicity limits in radionuclide therapy).