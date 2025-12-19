# Project Goals

The primary goal of the GPUMCRPTDosimetry project is to develop a high-performance, clinically-realistic Monte Carlo simulation tool for radionuclide dosimetry. The tool is designed to leverage the massive parallelism of modern GPUs to accelerate dose calculations, making it a viable option for clinical and research applications where speed and accuracy are paramount.

## Key Objectives

1.  **High-Performance GPU Acceleration:** The core of the project is to implement the physics of radionuclide decay, particle transport, and energy deposition on the GPU using frameworks like Triton. This is intended to provide a significant speed-up over traditional CPU-based Monte Carlo methods.

2.  **Clinical Realism:** The simulation must be able to model the complex physics of radionuclide therapy, including:
    *   The decay of various radionuclides and the emission of their full spectrum of particles (photons, electrons, positrons, etc.).
    *   The transport of these particles through patient-specific tissue models derived from CT images.
    *   Accurate modeling of physical interactions such as Compton scattering, photoelectric effect, Rayleigh scattering, condensed history transport for charged particles, and positron annihilation.
    *   The use of realistic material definitions, including elemental compositions for different tissue types.

3.  **Validation and Accuracy:** The results of the simulation must be verifiable against established and trusted Monte Carlo codes like Geant4/GATE. The project includes the development of tools and phantoms for this purpose.

4.  **User-Friendliness and Accessibility:** The tool should be easy to use, with a clear pipeline for running simulations from NIfTI files (CT and activity maps) and a flexible configuration system. The output should be in standard formats (NIfTI) for easy integration into existing clinical and research workflows.

5.  **Modularity and Extensibility:** The code is designed to be modular, with different physics models and transport engines that can be selected and extended. This allows for a phased development approach and facilitates future improvements and additions.
