# Validation Strategy

The validation of the GPUMCRPTDosimetry project is a multi-faceted process designed to ensure the correctness of the physics implementation, the performance of the GPU kernels, and the overall reliability of the simulation pipeline.

## Core Pillars of Validation

The validation strategy is built on three core pillars:

1.  **Physics Correctness:** Ensuring that the simulation accurately models the underlying physics of radionuclide decay and particle transport.
2.  **Performance and Numerical Stability:** Verifying that the GPU implementation is not only fast but also numerically stable and free of artifacts.
3.  **Energy Conservation:** A fundamental check to ensure that energy is conserved throughout the simulation process, from particle emission to deposition.

## Physics Validation

### Comparison with Reference Codes
The ultimate test of a Monte Carlo code is comparison with established and trusted codes. The project includes tools and workflows for comparing the dose distributions calculated by GPUMCRPTDosimetry with those from **Geant4/GATE**.

### Component-Level Physics Tests
The test suite includes numerous tests that verify individual physics components in isolation:
*   **Compton Kinematics:** The energy and scattering angle of photons undergoing Compton scattering are checked against the theoretical Klein-Nishina formula.
*   **Charged Particle Range:** The range of electrons and positrons is compared against expected values.
*   **Stopping Power:** The calculated stopping power is verified to be positive and monotonic as expected.
*   **Cross-Sections:** The cross-section data is checked for internal consistency (e.g., partial cross-sections must sum to less than or equal to the total cross-section).

## Performance and Numerical Stability

### GPU Kernel Validation
The GPU performance test suite specifically targets the Triton kernels:
*   **Numerical Stability:** Kernels are tested for stability with large number accumulation (using `atomic_add`) and for safety when dividing by small numbers.
*   **Boundary Conditions:** Voxel boundary checking is explicitly tested to prevent out-of-bounds memory access.
*   **Memory Efficiency:** The memory layout of particle queues and material volumes is tested to ensure it is optimal for coalesced memory access on the GPU.

### Determinism
A key aspect of validation is ensuring that the simulation is deterministic. With a fixed initial seed, the simulation should produce bit-wise identical results (within the expected limits of floating-point non-associativity). This is tested by:
*   Using a deterministic, counter-based RNG (Philox).
*   Running identical simulations twice and comparing the results.

## Energy and Particle Conservation

### Energy Conservation Checks
A fundamental principle of the simulation is that energy must be conserved. This is checked at multiple levels:
*   **Decay and Emission:** The total energy of the particles emitted in a decay event must equal the initial energy of the decay (minus the energy carried away by neutrinos).
*   **End-to-End Conservation:** For a simulation run in a fully absorbing phantom, the total energy deposited in the dose grid must equal the total energy of all the primary particles generated.
*   The `phase11_validation_gate.md` document explicitly recommends tracking `E_emitted_total`, `E_deposited_total`, and `E_escaped_total` to ensure that `E_emitted ≈ E_deposited + E_escaped`.

### No-Double-Counting
Care is taken to ensure that energy is not deposited twice. For example:
*   When a secondary particle is created (e.g., a delta-ray), its energy is subtracted from the parent particle.
*   In the photoelectric effect, the binding energy is accounted for in the relaxation cascade, not deposited separately.

### Overflow Checks
The system is tested for its ability to handle particle bank overflows gracefully. By artificially limiting the size of the particle banks, the validation process ensures that an overflow condition is detected and handled without memory corruption.
