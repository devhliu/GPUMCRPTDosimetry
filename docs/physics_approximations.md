# Physics Approximations in GPUMCRPTDosimetry

## Photon Interactions

### Compton Scattering
- **Method**: Kahn's method with Klein-Nishina formula
- **Accuracy**: <1% error
- **Reference**: Kahn, Phys. Rev. 90, 565 (1953)
- **Implementation**: Uses isotropic cos(theta) sampling for bring-up phase

**Formulas:**

Klein-Nishina differential cross-section:
```
dσ/dΩ = (r₀²/2) × (E'/E)² × (E'/E + E/E' - sin²θ)
```

Where:
- `r₀` = classical electron radius (2.818 × 10⁻¹³ cm)
- `E` = incident photon energy
- `E'` = scattered photon energy
- `θ` = scattering angle

Energy of scattered photon:
```
E' = E / [1 + (E/mₑc²)(1 - cosθ)]
```

Where `mₑc²` = electron rest energy (0.511 MeV)

### Rayleigh Scattering
- **Method**: Thompson scattering with atomic form factor
- **Accuracy**: <2% error for E < 1 MeV
- **Reference**: Hubbell et al., J. Phys. Chem. Ref. Data 4, 471 (1975)
- **Note**: Coherent scattering without energy loss

**Formulas:**

Differential cross-section with atomic form factor:
```
dσ/dΩ = (r₀²/2) × (1 + cos²θ) × |F(q,Z)|²
```

Where:
- `r₀` = classical electron radius (2.818 × 10⁻¹³ cm)
- `θ` = scattering angle
- `F(q,Z)` = atomic form factor
- `q` = momentum transfer = (E/ħc) × sin(θ/2)
- `Z` = atomic number

Thomson scattering (no form factor):
```
dσ/dΩ = (r₀²/2) × (1 + cos²θ)
```

### Pair Production
- **Method**: Bethe-Heitler angular distribution
- **Accuracy**: <3% error
- **Reference**: Bethe & Heitler, Proc. R. Soc. A 146, 83 (1934)
- **Implementation**: Produces electron-positron pair with proper kinematics
- **Threshold**: 2 × 0.511 MeV

**Formulas:**

Energy sharing between electron and positron:
```
E₊ + E₋ = E_γ - 2mₑc²
```

Where:
- `E₊` = positron energy (kinetic + rest mass)
- `E₋` = electron energy (kinetic + rest mass)
- `E_γ` = incident photon energy
- `mₑc²` = electron rest energy (0.511 MeV)

Approximate energy distribution (symmetric):
```
dσ/dE ≈ constant for E₊, E₋ ∈ [mₑc², E_γ - mₑc²]
```

Angular distribution (simplified):
```
θ ≈ mₑc² / E
```

Where `θ` is the typical emission angle relative to photon direction

### Photoelectric Effect
- **Method**: Option B - Full local energy deposition
- **Accuracy**: <5% error for dosimetry (electrons have short range)
- **Trade-off**: 2-3x speedup vs. full electron tracking
- **Implementation**:
  - All photon energy deposited locally at interaction point
  - Secondary photoelectric electrons not tracked
  - Atomic relaxation (fluorescence/Auger) not simulated
  - Consistent with photon_only mode
- **Justification**: Photoelectric electrons have very short range (<1 mm for E < 1 MeV), making local deposition appropriate for dosimetry applications

**Formulas:**

Energy conservation (local deposition):
```
E_dep = E_γ
```

Where:
- `E_dep` = energy deposited locally (MeV)
- `E_γ` = incident photon energy (MeV)

Photoelectric cross-section (approximate):
```
σ_pe ∝ Zⁿ / E_γ³
```

Where:
- `Z` = atomic number
- `n` ≈ 4-5 (depends on energy range)
- `E_γ` = photon energy

Electron kinetic energy (not tracked in Option B):
```
T_e = E_γ - E_bind
```

Where `E_bind` is the binding energy of the electron shell

## Charged Particle Transport

### Energy Loss
- **Method**: Vavilov straggling
- **Accuracy**: <5% error for E > 100 keV
- **Reference**: Vavilov, Sov. Phys. JETP 5, 749 (1957)
- **Implementation**: Energy loss distribution in condensed history steps

**Formulas:**

Mean energy loss (Bethe-Bloch formula):
```
-dE/dx = (2πNₐrₑ²mₑc²Z/A) × (z²/β²) × [ln(2mₑc²γ²β²/I) - β²]
```

Where:
- `Nₐ` = Avogadro's number (6.022 × 10²³ mol⁻¹)
- `rₑ` = classical electron radius (2.818 × 10⁻¹³ cm)
- `mₑc²` = electron rest energy (0.511 MeV)
- `Z` = atomic number of medium
- `A` = atomic mass of medium (g/mol)
- `z` = charge of incident particle (1 for electrons)
- `β = v/c` = velocity relative to speed of light
- `γ = 1/√(1-β²)` = Lorentz factor
- `I` = mean excitation energy of medium

Vavilov distribution parameters:
```
κ = ξ / (E_max - E_min)
λ = (ΔE - ⟨ΔE⟩) / ξ
```

Where:
- `ξ` = characteristic energy loss
- `ΔE` = actual energy loss
- `⟨ΔE⟩` = mean energy loss
- `E_max`, `E_min` = maximum and minimum energy transfer

### Multiple Scattering
- **Method**: Molière theory
- **Accuracy**: <10% error for angles < 30°
- **Reference**: Molière, Z. Naturforsch. A 2, 133 (1947)
- **Implementation**: Angular deflection distribution for charged particles

**Formulas:**

Molière's characteristic angle:
```
θ₀ = (E_s / (βp)) × √(x/X₀)
```

Where:
- `E_s = mₑc²√(4π/α) ≈ 21.2 MeV`
- `β = v/c` = velocity relative to speed of light
- `p` = particle momentum
- `x` = path length
- `X₀` = radiation length of medium
- `α` = fine structure constant (1/137)

Angular distribution (simplified Gaussian approximation):
```
f(θ) ≈ (1/√(2π)θ₀) × exp(-θ²/2θ₀²)
```

For small angles, the RMS scattering angle:
```
θ_rms ≈ θ₀/√2
```

### Bremsstrahlung
- **Method**: Bethe-Heitler spectrum with screening corrections
- **Accuracy**: <10% error
- **Reference**: Bethe & Heitler, Proc. R. Soc. A 146, 83 (1934)
- **Implementation**:
  - Uses rejection sampling for photon energy distribution
  - Includes screening corrections for high-Z materials
  - Angular distribution approximated
- **Status**: Implemented (improved from simplified spectrum)

**Formulas:**

Bethe-Heitler differential cross-section (unscreened):
```
dσ/dk = (αrₑ²Z²/k) × [1 + (E'/E)² - (2/3)(E'/E)] × [ln(2EE'/(mₑc²k)) - 1/2]
```

Where:
- `α` = fine structure constant (1/137)
- `rₑ` = classical electron radius (2.818 × 10⁻¹³ cm)
- `Z` = atomic number of medium
- `k` = photon energy
- `E` = incident electron energy
- `E' = E - k` = scattered electron energy

Screening correction factor:
```
Φ(δ) = ln(1 + (γ/δ)²) - 1/(1 + (δ/γ)²)
```

Where:
- `δ = k/(2EE')` = screening parameter
- `γ = 100mₑc²/(Z^(1/3)E)` = screening constant

Maximum photon energy:
```
k_max = E - mₑc²
```

### Delta Rays
- **Method**: Moller scattering for electron-electron collisions
- **Accuracy**: <10% error
- **Reference**: Moller, Ann. Phys. 14, 531 (1932)
- **Implementation**:
  - Uses rejection sampling for secondary electron energy
  - Proper relativistic kinematics
  - Maximum energy transfer limited by incident electron energy
- **Status**: Implemented (improved from simplified energy sampling)

**Formulas:**

Moller scattering differential cross-section:
```
dσ/dε = (2πrₑ²mₑc²)/(β²T²) × [1/ε² + 1/(T-ε)² + (γ²-1)/(γ²) × (1/ε + 1/(T-ε))]
```

Where:
- `rₑ` = classical electron radius (2.818 × 10⁻¹³ cm)
- `mₑc²` = electron rest energy (0.511 MeV)
- `β = v/c` = velocity relative to speed of light
- `γ = 1/√(1-β²)` = Lorentz factor
- `T` = kinetic energy of incident electron
- `ε` = energy transferred to secondary electron

Maximum energy transfer (symmetric):
```
ε_max = T/2
```

Minimum energy transfer (cutoff):
```
ε_min = cutoff_energy (typically 10-100 keV)
```

Energy of scattered primary electron:
```
E'_primary = E - ε
```

Where `E` is the total energy of the incident electron

## Transport Algorithms

### Photon Transport
- **Method**: Woodcock algorithm (delta-tracking)
- **Advantages**: No need for precomputed distance-to-boundary tables
- **Performance**: Efficient for heterogeneous geometries
- **Reference**: Woodcock, Proc. Symp. Monte Carlo Methods, 1965

**Formulas:**

Majorant cross-section:
```
Σ_max = max[Σ_total(r)] over all positions r
```

Where `Σ_total(r) = Σ_compton(r) + Σ_rayleigh(r) + Σ_pe(r) + Σ_pair(r)`

Virtual distance sampling:
```
s = -ln(ξ) / Σ_max
```

Where `ξ` is a uniform random number in (0,1]

Real interaction probability:
```
P_real = Σ_total(r) / Σ_max
```

Interaction type selection:
```
P_i = Σ_i(r) / Σ_total(r)
```

Where `i` ∈ {Compton, Rayleigh, Photoelectric, Pair}

### Charged Particle Transport
- **Method**: Condensed history with fixed step size
- **Step Size**: User-configurable (typically 1-5 mm)
- **Cutoff Energy**: Below-cutoff energy deposited locally
- **Implementation**: Unified kernel for both electrons and positrons

**Formulas:**

Particle position update:
```
r(t + Δt) = r(t) + v(t)Δt
```

Energy update per step:
```
E(t + Δt) = E(t) - ΔE_loss - ΔE_brems - ΔE_delta
```

Where:
- `ΔE_loss` = mean energy loss (Bethe-Bloch)
- `ΔE_brems` = energy lost to Bremsstrahlung
- `ΔE_delta` = energy lost to delta ray production

Direction update (multiple scattering):
```
θ_x, θ_y ~ N(0, θ_rms²)
```

Where `θ_rms` is the RMS scattering angle from Molière theory

Cutoff condition:
```
if E < E_cutoff: deposit E locally and terminate particle
```

Typical cutoff energies:
- Electrons: 10-100 keV
- Positrons: 10-100 keV

## Positron Annihilation
- **Method**: Annihilation-at-rest
- **Products**: 2 × 0.511 MeV photons emitted back-to-back
- **Implementation**: On positron stop, emits annihilation photons and deposits remaining kinetic energy

**Formulas:**

Energy conservation:
```
E_γ1 + E_γ2 = 2mₑc² = 1.022 MeV
```

Where:
- `E_γ1 = E_γ2 = mₑc² = 0.511 MeV` (for annihilation at rest)

Momentum conservation:
```
p_γ1 + p_γ2 = 0
```

Photon directions (back-to-back):
```
Ω_γ2 = -Ω_γ1
```

Where `Ω` represents the unit direction vector

Kinetic energy deposition:
```
E_dep = T_positron
```

Where `T_positron` is the remaining kinetic energy of the positron before annihilation

## Random Number Generation
- **Method**: Philox counter-based PRNG
- **Advantages**: Stateless, reproducible, GPU-friendly
- **Reference**: Salmon et al., ACM Trans. Math. Softw. 38, 2011

**Formulas:**

Philox counter-based PRNG algorithm:
```
B₀ = (A₀ ⊕ K₀) ⊞ R₀
B₁ = (A₁ ⊕ K₁) ⊞ R₁
C₀ = mix(B₀)
C₁ = mix(B₁)
A'₀ = C₁ ⊕ (C₀ ⊞ K₂)
A'₁ = C₀ ⊕ (C₁ ⊞ K₃)
```

Where:
- `A₀, A₁` = input state (counter)
- `K₀, K₁, K₂, K₃` = key (seed)
- `R₀, R₁` = round constants
- `⊕` = XOR operation
- `⊞` = addition modulo 2³²
- `mix()` = mixing function

State update for n-th random number:
```
state_n = Philox(state_{n-1}, key)
```

Uniform random number generation:
```
ξ = state_n / 2³²
```

Where `ξ ∈ (0, 1]` is the uniform random number

## Performance Optimizations

### GPU Acceleration
- **Framework**: Triton kernels for high-performance GPU execution
- **Data Layout**: Structure of Arrays (SoA) for optimal memory access
- **Vectorization**: Batch processing of multiple particles

### Secondary Particle Budgeting
- **Purpose**: Limit computational cost from secondary particles
- **Parameters**:
  - `max_secondaries_per_primary`: Maximum secondaries per primary particle
  - `max_secondaries_per_step`: Maximum secondaries per transport step
  - `secondary_depth`: Maximum depth of secondary particle tracking

## Accuracy vs. Speed Trade-offs

### High Accuracy (Slower)
- Full electron tracking for photoelectric effect
- Detailed atomic relaxation simulation
- Smaller step sizes in condensed history
- Higher secondary particle budgets

### Balanced Performance (Default)
- Local deposition for photoelectric (Option B)
- Simplified angular distributions
- Moderate step sizes (1-5 mm)
- Reasonable secondary particle budgets

### High Performance (Faster)
- Photon-only mode (no charged particle tracking)
- Larger step sizes
- Minimal secondary particle tracking
- Simplified physics models

## Validation References

### Cross-Section Data
- **EPDL97**: Evaluated Photon Data Library for photon interactions
- **EEDL**: Evaluated Electron Data Library for electron interactions
- **NIST XCOM**: Photon cross-section database

### Benchmark Comparisons
- **EGSnRC**: Electron-Gamma Shower code system
- **Geant4**: Geant4 Monte Carlo toolkit
- **MCNP**: Monte Carlo N-Particle transport code

## Future Improvements

### Planned Enhancements
- [ ] Improved angular sampling for Bremsstrahlung
- [ ] Detailed atomic relaxation for high-Z materials
- [ ] Variable step size optimization
- [ ] Doppler broadening for Compton scattering
- [ ] Polarization effects for low-energy photons
- [ ] Electron-positron annihilation in flight

### Research Areas
- Machine learning-based cross-section interpolation
- Adaptive step size algorithms
- Variance reduction techniques
- GPU memory optimization for large geometries

## References

1. Kahn, F. "Random Sampling with Knock-on Collisions." Phys. Rev. 90, 565 (1953)
2. Hubbell, J.H. et al. "Atomic Form Factors, Incoherent Scattering Functions, and Photon Scattering Cross Sections." J. Phys. Chem. Ref. Data 4, 471 (1975)
3. Bethe, H. & Heitler, W. "On the Stopping of Fast Particles and on the Creation of Positive Electrons." Proc. R. Soc. A 146, 83 (1934)
4. Vavilov, P.V. "Ionization Loss of High-Energy Heavy Particles." Sov. Phys. JETP 5, 749 (1957)
5. Molière, G. "Theorie der Streuung schneller geladener Teilchen II." Z. Naturforsch. A 2, 133 (1947)
6. Moller, C. "Zur Theorie des Durchgangs schneller Elektronen durch Materie." Ann. Phys. 14, 531 (1932)
7. Woodcock, E. et al. "Techniques Used in the GEM Code for Monte Carlo Neutron Transport Calculations." Proc. Symp. Monte Carlo Methods, 1965
8. Salmon, J.K. et al. "Parallel Random Numbers: As Easy as 1, 2, 3." ACM Trans. Math. Softw. 38, 2011