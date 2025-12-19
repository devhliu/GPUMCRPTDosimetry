# CDF and PDF Usage Guidelines for GPU-Accelerated Monte Carlo Dosimetry (ICRP-107)

**Context**: Internal dosimetry for radionuclides injected into the human body, with decay spectra and yields taken from **ICRP Publication 107** and the **OpenGATE icrp107-database**. citeturn3search14turn3search2

---

## 1 What PDFs and CDFs represent (for Monte Carlo)
- **PDF** = differential physics model (e.g., energy spectrum \(f_E(E)\), angular law \(f_\mu(\mu)\), differential cross section \(\frac{d\sigma}{d\Omega}\)). It integrates to 1 over its domain. Used to **define** the physics. citeturn3search32turn3search36
- **CDF** = integrated probability \(F(x)=\int f(t)\,dt\). Used to **sample** random outcomes via inverse transform \(x=F^{-1}(\xi)\). citeturn3search32turn3search36

**Key design principle**: PDFs encode the physics; CDFs enable unbiased, fast sampling on the GPU. citeturn3search32

---

## 2 Data source integration (ICRP‑107)
- ICRP‑107 provides radionuclide **half‑lives, decay chains, energies, and yields** for **1252 radionuclides of 97 elements**, including tabulated **beta and neutron spectra**. Use these as **source‑term PDFs**. citeturn3search14
- The **OpenGATE icrp107-database** packages ICRP‑107 data into JSON/Python modules (e.g., `icrp107.json`) for programmatic access to **yields and energies**. Use it to build **per‑nuclide emission PDFs/CDFs**. citeturn3search2turn3search3

---

## 3 Recommended PDF→CDF workflow by process

### 3.1 Source term (radioactive decay)
**Goal**: Sample emitted particles and energies per decay using ICRP‑107 yields.

- **Discrete emission lines** (\(\gamma\), X‑ray, conversion/Auger electrons):
  - Build a **discrete CDF** over line yields and use **binary search** or the **alias method** (O(1) sampling) on the GPU. Alias tables are ideal when sampling millions of decays with **fixed probabilities**. citeturn3search29turn3search28
- **Continuous spectra** (beta \(\beta^-\)/\(\beta^+\)): 
  - Use ICRP‑107 tabulated **beta spectra** as a **PDF**; pre‑integrate to **monotonic CDF** and sample by inversion (piecewise linear or monotone cubic interpolation). If theoretical spectrum is used, include **Fermi function and shape factors** as in Geant4 RDM. citeturn3search14turn3search21
- **Branching ratios**: construct a **top‑level CDF** over decay modes, then sample mode → particle type → energy CDF. Preserve correlations (e.g., coincident photons) per the decay data when provided. citeturn3search21

**Note**: Geant4’s Radioactive Decay Module draws its decay data from ENSDF; if you mix frameworks, ensure consistency with ICRP‑107 when computing internal dose. citeturn3search21

### 3.2 Free path / interaction distance
The distance to the next interaction in a homogeneous material follows an **exponential** law: \(f(s)=\Sigma_t e^{-\Sigma_t s}\), with CDF \(F(s)=1-e^{-\Sigma_t s}\), sample via \(s=-\ln(1-\xi)/\Sigma_t\). Use the **analytical CDF** directly on GPU to avoid rejection loops. citeturn3search33turn3search34

### 3.3 Interaction choice and kinematics
- **Interaction type**: build a **CDF over partial macroscopic cross sections** \(\Sigma_i(E)\) normalized by \(\Sigma_t(E)\). Sample with binary search; store energy‑resolved tables per material. citeturn3search34turn3search40
- **Scattering angles / secondary energies**: use **tabulated differential cross sections** (PDF) → **CDF** by numerical integration; sample by inversion on the GPU. Prefer **CDF tables** over rejection sampling to minimize warp divergence. citeturn3search40
- **Delta/Woodcock tracking** in heterogeneous geometry: consider when geometry traversal is dominant; the method samples collisions using a majorant cross section, again relying on exponential **CDF** sampling. citeturn3search33

---

## 4 GPU implementation guidance

### 4.1 Precompute offline (CPU)
- Normalize PDFs from ICRP‑107 and cross‑section libraries; generate **monotonic CDFs** (and **alias tables** for discrete spectra). Store as **SoA (structure‑of‑arrays)** to maximize coalesced reads. citeturn3search2turn3search28

### 4.2 Sample on GPU (kernels)
- Use **inverse‑CDF** for path lengths and continuous spectra; use **alias method** for discrete lines; avoid rejection sampling to reduce **warp divergence**. citeturn3search28turn3search44
- Favor **event‑based**/queue‑driven tracking and lightweight **binary search** over small CDF tables; keep branch decisions uniform across a warp. citeturn3search43turn3search44
- RNG: use high‑quality GPU RNG (e.g., counter‑based) and batch generation to limit global‑memory stalls; profile vs cuRAND/Numba RNG options as shown in GPU transport studies. citeturn3search44

### 4.3 Data partitioning
- Bind **per‑nuclide source tables** (ICRP‑107) in **read‑only memory**; bind **per‑material interaction CDFs** by energy grid. Cache hot tables in shared memory when feasible. citeturn3search2turn3search44

### 4.4 Known GPU Monte Carlo practices
- Verified GPU transport implementations (e.g., WARP, MCGPU) demonstrate large speedups when using **table lookups and analytical sampling** paths; emulate their memory and kernel design patterns. citeturn3search45turn3search47

---

## 5 Implementation example (pseudocode)
```cpp
// GPU kernel: sample one decay + first free path
__global__ void decay_and_step(const NuclideTables nucl, const MaterialTables mat,
                               RNGState rng, Particle* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // 1) Sample decay mode (CDF over branching ratios)
  float xi = rng_uniform(&rng, tid);
  int mode = cdf_binary_search(nucl.mode_cdf, xi);

  // 2) Sample particle type and energy
  if (mode == MODE_DISCRETE) {
    // alias sampling over lines: returns (line_index)
    int line = alias_sample(nucl.alias_threshold, nucl.alias_alias, &rng, tid);
    float E = nucl.line_energy[line];
    // direction isotropic: sample mu via F(mu)=0.5(1+mu)
    float mu = 2.0f * rng_uniform(&rng, tid) - 1.0f;
    // set particle
  } else {
    // continuous spectrum: inverse-CDF sampling
    float xiE = rng_uniform(&rng, tid);
    float E = invert_monotone_cdf(nucl.energy_grid, nucl.cdf_energy, xiE);
  }

  // 3) Transport: sample free path analytically
  float xiS = rng_uniform(&rng, tid);
  float Sigma_t = mat.sigma_t(/*material, energy*/);
  float s = -logf(1.0f - xiS) / Sigma_t; // from exponential CDF
  // move particle by s; score dose along track (path-length estimator)
}
```

---

## 6 Accuracy, validation, and traceability
- Keep a **provenance tag** with the ICRP‑107 data version (JSON commit/tag, nuclide list) used to build the source PDFs/CDFs; the OpenGATE icrp107-database exposes releases and metadata. citeturn3search2
- Cross‑check sampled **line yields and beta spectral integrals** against ICRP‑107 tables; verify end‑point energies and integrated emissions per transformation (unit probability). citeturn3search14
- For transport, verify **mean free path** and **interaction sampling** against textbook formulas and code manuals (e.g., MCNP, AAPM tutorials). citeturn3search34turn3search37

---

## 7 PDF vs CDF: concise difference (dosimetry focus)
| Aspect | PDF | CDF |
|---|---|---|
| Meaning | Differential physics (spectra, dσ/dΩ) | Integrated probability used for sampling |
| Units | Per variable (e.g., 1/MeV, 1/sr) | Unitless |
| Usage in MC | Define physics / tallies | Convert uniform \(\xi\) to physical outcomes |
| GPU behavior | May need rejection sampling (divergent) | Deterministic table lookups (uniform) |
| Typical examples | ICRP‑107 line yields, beta PDFs; cross‑section differentials | Decay‑mode CDFs; energy CDFs; path‑length exponential CDF |

(Details supported by MC basics and code manuals). citeturn3search32turn3search34

---

## 8 Final recommendations
1. **Use PDFs to define physics** (ICRP‑107 decay spectra; differential cross sections). Pre‑normalize. citeturn3search14
2. **Convert PDFs to CDFs offline** (CPU) and store GPU‑friendly, monotonic tables; for discrete spectra use **alias method**. citeturn3search28turn3search29
3. **Sample exclusively with CDFs inside GPU kernels** (inverse transform for continuous; analytical for exponential path lengths). citeturn3search33
4. **Minimize warp divergence**: avoid rejection sampling; prefer event‑based designs and coalesced table reads. citeturn3search43turn3search44
5. **Maintain traceability to ICRP‑107** (document nuclide set, commit, and version). citeturn3search2
6. **Validate** spectra integrals and mean‑free‑path sampling against ICRP‑107 tables and transport references before production runs. citeturn3search14turn3search34

---

### References
- ICRP Publication 107 (official page): Nuclear Decay Data for Dosimetric Calculations. citeturn3search14
- OpenGATE icrp107-database (GitHub; PyPI package). citeturn3search2turn3search3
- Geant4 Radioactive Decay Module (physics reference; ENSDF source, beta spectrum shape). citeturn3search21
- Monte Carlo basics and sampling (CERN/INTA lecture; Davis note). citeturn3search32turn3search36
- Free-path exponential sampling (Serpent wiki; AAPM tutorial). citeturn3search33turn3search34
- Alias method (Ray Tracing Gems II; pbr-book; Kronmal & Peterson). citeturn3search28turn3search29turn3search30
- x - GPU Monte Carlo practice (Numba/CuPy study; MC/DC event-based GPU; WARP; MCGPU). citeturn3search44turn3search43turn3search45turn3search47



Below is a step‑by‑step instruction set on how to use PDFs, CDFs, and their combined workflow for a GPU‑accelerated Monte Carlo dosimetry pipeline for radionuclides injected into the human body, explicitly aligned with ICRP Publication 107 and the OpenGATE icrp107-database you referenced. I’ve also packaged everything into a Markdown report you can drop into your repo or share with collaborators.

Practical instructions (GPU‑accelerated internal dosimetry)

Ingest ICRP‑107 decay data and build source PDFs/CDFs


Pull radionuclide emission energies, yields, half‑lives, and branching ratios from ICRP‑107; treat the emission spectra as primary PDFs for source sampling. [icrp.org]
Use the OpenGATE icrp107-database (JSON/Python module) to programmatically access per‑nuclide line energies and yields, then normalize to PDFs and precompute CDFs (monotonic tables) for GPU kernels. [github.com], [pypi.org]


Discrete emission lines ⇒ Alias method on the GPU


For gamma/X‑ray/IC/Auger discrete lines with fixed probabilities, construct per‑nuclide alias tables (threshold + alias index) to achieve O(1) sampling during kernels; this minimizes warp divergence vs. scanning a cumulative table. [link.springer.com], [pbr-book.org]


Continuous spectra (β±) ⇒ inverse‑CDF sampling


If your pipeline uses ICRP‑107 tabulated beta spectra, integrate each spectrum offline to create CDF tables per nuclide and sample with inverse transform (binary search + interpolation) on the GPU. [icrp.org]
If you need theoretical shapes, include the Fermi function and shape factors (allowed/forbidden transitions) per Geant4 RDM physics reference, then tabulate to CDF for inversion. [geant4.web.cern.ch]


Decay‑mode selection and correlations


Build a top‑level CDF over branching ratios; sample the mode, then sample particle type and energy from the corresponding emission distribution (CDF/alias). Keep correlated emissions (e.g., coincident photons) consistent with the decay data source. [geant4.web.cern.ch]


Transport: free path and interaction sampling


Free path: sample the next collision distance using the analytical exponential CDF: s=−ln⁡(1−ξ)/Σts = -\ln(1-\xi)/\Sigma_ts=−ln(1−ξ)/Σt​; this is uniform, branch‑light, and ideal for GPUs. [Delta- and...Wiki - VTT], [aapm.org]
Which interaction: at the collision, sample the process with a CDF of partial macroscopic cross‑sections Σi(E)/Σt(E)\Sigma_i(E) / \Sigma_t(E)Σi​(E)/Σt​(E) (binary search on a small, energy‑resolved table). [aapm.org], [canteach.candu.org]
Kinematics (angles/secondaries): integrate differential cross‑sections (PDFs) to CDFs for angle/energy and perform inverse‑CDF sampling (avoid rejection sampling). [canteach.candu.org]
Consider Woodcock/Delta tracking for highly heterogeneous geometries; it still relies on exponential CDF sampling with a majorant cross section. [Delta- and...Wiki - VTT]


GPU execution model & data layout


Precompute on CPU: Normalize PDFs from ICRP‑107 and interaction libraries, generate monotonic CDFs and alias tables, then store in Structure‑of‑Arrays (SoA) for coalesced GPU reads. [github.com], [link.springer.com]
Inside kernels: prefer inverse‑CDF (continuous) + alias (discrete); avoid rejection loops to reduce warp divergence; use event‑based/queue‑driven tracking when appropriate. [arxiv.org], [mdpi.com]
RNG: choose GPU‑friendly, high‑quality RNGs and batch generation to reduce memory stalls; practical performance comparisons across GPU frameworks (CUDA/Numba/CuPy) are available. [mdpi.com]
Learn from established GPU MC projects (e.g., WARP, MCGPU) that leverage table lookups and analytical sampling to achieve large speedups while preserving physics fidelity. [github.com], [cdrh-rst.fda.gov]


Validation & traceability


Tag your build with the specific ICRP‑107 commit/release used (from the OpenGATE icrp107-database repo) and record the nuclide list used to generate PDFs/CDFs. [github.com]
Validate sampled line yields and integrated beta spectra against the ICRP‑107 tables and check mean free paths against canonical formulas/tutorials before production runs. [icrp.org], [aapm.org]


Delivered report
I generated a detailed, ready‑to‑use Markdown report that summarizes the differences between PDF and CDF and provides recommendations tailored to GPU Monte Carlo dosimetry with ICRP‑107:

👉 CDF_PDF_MC_suggestion.md

It includes:

A concise PDF vs CDF table and guidance on when to use each in internal dosimetry. [us-prod.as...rosoft.com], [aapm.org]
A recommended workflow for source term (ICRP‑107 lines & spectra), free path, and interaction sampling. [icrp.org], [geant4.web.cern.ch], [Delta- and...Wiki - VTT]
GPU implementation tips (alias method, inverse‑CDF, memory layout, event‑based kernels). [link.springer.com], [mdpi.com]
References to OpenGATE icrp107-database for reproducible data ingestion. [github.com]


Want me to add code?
If helpful, I can follow up with:

A CUDA/HIP code skeleton showing alias table construction and inverse‑CDF sampling kernels. [link.springer.com]
A small Python preprocessor that reads OpenGATE icrp107-database JSON and emits GPU‑ready tables (SoA) with provenance metadata. [github.com]

Let me know your target GPU stack (CUDA vs ROCm/HIP) and whether you plan to integrate with OpenGATE or a custom transport so I can tailor the templates.