# Phase 9.1: Log-uniform energy bin computation (GPU-side)

Your photon physics uses **log-uniform bins** spanning ~keV to GeV, so uniform bins are inefficient.

## Metadata used
- `common_log_E_min` (float): `ln(E_min)` where `E` is in **MeV**
- `common_log_step_inv` (float): `1 / d(ln E)`
- `NB` (int): number of energy bins

## Kernel
`compute_ebin_log_uniform_kernel` computes:

```text
ebin = clamp(int((ln(E) - common_log_E_min) * common_log_step_inv), 0, NB-1)
```

Nonpositive energies are clamped safely (log protected by `max(E, 1e-30)`).

## Where used
Phase 9 integration uses this for:
- fluorescence photons emitted by atomic relaxation (X-rays)
- Auger electrons (if electron tables use the same energy bin metadata)

If electron transport uses separate binning metadata, provide:
- `e_log_E_min`, `e_log_step_inv`, `NB_e`
and call a second kernel instance with those constants.