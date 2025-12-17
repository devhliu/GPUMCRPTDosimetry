# Decay Database (v1): ICRP107 JSON Files Only

This codebase uses **local ICRP107 JSON nuclide files** as the only decay database source.

## Location
Place JSON files here:

```
GPUMCRPTDosimetry/data/decaydb/icrp107/
  Lu-177.json
  Y-90.json
  Ac-225.json
  Pb-212.json
  At-211.json
  Tb-161.json
  Zr-89.json
  ...
```

The JSON format matches OpenGATE `icrp107-database` nuclide records:
- `name`
- `half_life`
- `time_unit`
- `emissions` with keys like `gamma`, `X`, `IE`, `auger`, `alpha`, `b-spectra`

## Configuration
In `configs/example_simulation.yaml`:

```yaml
decaydb:
  type: "icrp107_json"
  path: "data/decaydb/icrp107"

nuclide:
  name: "Lu-177"
```

## Beta spectrum sampling
`b-spectra` is treated as a **PDF-like tabulation** and normalized to a CDF on-the-fly by trapezoidal integration.

## Notes
- Neutrinos are not represented in the JSON and are never scored.
- Discrete emission sampling uses an accepted v1 simplification:
  - ≤1 line per history per discrete category (gamma, X, IE, auger, alpha).