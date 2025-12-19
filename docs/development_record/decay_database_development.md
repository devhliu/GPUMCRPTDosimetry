# Development Record: Decay Database

This document describes the data source and format for the radionuclide decay database used in the GPUMCRPTDosimetry project.

## ICRP-107 JSON Database

The project has standardized on using a local directory of JSON files based on the **ICRP-107 database** as the sole source for decay data. This approach replaced a previous, now-deprecated method that used CSV/YAML-based imports.

### Location and Format

The JSON files for each nuclide are expected to be located in the following directory:
`data/decaydb/icrp107/`

Each file (e.g., `Lu-177.json`, `Y-90.json`) represents a single radionuclide and contains the following information, in a format that matches the OpenGATE `icrp107-database`:

*   **`name`**: The name of the nuclide.
*   **`half_life`**: The half-life of the nuclide.
*   **`time_unit`**: The unit of the half-life (e.g., "d" for days).
*   **`emissions`**: An object containing the different emission types, with keys such as:
    *   `gamma`
    *   `X` (X-rays)
    *   `IE` (Internal conversion electrons)
    *   `auger` (Auger electrons)
    *   `alpha`
    *   `b-spectra` (Beta spectra)

### Configuration

The simulation is configured to use this database via the main YAML configuration file:

```yaml
decaydb:
  type: "icrp107_json"
  path: "data/decaydb/icrp107"

nuclide:
  name: "Lu-177"
```

### Beta Spectrum Sampling

The beta spectra (`b-spectra`) are provided as a PDF-like tabulation in the JSON files. The simulation normalizes this data into a CDF on-the-fly using trapezoidal integration to allow for proper sampling.

### Notes and Simplifications

*   Neutrino emissions are not included in the JSON files and are not tracked or scored in the simulation.
*   For discrete emissions (gamma, X-ray, etc.), the v1 implementation uses a simplification where a maximum of one emission line per category is sampled per decay event.
