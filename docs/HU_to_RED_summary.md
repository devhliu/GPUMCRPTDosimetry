# HU to Electron Density Mapping Summary

## 0) HU Definition & Conversion Formula

**Formula:**
```
HU = 1000 × (μ_tissue − μ_water) / μ_water
```

---

## 1) Five-Compartment Model

| Compartment | HU Range | Density (g/cm³) | RED | Elemental Composition |
|---|---:|---:|---:|---|
| Air | −1000 | 0.0012 | ~0.001 | N 75.5%, O 23.2%, Ar 1.3% |
| Lung | −850 to −910 | 0.205–0.507 | 0.20–0.50 | Similar to soft tissue, lower density |
| Fat | −100 to −50 | 0.95 | 0.949 | H 11.4%, C 59.8%, N 0.7%, O 27.8% |
| Muscle | +10 to +40 | 1.06 | 1.043 | H 10.2%, C 12.3%, N 3.5%, O 72.9% |
| Soft Tissue | 0 to +50 | 1.00 | 1.000 | H 10.1%, C 11.1%, N 2.6%, O 76.2% |
| Bone | +150 to +3000 | 1.16–1.85 | 1.117–1.695 | H 3.4%, C 15.5%, N 4.2%, O 43.5%, P 10.3%, Ca 22.5% |

---

## 2) Fine Composition Model

### Soft Tissue Subclasses
| Tissue | HU | Density | RED | Composition |
|---|---:|---:|---:|---|
| Blood | +30 to +45 | 1.06 | ~1.05 | H 10.2%, C 11%, N 3.3%, O 74.5% |
| Brain | +20 to +45 | 1.04 | ~1.03 | H 10.7%, C 14.5%, N 2.2%, O 71.2% |
| Liver | +40 to +60 | 1.07 | 1.052 | Similar to soft tissue |

### Bone Classes
| Bone | HU | Density | RED | Composition |
|---|---:|---:|---:|---|
| Trabecular | +150 to +700 | 1.16 | 1.117 | Lower mineral content |
| Cortical | +700 to +3000 | 1.85–1.92 | >1.6 | High Ca/P fractions |

### Contrast & Metals
- **Iodinated Contrast:** ~26 HU per mg I/mL at 120 kVp; arterial phase ~250–350 HU.
- **Metals:** HU saturates at ~3071 in 12-bit; extended HU: Ti ~8088 HU, Steel ~9971 HU.

---

## Key Notes
- HU depends on scanner, kVp, and reconstruction.
- RED calibration uses phantom-based or stoichiometric methods.
- Elemental compositions from ICRU-44/46.

**References:** ICRU-44/46, NIST tables, Radiopaedia, CIRS phantom data.


| Compartment | HU Range | Density (g/cm³) | RED | Elemental Composition |
|---|---:|---:|---:|---|
| Air | −1000 | 0.0012 | ~0.001 | N 75.5%, O 23.2%, Ar 1.3% |
| Lung | −850 to −910 | 0.205–0.507 | 0.20–0.50 | Similar to soft tissue, lower density |
| Fat | −100 to −50 | 0.95 | 0.949 | H 11.4%, C 59.8%, N 0.7%, O 27.8% |
| Muscle | +10 to +40 | 1.06 | 1.043 | H 10.2%, C 12.3%, N 3.5%, O 72.9% |
| Soft Tissue | 0 to +50 | 1.00 | 1.000 | H 10.1%, C 11.1%, N 2.6%, O 76.2% |
| Bone | +150 to +3000 | 1.16–1.85 | 1.117–1.695 | H 3.4%, C 15.5%, N 4.2%, O 43.5%, P 10.3%, Ca 22.5% |
| Trabecular Bone | +150 to +700 | 1.16 | 1.117 | Lower mineral content |
| Cortical Bone | +700 to +3000 | 1.85–1.92 | >1.6 | High Ca/P fractions |