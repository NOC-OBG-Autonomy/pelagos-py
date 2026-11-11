# Derived CTD Variables

```{image} ../_static/derived/derived_section.png
:alt: Density against pressure and time for the demo deployment
:width: 100%
```

## Overview

The {doc}`Derive CTD </api/pelagos_py/steps/processing/derive_ctd/index>` step calculates depth, salinity, conservative temperature and density from the measured pressure, temperature and conductivity. It uses the TEOS-10 equation of seawater through the GSW (Gibbs SeaWater) library.

Each output is written as `<name>_ADJUSTED` (e.g. `PSAL_ADJUSTED`), so any raw version in the input file is kept. Inputs are read from their `_ADJUSTED` version where one exists, so corrected `TEMP` or `CNDC` feed straight through. A `PSAL_ADJUSTED` already made by Salinity Adjustment is never overwritten.

---

# 1. Variables

| Output | gsw function | Inputs | Units |
|---|---|---|---|
| `DEPTH_ADJUSTED` | `-gsw.z_from_p` | `PRES`, `LATITUDE` | m |
| `PSAL_ADJUSTED` | `gsw.SP_from_C` | `CNDC`, `TEMP`, `PRES` | 1 (PSS-78) |
| `ASAL_ADJUSTED` | `gsw.SA_from_SP` | `PSAL`, `PRES`, `LONGITUDE`, `LATITUDE` | g/kg |
| `CONS_TEMP_ADJUSTED` | `gsw.CT_from_t` | `ASAL`, `TEMP`, `PRES` | °C |
| `DENSITY_ADJUSTED` | `gsw.rho` | `ASAL`, `CONS_TEMP`, `PRES` | kg/m<sup>3</sup> |

Variables are derived in the order of the table, so later ones can use earlier ones from the same step. Each output's `_QC` is combined from the QC of its inputs.

**Depth.** `gsw.z_from_p` returns height, which is negative below the surface. pelagos-py negates it, so `DEPTH` is positive down, as in OG1.

**Practical salinity.** GSW expects conductivity in mS/cm. The step reads the `units` attribute of `CNDC`: values already in mS/cm are used as they are, anything else is treated as S/m and multiplied by 10. A missing `units` attribute is also treated as S/m.

**Absolute salinity.** Converting practical to absolute salinity adds a small correction for dissolved material that varies across the ocean. GSW looks it up from position, so `LONGITUDE` and `LATITUDE` are needed.

**Density.** `DENSITY` is in-situ density at the measured pressure. It increases with depth from compression alone, so it is not suitable for finding the mixed layer; {doc}`Mixed layer depth <Pelagos-py_MLD>` calculates potential density (σ<sub>0</sub>) itself.

---

# 2. Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `to_derive` | required | List of outputs: any of `DEPTH`, `PSAL`, `ASAL`, `CONS_TEMP`, `DENSITY`. `DEPTH_ADJUSTED` is filled onto every row by time interpolation (flagged `8` where filled) |

The step needs `TIME`, `LATITUDE`, `LONGITUDE`, `CNDC`, `PRES` and `TEMP` in the dataset, whichever outputs are requested.

The default pipeline runs it twice. First, `DEPTH` only, because Find Profiles and the profile QC need it:

```yaml
- name: Derive CTD
  parameters:
    to_derive: [DEPTH]
```

Then, after Salinity Adjustment has written `PSAL_ADJUSTED`, the rest are derived from the corrected salinity:

```yaml
- name: Derive CTD
  parameters:
    to_derive: [ASAL, CONS_TEMP, DENSITY]
```

---

# References

- IOC, SCOR, and IAPSO (2010), *The International Thermodynamic Equation of Seawater – 2010: Calculation and Use of Thermodynamic Properties*, Intergovernmental Oceanographic Commission, Manuals and Guides No. 56, 196 pp., UNESCO, Paris. Available from http://www.TEOS-10.org
- GSW-Python: the Python implementation of the Gibbs SeaWater (GSW) Oceanographic Toolbox of TEOS-10. https://teos-10.github.io/GSW-Python/
