# Pressure Processing

## Overview

Pressure (`PRES`, dbar) is measured directly by the CTD. It is used to find profiles, to derive `DEPTH`, and in the salinity and density calculations, so it is quality controlled early in the pipeline.

pelagos-py applies no physical correction to pressure (no calibration, offset or drift correction). Processing is limited to:

1. **Quality control** — range and stuck-value tests, with flags shared across the CTD triad (`PRES`, `TEMP`, `CNDC`).
2. **Interpolation** — flagged and missing pressure values are filled by linear interpolation in time.

---

# 1. Quality control

Pressure QC uses the generic **Range QC** and **Stuck Value QC** steps (see the **Quality Control** documentation for the algorithms). Default configuration:

```yaml
qc_settings:
  range qc:
    variable_ranges:
      PRES:
        3: [-2.4, -5, inside]   # probably bad between -2.4 and -5 dbar
        4: [-5, -.inf, inside]  # bad below -5 dbar
    also_flag:
      PRES: [CNDC, TEMP]
  stuck value qc:
    variables:
      PRES: 2                   # more than 2 identical consecutive values are bad
    also_flag:
      PRES: [CNDC, TEMP]
```

A small negative pressure near the surface is normal sensor offset; large negative values indicate a fault. No upper limit is set, so deep dives are never clipped. A frozen pressure reading indicates a stuck sensor. Because `PRES`, `TEMP` and `CNDC` come from the same instrument, `also_flag` copies any pressure flag onto the other two.

No spike test is applied: pressure changes smoothly with the platform's vertical speed.

---

# 2. Interpolation

After QC, the **Interpolate Data** step fills pressure values flagged probably bad (`3`), bad (`4`) or missing (`9`) by linear interpolation in time, and re-flags them as interpolated (`8`). This gives every sample a usable pressure for `DEPTH` and profile finding.

```yaml
- name: Interpolate Data
  parameters:
    variables:
      PRES: [3, 4, 9]
```

---

# References

- Wong Annie, Keeley Robert, Carval Thierry, Argo Data Management Team (2025). Argo Quality Control Manual for CTD and Trajectory Data. Ifremer. https://doi.org/10.13155/33951
- IOC, SCOR, and IAPSO (2010), *The International Thermodynamic Equation of Seawater – 2010: Calculation and Use of Thermodynamic Properties*, Intergovernmental Oceanographic Commission, Manuals and Guides No. 56, 196 pp., UNESCO, Paris. Available from http://www.TEOS-10.org
