# Pressure Processing

```{image} ../_static/pressure/pressure_track.png
:alt: Pressure against time for three days of the demo deployment
:width: 100%
```

## Overview

Pressure (`PRES`, dbar) is measured directly by the CTD. It is used to find profiles, to derive `DEPTH`, and in the salinity and density calculations, so it is quality controlled early in the pipeline. Processing consists of:

1. **Quality control** — range and stuck-value tests, with flags shared across the CTD triad (`PRES`, `TEMP`, `CNDC`).
2. **Depth** — `PRES` itself is not interpolated; `DEPTH_ADJUSTED` derived from it is filled onto every row.

---

# 1. Quality control

Pressure QC uses {doc}`Range QC </api/pelagos_py/steps/quality_control/range_qc/index>` and {doc}`Stuck Value QC </api/pelagos_py/steps/quality_control/stuck_value_qc/index>`. Default configuration:

```yaml
qc_settings:
  range qc:
    variable_ranges:
      PRES:
        3: [-2.4, -5, inside]   # probably bad
        4: [-5, -.inf, inside]  # bad
    also_flag:
      PRES: [CNDC, TEMP]
  stuck value qc:
    variables:
      PRES: 2                   # max repeats
    also_flag:
      PRES: [CNDC, TEMP]
```

A small negative pressure near the surface is normal sensor offset; large negative values indicate a fault. No upper limit is set, so deep dives are never clipped. A frozen pressure reading indicates a stuck sensor. Because `PRES`, `TEMP` and `CNDC` come from the same instrument, `also_flag` copies any pressure flag onto the other two.

No spike test is applied: pressure changes smoothly with the platform's vertical speed.

---

# 2. Depth

`PRES` keeps only its measured values. {doc}`Derive CTD <Pelagos-py_DERIVED>` derives `DEPTH_ADJUSTED` from it and fills rows with no usable pressure (flagged `3`, `4` or `9`, or without a CTD sample) by linear interpolation in time, flagging them interpolated (`8`). This gives every sample a depth for profile finding and plotting.

---

# References

- Wong Annie, Keeley Robert, Carval Thierry, Argo Data Management Team (2025). Argo Quality Control Manual for CTD and Trajectory Data. Ifremer. https://doi.org/10.13155/33951
- IOC, SCOR, and IAPSO (2010), *The International Thermodynamic Equation of Seawater – 2010: Calculation and Use of Thermodynamic Properties*, Intergovernmental Oceanographic Commission, Manuals and Guides No. 56, 196 pp., UNESCO, Paris. Available from http://www.TEOS-10.org
