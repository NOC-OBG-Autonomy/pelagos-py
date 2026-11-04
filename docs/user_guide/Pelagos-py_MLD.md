# Mixed Layer Depth

```{image} ../_static/mld/mld_demo.png
:alt: Mixed layer depth from temperature and density over the top 200 m of the demo deployment
:width: 100%
```

## Overview

The {doc}`Mixed Layer Depth </api/pelagos_py/steps/processing/mixed_layer_depth/index>` step finds the mixed layer depth (MLD) of each profile with the threshold method of de Boyer Montégut et al. (2004). Several CHLA quenching corrections use it to define the layer they correct.

It needs `PROFILE_NUMBER` (from {doc}`Find Profiles <Pelagos-py_PROFILES>`) and `DEPTH` (from Derive CTD), plus either `ASAL` and `CONS_TEMP` (density method) or `TEMP` (temperature method). Inputs are read from their `_ADJUSTED` version where one exists.

---

# 1. Method

```{figure} ../_static/mld/mld_method.png
:alt: The threshold method on simulated density and temperature profiles
:width: 100%

The threshold method on simulated profiles, using the default parameters.
```

For each profile:

1. **Reference value.** The median of the samples within 2.5 m of `reference_depth` (10 m). Using the median means a single noisy sample cannot shift it.
2. **Threshold crossing.** Moving down from `reference_depth`, the MLD is the first depth where the value departs from the reference by the threshold:
   - **Density:** potential density (σ<sub>0</sub>) **increases** by `density_threshold` (0.03 kg/m<sup>3</sup>).
   - **Temperature:** temperature **changes in either direction** by `temp_threshold` (0.2 °C).
3. **No result.** If no sample lies near the reference depth, or the threshold is never crossed, the profile's MLD is NaN.

The density method uses σ<sub>0</sub> (potential density referenced to 0 dbar), calculated here from `ASAL` and `CONS_TEMP`. It does not use the in-situ `DENSITY`, because the pressure effect alone would cross the threshold within a few metres of the surface. σ<sub>0</sub> is not saved to the dataset.

Samples flagged by `calculation_flag_filter` take no part in the search, so a bad value cannot set the MLD. The result is still written to every sample of the profile.

---

# 2. Outputs

| Variable | Meaning |
|---|---|
| `MLD` | Mixed layer depth of the profile in m (positive down, like `DEPTH`), repeated on every sample of the profile; NaN where undefined |
| `MLD_BOOL` | `0` above the MLD, `1` at or below it, NaN where undefined |

Both get a `_QC` variable built from the QC of their inputs.

---

# 3. Configuration

All parameters are optional.

| Parameter | Default | Meaning |
|---|---|---|
| `method` | `auto` | `density`, `temp`, or `auto` (density if `ASAL` and `CONS_TEMP` exist, otherwise temperature) |
| `reference_depth` | `10` | Near-surface reference depth (m) |
| `density_threshold` | `0.03` | Density increase (kg/m<sup>3</sup>) marking the MLD |
| `temp_threshold` | `0.2` | Temperature change (°C) marking the MLD |

Default configuration:

```yaml
- name: Mixed Layer Depth
```

With `diagnostics: true` the step draws the figure at the top of this page: temperature and σ<sub>0</sub> over the top 200 m, with the MLD from both methods for comparison.

---

# References

- de Boyer Montégut, C., Madec, G., Fischer, A. S., Lazar, A., & Iudicone, D. (2004). Mixed layer depth over the global ocean: An examination of profile data and a profile-based climatology. *Journal of Geophysical Research: Oceans*, 109(C12), C12003. https://doi.org/10.1029/2004JC002378
