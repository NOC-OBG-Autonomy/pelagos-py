# Cross-Calibration

## Overview

Two sensors measuring the same water rarely agree exactly. Cross-calibration adjusts one sensor to match a reference, such as another glider, a ship CTD cast or a lab sample.

pelagos-py applies a cross-calibration with the {doc}`Correct Values </api/pelagos_py/steps/input_output/correct_values/index>` step. The step does not compare against the reference itself. For now you work out the correction from your own comparison and enter it as a slope and an intercept.

The default pipeline runs it twice (for `TEMP` and `PSAL_ADJUSTED`) as placeholders that change nothing until real values are entered.

---

# 1. How the correction works

Each value is corrected as

```
corrected = slope * value + intercept
```

- **Input.** If `X_ADJUSTED` already exists it is corrected instead of `X`, so corrections build on each other.
- **Time window.** With `time_start` and/or `time_end`, only samples in that window (bounds included) are corrected; the rest keep their value. Use this when a sensor drifts or is swapped part-way through a deployment.
- **Expected range check.** With `expected_range: [min, max]`, the step first takes the median of the input (within the time window). If the median is already inside the range, the data is assumed correct and nothing is changed. Without `expected_range` the correction is always applied. This is an advanced use case to automatically apply corrections to data with known errors.
- **No-op.** `slope: 1` and `intercept: 0` change nothing, so the step skips.

All samples are corrected, including flagged ones, and their QC flags are not changed.

---

# 2. Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `target_variable` | (required) | Variable to correct, e.g. `TEMP` |
| `slope` | `1.0` | Multiplies each value |
| `intercept` | `0.0` | Added after scaling |
| `output_as` | `X_ADJUSTED` | Name, or list of names, to write the result to |
| `time_start`, `time_end` | none | Only correct samples in this time window |
| `expected_range` | none | Skip when the median is already inside `[min, max]` |
| `corrected_units` | none | New `units` attribute, set only when corrected |
| `append_description` | none | Text added to the `comment` attribute |
| `overwrite_description` | none | Text replacing the `comment` attribute |
| `optional` | `false` | Skip, rather than stop, if the variable is missing |

A temperature offset against a reference:

```yaml
- name: Correct Values
  parameters:
    target_variable: TEMP
    slope: 1.0
    intercept: -0.012
    append_description: "Offset to ship CTD"
```

A salinity correction for part of a deployment only:

```yaml
- name: Correct Values
  parameters:
    target_variable: PSAL_ADJUSTED
    slope: 1.0
    intercept: 0.008
    time_start: "2024-06-15T00:00:00"
```

With `diagnostics: true` the step plots the raw and corrected values, and their difference.

---

# 3. Outputs

| Output | Change |
|---|---|
| `X_ADJUSTED` (or `output_as`) | Corrected values; the raw `X` is never changed |
| its `_QC` | A copy of the input's QC flags, when a new variable is written |
| `units` attribute | Set to `corrected_units`, if given and a correction ran |
| `comment` attribute | Updated by `append_description` or `overwrite_description` |

If the correction is skipped (expected range or no-op) and no `output_as` or description is set, the dataset is left unchanged.
