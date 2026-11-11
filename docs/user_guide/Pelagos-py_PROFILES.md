# Profile Finding

```{image} ../_static/profiles/find_profiles_demo.png
:alt: One day of the demo deployment coloured by scientific phase
:width: 100%
```

## Overview

The {doc}`Find Profiles </api/pelagos_py/steps/processing/find_profiles/index>` step splits a deployment into profiles. It labels every sample with an OceanGliders OG1 phase (ascent, descent, surfacing, ...) using only pressure and time, then numbers the profiles and dive cycles. Many steps depends on the data being split into profiles.

The step adds:

| Variable | Meaning |
|---|---|
| `PHASE_SCI` | OG1 phase of each sample (see §2) |
| `PROFILE_NUMBER` | Number of the ascent or descent the sample belongs to; NaN outside profiles |
| `PROFILE_DIRECTION` | `1` descent, `-1` ascent, `0` surfacing, parking or propelled, NaN otherwise |
| `CYCLE` | Dive cycle number, starting at 1 |
| `PROFILE_GRADIENT` | Smoothed vertical velocity used for the labelling (dbar/s for `PRES`) |

Each also gets a `_QC` variable (see §1.4).

---

# 1. Algorithm

```{figure} ../_static/profiles/find_profiles_method.png
:alt: The three stages of profile finding on a simulated dive
:width: 100%

The three stages on a simulated dive with sensor noise, using the default parameters.
```

## 1.1 Vertical velocity

Pressure is smoothed with a rolling mean over `smoothing_window_seconds` (30 s), differentiated in time, and the result smoothed again with a rolling median to remove spikes. Both windows are in seconds, so they work the same for any sampling rate.

A time gap longer than `gap_threshold_minutes` (5 min) splits the record into separate chunks. Velocity is never computed across a gap and no phase or profile runs across one, so a profile whose data stops early stays a valid profile of its own.

## 1.2 Ascent and descent

Samples faster than `velocity_threshold` (0.033 dbar/s) are descent (going down) or ascent (going up). A run shorter than `min_duration_seconds` (60 s) is too short to trust and is reset to unknown. This pass runs first, and nothing later overrides it.

## 1.3 Everything else

Samples still unknown are labelled in this order:

1. **Surfacing** — a run whose median depth is within `surfacing_depth_threshold` (2 dbar) of the surface and lasts at least `min_duration_seconds`.
2. **Parking** or **propelled** — a run away from the surface lasting at least `min_transect_duration_seconds` (300 s). `transect_phase` chooses between the two: `auto` uses propelled when the file name contains `ALR` (platforms with thrusters), and parking otherwise (for gliders).
3. **Inflection** — the single deepest (or shallowest) sample of a turn between a descent and an ascent, including turns in mid-water. A turn shallower than `surfacing_depth_threshold` is surfacing instead.
4. **Transition** — the remaining samples on the shoulders of a turn, between the inflection or surfacing and the ascent or descent either side.

Anything left after this stays unknown (`0`).

## 1.4 Flagged samples

Samples whose pressure is flagged probably bad (`3`), bad (`4`), missing (`9`) or interpolated (`8`) are left out of the calculations. Afterwards they take the labels of the samples either side when those agree.

---

# 2. OG1 phases

`PHASE_SCI` uses the OceanGliders OG1 phase vocabulary, adapted for the toolbox:

| Code | Phase | Meaning |
|---|---|---|
| 0 | Unknown | Could not be labelled |
| 1 | Ascent | Rising faster than `velocity_threshold` |
| 2 | Descent | Sinking faster than `velocity_threshold` |
| 3 | Surfacing | At the surface, including shallow turns |
| 4 | Parking | Holding depth away from the surface (drifting) |
| 5 | Inflection | The deepest or shallowest point of a turn |
| 6 | Propelled | Holding depth under power (e.g. ALR) |
| 7 | Transition | The shoulders either side of a turn |

```{figure} ../_static/profiles/find_profiles_phases.png
:alt: A simulated record showing every phase and the derived numbering
:width: 100%

A simulated record with a dive, a mid-water turn, a parking stretch and a data gap. The lower panels show the profile numbering, direction and cycle derived from the phases.
```

---

# 3. Profiles, direction and cycles

- **`PROFILE_NUMBER`.** Each ascent or descent is its own profile; an ascent does not need a matching descent (though there should almost always be one). A profile also includes the transition samples next to it, and the bottom inflection where it starts. Surfacing, parking and propelled samples belong to no profile.
- **`PROFILE_DIRECTION`.** Follows the phase: `1` for descent, `-1` for ascent, `0` for surfacing, parking and propelled. Inflection, transition and unknown samples are NaN.
- **`CYCLE`.** A cycle runs from leaving the surface to the end of the next surfacing. It only advances when the platform leaves the surface after having dived, so mid-water turns, parking and data gaps stay within one cycle. The figure above has three cycles.

---

# 4. Configuration

All parameters are optional; the defaults suit typical glider data.

| Parameter | Default | Meaning |
|---|---|---|
| `depth_column` | `PRES` | Pressure or depth variable to use |
| `smoothing_window_seconds` | `30` | Rolling-mean window before differentiating |
| `velocity_threshold` | `0.033` | Speed (units/s) above which a sample is ascent or descent |
| `min_duration_seconds` | `60` | Shortest ascent, descent or surfacing run kept |
| `gap_threshold_minutes` | `5` | Time gap that splits the record |
| `surfacing_depth_threshold` | `2.0` | Runs shallower than this count as surfacing |
| `min_transect_duration_seconds` | `300` | Shortest parking or propelled run |
| `transect_phase` | `auto` | `parking`, `propelled`, or `auto` (propelled for ALR files) |

Default configuration:

```yaml
- name: Find Profiles
```

The default pipeline follows it with {doc}`Valid Profile QC </api/pelagos_py/steps/quality_control/valid_profile_qc/index>`, which flags profiles that are too short or never reach a given depth range.

---

# References

- OceanGliders Community. OG1 format user manual: `phase` vocabulary. https://github.com/OceanGlidersCommunity/OG-format-user-manual/blob/main/vocabularyCollection/phase.md
