# Position and Time Processing

```{image} ../_static/position/position_track.png
:alt: Glider track of the demo deployment coloured by time
:width: 100%
```

## Overview

Time (`TIME`) and position (`LATITUDE`, `LONGITUDE`) place every measurement. They are checked first in the pipeline, before any science variable is processed. Processing consists of:

1. **Time filtering on load** — records outside the deployment are dropped.
2. **Quality control** — impossible dates, impossible positions, positions on land and impossible speeds are flagged.
3. **Interpolation** — short gaps in position are filled by linear interpolation in time.

---

# 1. Time filtering on load

{doc}`Load OG1 </api/pelagos_py/steps/input_output/load_data/index>` removes records whose `TIME` is outside the deployment window. By default this runs from `DEPLOYMENT_TIME` (or 1 Jan 1990 if the file has none) to the time the pipeline runs.

```yaml
- name: Load OG1
  parameters:
    file_path: glider.nc
    filter_bad_time: true   # default
    data_start: null        # default: deployment
    data_end: null          # default: now
```

Missing times (`NaT`) stop the pipeline, and a warning is logged if `TIME` is not increasing.

If the file has no `LATITUDE`/`LONGITUDE`, {doc}`Prepare OG1 </api/pelagos_py/steps/input_output/prepare_og1/index>` renames them from `LATITUDE_GPS`/`LONGITUDE_GPS` (or `ALATPT01`/`ALONPT01`) if available.

---

# 2. Quality control

All four tests flag bad values `4` and missing values `9`. Default configuration:

```yaml
- name: Apply QC
  parameters:
    qc_settings:
      impossible date qc: {}
      impossible location qc: {}
      position on land qc: {}
      impossible speed qc: {}
```

## 2.1 Impossible date

{doc}`Impossible Date QC </api/pelagos_py/steps/quality_control/impossible_date_qc/index>` flags `TIME` before 1 Jan 1985 or after the current time. It has no parameters.

## 2.2 Impossible location

{doc}`Impossible Location QC </api/pelagos_py/steps/quality_control/impossible_location_qc/index>` flags `LATITUDE` outside ±90° and `LONGITUDE` outside ±180°. Each coordinate is flagged on its own. It has no parameters.

## 2.3 Position on land

{doc}`Position On Land QC </api/pelagos_py/steps/quality_control/position_on_land_qc/index>` flags `LATITUDE` and `LONGITUDE` together where the position falls on land, using the Natural Earth 1:110m land polygons. Positions very close to the coast can be missed or incorrectly flagged.

## 2.4 Impossible speed

{doc}`Impossible Speed QC </api/pelagos_py/steps/quality_control/impossible_speed_qc/index>` flags fixes that imply an impossible horizontal speed. Each fix is compared with the last good fix at least `min_interval` seconds earlier; if the speed between them reaches `threshold`, the fix is flagged on `TIME`, `LATITUDE` and `LONGITUDE`.

```yaml
impossible speed qc:
  threshold: 3      # m/s
  min_interval: 60  # s
```

---

# 3. Interpolation

The GPS and science sensors do not sample at the same times, so many samples have no position. The {doc}`Interpolate Data </api/pelagos_py/steps/processing/interpolate_data/index>` step fills positions flagged `3`, `4` or `9` by linear interpolation in time and flags them interpolated (`8`).

```yaml
- name: Interpolate Data
  parameters:
    variables:
      PRES: [3, 4, 9]
      LATITUDE: [3, 4, 9]
      LONGITUDE: [3, 4, 9]
```

Only gaps of up to `max_interp_time` (5 minutes) between real values are filled. Values before the first or after the last real value are never filled. `TIME` itself is never interpolated.
