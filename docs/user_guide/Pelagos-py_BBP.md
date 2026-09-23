# Particulate Backscattering (b<sub>bp</sub>) Processing

## Overview

Optical backscattering (b<sub>b</sub>) is an inherent optical property defined as the limit, as path length approaches zero, of the fraction of light is scattered in a backwards direction, per unit distance traveled through the water. It is a fundamental inherent optical property of seawater, critical to ocean color, and important for visibility and light penetration in the ocean. Particulate optical backscattering (b<sub>bp</sub>), is the backscattering by particulate matter. As measured by autonomous oceanographic platforms, (b<sub>bp</sub>) provides a proxy for particle concentration and size. In the open ocean, at low inorganic particle concentrations, it is a useful proxy for particulate organic carbon (POC) concentration and and is a required optical reference for several of the non-photochemical quenching (NPQ) correction methods used in chlorophyll-*a* processing. Raw backscattering sensors measure the volume scattering function at a single angle and wavelength (beta, in m<sup>-1</sup> sr<sup>-1</sup>), which must be converted to the particulate backscattering coefficient (b<sub>bp</sub>, in m<sup>-1</sup>) and separated from isolated spikes before it can be interpreted or used downstream.

pelagos-py implements a short, focused backscatter processing workflow that combines generic quality-control algorithms with a variable-specific physical conversion and despiking step. Unlike chlorophyll, backscatter cannot be dark corrected in situ, so factory dark values are used by default without adjustment. b<sub>bp</sub> is processed once, early in the pipeline, and its despiked baseline is then consumed as an input by several chlorophyll NPQ correction methods (see [4. Use in chlorophyll NPQ correction](#4-use-in-chlorophyll-npq-correction) and the **Chlorophyll-*a* Processing** documentation).

The backscatter processing workflow consists of three sequential stages:

1. **Initial quality control** — applies the generic Range QC and Stuck Value QC modules to the raw sensor signal to identify grossly implausible or frozen readings. Deliberately, no spike test is applied at this stage: backscatter spikes correspond to real particles or aggregates rather than sensor artefacts, and are separated out (not discarded) in stage 3.

2. **BBP from Beta** — converts the raw volume scattering function (beta, m<sup>-1</sup> sr<sup>-1</sup>) to the total (particulate + seawater) backscattering coefficient (b<sub>bp</sub>, m<sup>-1</sup>) using the pure-seawater scattering model of Zhang et al. (2009) and the chi-factor scaling of Sullivan et al. (2013). Note that Zhang et al. (2009) has been superceded by Hu et al. (2019), which adds a small pressure effect, so this should eventually be changed.

3. **Isolate BBP Spikes** — applies a despiking routine (following Briggs et al., 2011) directly to b<sub>bp</sub> to separate a smooth baseline from isolated spikes, producing `<var>_BASELINE` and `<var>_SPIKES`. The baseline is the variable subsequently used as the optical reference by chlorophyll NPQ correction methods.

Each processing stage is described in detail below, including the default pelagos-py configuration, the physical/scientific rationale, and implementation details that depart from the equivalent chlorophyll processing steps.

---

# 1. Initial quality control

Backscatter QC is performed with the same generic **Range QC** and **Stuck Value QC** modules used elsewhere in pelagos-py (see the **Quality Control** documentation for the full algorithm descriptions). Unlike the chlorophyll initial range test, these QC steps have no code-level defaults of their own — the values below are drawn from pelagos-py's reference deployment configurations (Churchill, Nelson, ALR) as a documented starting point.

## 1.1 Range test

Recommended configuration for open ocean:

```yaml
qc_settings:
  range qc:
    variable_ranges:
      BETA_BACKSCATTERING700:      # or BBP700, if the platform already reports beta under that name
        3: [0, 0.01, outside]        # Negative = dark-offset noise, high = suspect
        4: [-1.0e-4, 0.05, outside]  # Physically absurd / sensor rail
```

Backscatter cannot physically be negative, but a small negative reading is typically dark-offset noise in clear water rather than a hardware fault, so values just below zero are only flagged probably bad (`3`). The wider band (`4`) catches values far enough out to be physically absurd or indicative of a sensor rail. Some platforms (e.g. Nelson) report beta directly under the `BBP700` name rather than a separate `BETA_BACKSCATTERING700` variable; the range test is applied to whichever variable holds the raw signal, and the BBP from Beta step (Section 2) carries the resulting flags across to the converted variable.

## 1.2 Stuck value test

Recommended configuration:

```yaml
qc_settings:
  stuck value qc:
    variables:
      BETA_BACKSCATTERING700: 3
```

A frozen reading (three or more identical consecutive values, using the recommended threshold) indicates a dead or fouled sensor and is flagged probably bad (`3`).

## 1.3 No spike test

Deliberately, **no Spike QC test is applied** to the raw backscatter signal. Large isolated spikes in b<sub>bp</sub> typically correspond to individual particles or aggregates passing through the sensor's sample volume — genuine signal rather than sensor noise. Discarding them via a generic spike test would remove real information; instead, the Isolate BBP Spikes step (Section 3) separates the spike component from the baseline so that both are retained as independent variables.

---

# 2. BBP from Beta

## 2.1 Physical basis

The conversion follows the OOI/WET Labs formulation implemented in `glidertools.flo_functions.flo_bback_total`, which itself follows Zhang et al. (2009) for the pure-seawater scattering contribution and Sullivan et al. (2013) for the particulate chi-factor scaling.

The raw sensor measures the volume scattering function at the instrument's effective scattering angle $\theta$ and wavelength $\lambda$:

$$
\beta(\theta,\lambda) = \beta_{sw}(\theta,\lambda) + \beta_{p}(\theta,\lambda),
$$

the sum of a theoretical seawater-only contribution $\beta_{sw}$ and the contribution of particles $\beta_p$.

**Seawater scattering.** $\beta_{sw}(\theta,\lambda)$ and the total (forward + backward) seawater scattering coefficient $b_{sw}(\lambda)$ are computed from in-situ temperature and practical salinity following the Zhang et al. (2009) pure-seawater model, which combines scattering due to density fluctuations and due to salt-concentration fluctuations, each weighted by a depolarisation-ratio correction ($\delta = 0.039$).

**Particulate scattering.** The particulate contribution at the measurement angle is obtained by subtraction,

$$
\beta_p(\theta,\lambda) = \beta(\theta,\lambda) - \beta_{sw}(\theta,\lambda),
$$

and scaled to the particulate backscattering coefficient (integrated over all backward angles) using the chi factor $\chi$:

$$
b_{bp}(\lambda) = \chi \cdot 2\pi \cdot \beta_p(\theta,\lambda).
$$

**Seawater backscatter.** Because the effective scattering centres in pure seawater are much smaller than the measurement wavelength, seawater scattering is symmetric in the forward and backward directions, so the seawater backscattering coefficient is simply half the total seawater scattering coefficient:

$$
b_{bsw}(\lambda) = \frac{b_{sw}(\lambda)}{2}.
$$

**Output.** The value written to the dataset is the **total** (particulate + seawater) backscattering coefficient,

$$
b_{b}(\lambda) = b_{bp}(\lambda) + b_{bsw}(\lambda),
$$

with units of m<sup>-1</sup>.

> **Note — known issue: `BBP700` currently reports total, not particulate, backscattering**
>
> The output variable's `long_name` attribute reads "Total particulate backscatter", and this is unfortunately accurate: the value returned by `flo_bback_total` — and therefore the value pelagos-py writes to `BBP700` (or whichever `output_as` name is configured) — is the **total** backscattering coefficient $b_{b}$, i.e. particulate **plus** the residual seawater contribution $b_{bsw}$, not $b_{bp}$ alone. This is a bug rather than an intentional design choice: `BBP700` is documented and used throughout this pipeline as a particulate/POC proxy, and the seawater term is never subtracted out.
>
> The added seawater term is a near-constant offset of roughly 3.2 &times; 10<sup>-4</sup> m<sup>-1</sup> at 700 nm (varying only slightly with temperature and salinity). Because it is additive, its effect depends heavily on how much particulate signal is present:
>
> | Regime | true $b_{bp}$ (m<sup>-1</sup>) | reported `BBP700` (m<sup>-1</sup>) | overestimate |
> |---|---|---|---|
> | Deep ocean (>1000 m) | 5 &times; 10<sup>-5</sup> | 3.7 &times; 10<sup>-4</sup> | ~640% |
> | Oligotrophic surface | 3 &times; 10<sup>-4</sup> | 6.2 &times; 10<sup>-4</sup> | ~106% |
> | Temperate surface | 1 &times; 10<sup>-3</sup> | 1.3 &times; 10<sup>-3</sup> | ~32% |
> | Spring bloom | 4 &times; 10<sup>-3</sup> | 4.3 &times; 10<sup>-3</sup> | ~8% |
>
> The error is largest exactly where $b_{bp}$ is smallest — deep water and oligotrophic surface water — so "small relative to the particulate signal" only holds in blooms; elsewhere the added seawater term can exceed the true particulate signal several times over. It also biases the chlorophyll/backscatter ratio used by the backscatter-based NPQ correction methods (see [4. Use in chlorophyll NPQ correction](#4-use-in-chlorophyll-npq-correction)).
>
> **Status**: fix proposed in [pelagos-py PR #162](https://github.com/NOC-OBG-Autonomy/pelagos-py/pull/162), which computes $b_{bp}$ directly ($b_{bp} = 2\pi\chi(\beta_p - \beta_{sw})$, following the BGC-Argo definition) instead of using `flo_bback_total`. Until that PR is merged, any `BBP700` produced by this step should be treated as $b_{b}$ rather than $b_{bp}$; users who need particulate-only backscatter should subtract the seawater term themselves (Section 2.1 above) or apply the fix from that PR.

## 2.2 Default configuration

```yaml
- name: "BBP from Beta"
  parameters:
    apply_to: BBP700
    output_as: BBP700
    theta: 124
    xfactor: 1.076
  diagnostics: false
```

`theta` (124°) and `xfactor` (1.076) are the effective scattering angle and chi factor for a three-channel WET Labs ECO-BB3-type instrument (Sullivan et al., 2013, Table 6.2b). Two-channel ECO instruments (e.g. FLBB, FLNTU) use a different geometry — an effective angle of 140° and a chi factor of 1.096 — and `theta`/`xfactor` should be adjusted accordingly for those sensors.

`apply_to` and `output_as` default to the same name (`BBP700`), so the step overwrites its input in place unless a different `output_as` is configured — see the implementation note below. Deployments that log the raw signal under a distinct name (e.g. Churchill's `BETA_BACKSCATTERING700`) set `apply_to` to that name and `output_as: BBP700` to derive a new variable.

> **Note — wavelength is not a configurable parameter**
>
> Although the step is named for the 700 nm channel and the default output is `BBP700`, the wavelength passed into the underlying seawater-scattering calculation is hard-coded to 700 nm in the current implementation, independent of the `apply_to`/`output_as` variable names. Using this step to derive a variable at a different wavelength (e.g. a 532 nm channel) would apply the 700 nm seawater-scattering correction rather than one calculated for that channel's actual wavelength.

## 2.3 Implementation notes

- **Required inputs**: `TEMP`, `PRAC_SALINITY`, `DEPTH` and the beta variable named by `apply_to`. Gaps in `TEMP` or `PRAC_SALINITY` are left as `NaN` — the step does not gap-fill its inputs — so b<sub>bp</sub> is not derived at those samples and the corresponding output is flagged missing (`9`). An **Interpolate Data** step run beforehand on `TEMP`/`PRAC_SALINITY` produces a gap-free b<sub>bp</sub> record.
- **QC flag filtering**: unlike the Isolate BBP Spikes step (Section 3) and the generic `deep_correction` step, BBP from Beta does **not** automatically exclude samples carrying bad/probably-bad/missing QC flags (`3`, `4`, `9`) from the conversion — every finite `apply_to` value is converted regardless of its QC flag, unless the user explicitly configures `qc_handling_settings` to NaN-out flagged beta first. Flags on the input are still carried across to the output's QC via `generate_qc`, so a flagged input value converts to a flagged output value; the flag simply does not prevent the arithmetic conversion itself.
- **In-place overwrite**: if `apply_to` and `output_as` are the same (the default), the QC for the output variable is **not** regenerated — the pre-existing `_QC` array is left untouched and a warning is logged. This means that, for platforms using the default in-place configuration, any new missing (`9`) flags arising from `TEMP`/`PRAC_SALINITY` gaps during the conversion are not automatically reflected in `BBP700_QC` unless `output_as` is set to a distinct variable name.
- **DEPTH is never modified** by this step, consistent with its treatment throughout the pipeline.

## 2.4 Figure placeholder

```{figure} ../_static/bbp/bbp_from_beta_boxplot.png
:alt: Placeholder for the Beta vs bbp diagnostic box plot.
:width: 70%

**Placeholder.** Box plot comparing the value distribution of the raw beta input and the converted b<sub>bp</sub> output.
```

---

# 3. Isolate BBP Spikes

## 3.1 Principle

Following conversion, pelagos-py separates each b<sub>bp</sub> record into a smooth baseline and an isolated-spike component using a rolling-window filter. The implementation (`glidertools.cleaning.despike`) follows Briggs et al. (2011), and offers two filter methods:

- **`median`** (default) — a rolling median is applied directly to the record; this forms the baseline, and the spike component is the residual (measurement − baseline), which can be positive or negative.
- **`minmax`** — a rolling minimum is applied first, followed by a rolling maximum of that result, forming a baseline that tracks the lower envelope of the signal; spikes under this method are strictly positive or zero, consistent with backscatter spikes being interpreted as discrete particles adding to a background baseline, but instrument noise will add a positive bias to the spike signal.

$$
\text{baseline}(t) = \operatorname{filter}\left[b_{b}(t);\ \text{window\_size}\right],
\qquad
\text{spikes}(t) = b_{b}(t) - \text{baseline}(t).
$$

Both the baseline and spike components are retained as independent output variables; neither is discarded.

## 3.2 Default configuration

```yaml
- name: "Isolate BBP Spikes"
  parameters:
    apply_to: BBP700
    window_size: 50
    method: median
  diagnostics: true
```

> **Note — known issue: default `window_size` of 50 is likely too large, and its stated justification does not match the code**
>
> This documentation has stated that "the default 50-point rolling window is the same window size used by the CHLA fluorescence despiking step" — but no such CHLA despiking step exists in pelagos-py (checked both `main` and `DEMO`): the only `filter_window_size` in the codebase belongs to an unrelated PSAL smoothing filter (default 21, in `salinity.py`), not to any chlorophyll despiking routine. The stated rationale for 50 does not currently anchor to anything in the code.
>
> Briggs et al. (2011) — the method this despiking step implements (`glidertools.cleaning.despike`, called with `apply_to: BBP700`) — used a **7-point** rolling filter, applied over valid (non-NaN) b<sub>bp</sub> points. `window_size` in pelagos-py has the same "N valid points" semantics (QC-flagged samples are masked to `NaN` and excluded before the rolling window is applied — see Section 3.3), so `window_size: 50` is directly comparable to that 7-point choice, not confounded by a different windowing convention. Even allowing for much higher-resolution modern glider sampling than the original Briggs et al. (2011) float data, 50 is roughly 3–7&times; larger than would typically be expected (a window of order 11–15 points).
>
> This is a tuning/default question rather than a correctness bug — the "right" window size depends on sensor sampling rate and platform vertical speed, and reference deployment configs (e.g. `example_config_nelson.yaml`) currently set `window_size: 50` explicitly, so it is genuinely in use, not just an unexercised default. See [pelagos-py issue #163](https://github.com/NOC-OBG-Autonomy/pelagos-py/issues/163) for discussion before assuming 50 is appropriate for a new deployment; consider comparing against a smaller window (e.g. 11–15 points) for high-resolution data.

## 3.3 QC filtering and missing-value propagation

Unlike BBP from Beta (Section 2.3), Isolate BBP Spikes **does** exclude flagged samples from the despike calculation by default: samples carrying QC flags `3`, `4` or `9` on `apply_to` are masked to `NaN` before the rolling filter is applied, so a flagged (e.g. biofouled) stretch cannot drag down the rolling baseline computed for an isolated good sample nearby. Samples excluded this way receive **no** baseline or spike value at all (both `NaN`), and are flagged missing (`9`) in the corresponding `_BASELINE_QC`/`_SPIKES_QC` outputs — this differs from steps such as `deep_correction`, where excluded samples are still corrected even though they do not inform the correction. A gap left this way in the baseline is not filled by this step; an **Interpolate Data** step run afterward on `<var>_BASELINE` produces a gap-free baseline for downstream use (e.g. by the chlorophyll NPQ methods that require a continuous backscatter reference).

## 3.4 Figure placeholder

```{figure} ../_static/bbp/bbp_baseline_spikes_timeseries.png
:alt: Placeholder for the bbp baseline and spikes diagnostic time series.
:width: 90%

**Placeholder.** Raw b<sub>bp</sub> time series with the despiked baseline overlaid (top panel), and the isolated spike component (bottom panel).
```

---

# 4. Use in chlorophyll NPQ correction

b<sub>bp</sub> does not receive a dedicated correction analogous to chlorophyll's deep-offset or NPQ corrections. Instead, the despiked baseline produced in Section 3 is consumed as the optical reference by several of the chlorophyll NPQ correction methods (Sackmann et al., 2008; Swart et al., 2015; Hemsley et al., 2015; Thomalla et al., 2018; Mitchell et al., 2024; Xing et al., 2018; Terrats et al., 2020), under the assumption that non-photochemical quenching alters chlorophyll fluorescence yield without substantially affecting particulate backscattering.

When a backscatter-based NPQ method is configured without an explicit `bbp_var`, pelagos-py searches for a usable backscatter variable in the following order:

1. `BBP700_BASELINE`;
2. `BBP700`;
3. `BBP532_BASELINE`; and
4. `BBP532`.

A despiked baseline product (`_BASELINE`) is preferred over the raw/undespiked variable because isolated spikes or unrealistically small backscatter values can produce artificially large fluorescence-to-backscatter ratios and consequently overestimate the reconstructed chlorophyll fluorescence.

Full descriptions of the individual NPQ correction methods, their assumptions, and the shared implementation rules governing how backscatter is used within them, are documented in the **Chlorophyll-*a* Processing** documentation (`pelagos-py_CHLA.md`), Section 5.

---

# 5. Outputs

The backscatter processing workflow produces or updates the following variables:

| Variable | Description |
|---|---|
| `BETA_BACKSCATTERING700` (or platform-specific equivalent) | Raw volume scattering function in m<sup>-1</sup> sr<sup>-1</sup>, measured at the sensor's effective angle and wavelength. This is the native measurement used as the input to the BBP from Beta conversion. |
| `<apply_to>_QC` | Quality-control flags associated with the raw signal, including the results of the initial range and stuck-value tests. |
| `BBP700` (or configured `output_as`) | Total (particulate + seawater) backscattering coefficient in m<sup>-1</sup>, derived from the raw signal using the Zhang et al. (2009) seawater-scattering model and the Sullivan et al. (2013) chi-factor scaling. See the note in Section 2.1 regarding the seawater contribution retained in this value. |
| `BBP700_QC` | Quality-control flags associated with `BBP700`. Inherited from the input's QC when `output_as` differs from `apply_to`; left unmodified (with a logged warning) when the conversion is applied in place. |
| `BBP700_BASELINE` | Despiked baseline backscatter signal, produced by the Isolate BBP Spikes step. This is the variable most commonly consumed by the backscatter-based chlorophyll NPQ correction methods. |
| `BBP700_SPIKES` | Backscatter spike component separated from the baseline, retained independently as it represents discrete particle/aggregate signal rather than an artefact. |
| `BBP700_BASELINE_QC`, `BBP700_SPIKES_QC` | Quality-control flags associated with the baseline and spike products, inherited from `BBP700_QC` with any resulting `NaN` automatically flagged missing (`9`). |

---

# 6. Recommended reporting

A scientific methods section describing backscatter processing with pelagos-py should report:

- the raw signal variable and its initial range- and stuck-value-test thresholds;

- that no spike test was applied to the raw signal, and why (spikes are treated as real particulate signal);

- the effective scattering angle (`theta`) and chi factor (`xfactor`) used for the beta-to-b<sub>bp</sub> conversion, and the sensor model/geometry they correspond to;

- that the reported b<sub>bp</sub> value is the total (particulate + seawater) backscattering coefficient, not the particulate component alone;

- the despiking window size and method (median or minmax) used to separate the baseline and spike components; and

- which backscatter variable (raw, or despiked baseline) was used as the optical reference for any chlorophyll NPQ correction applied.

An example methods description is given below.

> Particulate backscattering was processed using pelagos-py. The raw volume scattering function (`BETA_BACKSCATTERING700`, m<sup>-1</sup> sr<sup>-1</sup>) was screened using a range test, with values outside 0 to 0.01 m<sup>-1</sup> sr<sup>-1</sup> flagged probably bad and values outside -1.0 &times; 10<sup>-4</sup> to 0.05 m<sup>-1</sup> sr<sup>-1</sup> flagged bad, together with a stuck-value test flagging three or more identical consecutive readings as probably bad. No spike test was applied to the raw signal, as backscatter spikes were considered to represent genuine particulate/aggregate signal rather than sensor artefacts.
>
> The raw signal was converted to the total (particulate + seawater) backscattering coefficient at 700 nm (`BBP700`, m<sup>-1</sup>) following the pure-seawater scattering model of Zhang et al. (2009) and the chi-factor scaling of Sullivan et al. (2013), using an effective scattering angle of 124&deg; and a chi factor of 1.076, appropriate for a three-channel WET Labs ECO-type instrument. The converted signal was subsequently despiked using a centred rolling-median filter of 50 observations following Briggs et al. (2011), producing a despiked baseline (`BBP700_BASELINE`) and an isolated spike component (`BBP700_SPIKES`). The despiked baseline was used as the optical reference for chlorophyll non-photochemical quenching correction.

---

# References

- Briggs, N., Perry, M. J., Cetini&#263;, I., Lee, C., D'Asaro, E., Gray, A. M., & Rehm, E. (2011). High-resolution observations of aggregate flux during a sub-polar North Atlantic spring bloom. *Deep-Sea Research Part I*, **58**(10), 1031–1039. https://doi.org/10.1016/j.dsr.2011.07.007

- Sullivan, J.M., Twardowski, M.S., Zaneveld, J.R.V., & Moore, C.C. (2013). Measuring optical backscattering in water. Chapter 6 in *Light Scattering Reviews 7: Radiative Transfer and Optical Properties of Atmosphere and Underlying Surface*, pp 189–224.

- Zhang, X., Hu, L., & He, M.-X. (2009). Scattering by pure seawater: Effect of salinity. *Optics Express*, **17**(7), 5698–5710. https://doi.org/10.1364/OE.17.005698

- OOI (2012). *Data Product Specification for Optical Backscatter (Red Wavelengths)*. Document Control Number 1341-00540 (version 1-05).

Full references for the chlorophyll NPQ correction methods that consume b<sub>bp</sub> as an optical reference (Sackmann et al., 2008; Swart et al., 2015; Hemsley et al., 2015; Thomalla et al., 2018; Mitchell et al., 2024; Xing et al., 2018; Terrats et al., 2020) are listed in the **Chlorophyll-*a* Processing** documentation.
