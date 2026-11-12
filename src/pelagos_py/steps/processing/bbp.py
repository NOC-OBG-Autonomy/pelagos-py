# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline steps for deriving particulate backscatter (BBP) from beta and isolating BBP spikes."""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.qc_handling import QCHandlingMixin, prefer_adjusted, raw_name
from pelagos_py.utils.processing_utils import remove_outliers

#### Custom imports ####
import re
import numpy as np
import pandas as pd
from pelagos_py.utils import fig_spec


@register_step
class BBPFromBeta(BaseStep, QCHandlingMixin):
    """
    Converts measured beta backscatter to the particulate backscattering coefficient
    (BBP), using temperature and salinity for the seawater term.

    See :doc:`/user_guide/Pelagos-py_BBP`.

    Examples
    --------
    .. code-block:: yaml

        - name: "BBP from Beta"
          parameters:
            apply_to: "BETA_BACKSCATTERING700"
            output_as: "BBP700"
            theta: 124
            xfactor: 1.076
          diagnostics: false
    """

    step_name = "BBP from Beta"
    required_variables = ["TIME", "DEPTH", "TEMP", "PSAL"]
    provided_variables = []
    variable_parameters = ["apply_to", "output_as"]
    # apply_to is resolved at run time with fallbacks (see _resolve_beta_var)
    variable_parameters_optional = ("apply_to",)
    uses_data_subset = True

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "BETA_BACKSCATTERING700",
            "description": (
                "Name of the beta backscatter variable to convert. If not found, "
                "falls back to any other BETA_BACKSCATTERING<wavelength> variable "
                "(closest to 700nm if several are present)."
            ),
        },
        "output_as": {
            "type": str,
            "default": "BBP700",
            "description": "Name for the output variable added to the dataset.",
        },
        "theta": {
            "type": float,
            "default": 124,
            "description": "Effective optical backscatter scattering angle (degrees).",
        },
        "xfactor": {
            "type": float,
            "default": 1.076,
            "description": "Chi factor scaling particulate scattering to total backscatter.",
        },
    }

    def run(self):
        self.filter_qc()

        self.beta_var = self._resolve_beta_var()

        # The CTD samples on other rows: its TEMP/PSAL are interpolated onto the beta rows
        # for this calculation only. Where that fails, BBP is not derived (flagged 9).
        temp, _ = self.values_at_rows_of("TEMP", self.beta_var)
        psal, _ = self.values_at_rows_of("PSAL", self.beta_var)

        # Particulate only, as BGC-Argo (Schmechtig et al.): the seawater term is subtracted.
        beta_sw = seawater_beta(temp, psal, self.theta, 700)
        bbp_corrected = 2 * np.pi * self.xfactor * (self.data[self.beta_var] - beta_sw)

        # Stitch back into the data
        self.data[self.output_as] = bbp_corrected
        self.data[self.output_as].attrs["units"] = "m-1"
        self.data[self.output_as].attrs["long_name"] = "Particulate backscatter"
        self.data[self.output_as].attrs["standard_name"] = self.output_as

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        if self.beta_var != self.output_as:
            self.generate_qc({f"{self.output_as}_QC": [f"{self.beta_var}_QC"]})
        else:
            self.log_warn(
                f"'apply_to' and 'output_as' are the same. This will cause {self.beta_var} to be overwritten."
            )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"].update(self.data)
        return self.context

    def _resolve_beta_var(self):
        # apply_to, else the BETA_BACKSCATTERING<wl> closest to 700 nm
        full_vars = self.context["data"].data_vars
        beta_var = prefer_adjusted(self.apply_to, full_vars)
        if beta_var in full_vars:
            return self._pull_into_subset(beta_var)

        fallback = self._closest_wavelength_var(full_vars, "BETA_BACKSCATTERING")
        if fallback:
            self.log_warn(f"'{self.apply_to}' not found; using '{fallback}' instead.")
            if not fallback.endswith("700"):
                self.log_warn(
                    "Seawater scattering is still removed at 700 nm, so BBP from "
                    f"'{fallback}' will be slightly off."
                )
            return self._pull_into_subset(fallback)

        # Raw beta mislabelled as BBP<wl> is handled upstream by Prepare OG1 (bbp700_is_beta).
        raise ValueError(
            f"'{self.apply_to}' not found, and no BETA_BACKSCATTERING* variable is "
            "present to fall back to."
        )

    def _pull_into_subset(self, name):
        # QCHandlingMixin only subsets the configured apply_to, so a fallback may be missing
        if name not in self.data:
            self.data[name] = self.context["data"][name]
            if f"{name}_QC" in self.context["data"]:
                self.data[f"{name}_QC"] = self.context["data"][f"{name}_QC"]
        return name

    @staticmethod
    def _closest_wavelength_var(names, prefix):
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        candidates = [
            (abs(int(match.group(1)) - 700), name)
            for name in names
            if (match := pattern.match(name))
        ]
        return min(candidates)[1] if candidates else None

    diagnostic_figures = {"boxplot": ("Beta vs BBP value distributions", True)}

    def draw_figure(self, name):
        beta_clean = remove_outliers(self.data[self.beta_var])
        bbp_clean = remove_outliers(self.data[self.output_as])
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        # boxplot is not WebGL-safe (PNG-only in the dashboard) but is the point of this figure
        ax.boxplot(
            [beta_clean, bbp_clean],
            patch_artist=True,
            boxprops=dict(facecolor=fig_spec.CATEGORY[1], alpha=0.6),
        )
        ax.set_xticks(
            [1, 2],
            [
                fig_spec.axis_label(
                    self.beta_var, self.data[self.beta_var].attrs.get("units")
                ),
                fig_spec.axis_label(self.output_as, "m-1"),
            ],
        )
        fig_spec.style_axes(ax, ylabel="value")
        fig_spec.finish(fig, f"{self.beta_var} vs {self.output_as}")
        return fig


def seawater_beta(temp, psal, theta, wavelength, delta=0.039):
    """Volume scattering function of pure seawater (m-1 sr-1) at angle ``theta`` (degrees).

    Zhang, Hu & He (2009), Optics Express 17(7), doi:10.1364/OE.17.005698; the same
    model as glidertools' ``flo_zhang_scatter_coeffs``.
    """
    t, s = np.asarray(temp, dtype=float), np.asarray(psal, dtype=float)
    avogadro = 6.0221417930e23
    boltzmann = 1.3806503e-23
    water_molar_mass = 0.018  # kg/mol

    # refractive index of air (Ciddor 1996) and seawater (Quan & Fry 1995)
    inv_um2 = 1 / (wavelength / 1e3) ** 2
    n_air = 1 + (5792105 / (238.0185 - inv_um2) + 167917 / (57.362 - inv_um2)) / 1e8
    n_sw = n_air * (
        1.31405
        + (1.779e-4 - 1.05e-6 * t + 1.6e-8 * t**2) * s
        - 2.02e-6 * t**2
        + (15.868 + 0.01155 * s - 0.00423 * t) / wavelength
        - 4382 / wavelength**2
        + 1.1455e6 / wavelength**3
    )
    dn_ds = n_air * (1.779e-4 - 1.05e-6 * t + 1.6e-8 * t**2 + 0.01155 / wavelength)

    # isothermal compressibility (Pa-1) from the secant bulk modulus (Millero et al. 1980)
    bulk_water = (
        19652.21
        + 148.4206 * t
        - 2.327105 * t**2
        + 1.360477e-2 * t**3
        - 5.155288e-5 * t**4
    )
    bulk = (
        bulk_water
        + (54.6746 - 0.603459 * t + 1.09987e-2 * t**2 - 6.167e-5 * t**3) * s
        + (7.944e-2 + 1.6483e-2 * t - 5.3009e-4 * t**2) * s**1.5
    )
    compressibility = 1e-5 / bulk

    # surface density (UNESCO 1981)
    rho_water = (
        999.842594
        + 6.793952e-2 * t
        - 9.09529e-3 * t**2
        + 1.001685e-4 * t**3
        - 1.120083e-6 * t**4
        + 6.536332e-9 * t**5
    )
    rho = (
        rho_water
        + (
            8.24493e-1
            - 4.0899e-3 * t
            + 7.6438e-5 * t**2
            - 8.2467e-7 * t**3
            + 5.3875e-9 * t**4
        )
        * s
        + (-5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t**2) * s**1.5
        + 4.8314e-4 * s**2
    )

    # d ln(water activity) / dS (Millero & Leung 1976)
    a0 = -5.58651e-4 + 2.40452e-7 * t - 3.12165e-9 * t**2 + 2.40808e-11 * t**3
    a1 = 1.79613e-5 - 9.9422e-8 * t + 2.08919e-9 * t**2 - 1.39872e-11 * t**3
    a2 = -2.31065e-6 - 1.37674e-9 * t - 1.93316e-11 * t**2
    dlnaw_ds = a0 + 1.5 * a1 * s**0.5 + 2 * a2 * s

    # scattering at 90 degrees from density and concentration fluctuations
    density_term = (n_sw**2 - 1) * (
        1 + 2 / 3 * (n_sw**2 + 2) * (n_sw / 3 - 1 / (3 * n_sw)) ** 2
    )
    depolarisation = (6 + 6 * delta) / (6 - 7 * delta)
    lambda4 = (wavelength * 1e-9) ** -4
    thermal = boltzmann * (t + 273.15) * compressibility
    beta90_density = np.pi**2 / 2 * lambda4 * thermal * density_term**2 * depolarisation
    concentration = s * water_molar_mass * dn_ds**2 / rho / -dlnaw_ds / avogadro
    beta90_concentration = (
        2 * np.pi**2 * lambda4 * n_sw**2 * concentration * depolarisation
    )
    beta90 = beta90_density + beta90_concentration

    angle = np.radians(theta)
    return beta90 * (1 + (1 - delta) / (1 + delta) * np.cos(angle) ** 2)


def _rolling(arr, n, name):
    # same as glidertools.cleaning.rolling_window, which uses expanding windows at both ends
    i0 = n // 2
    interior = getattr(pd.Series(arr).rolling(n), name)().to_numpy()[
        n - 1 : arr.size - 1
    ]
    head = getattr(pd.Series(arr[:i0]).expanding(), name)().to_numpy()
    tail = getattr(pd.Series(arr[::-1][: n - i0]).expanding(), name)().to_numpy()
    return np.concatenate([head, interior, tail])


def despike(var, window_size, spike_method="median"):
    """Briggs et al. (2011) despike, as glidertools.cleaning.despike. Returns (baseline, spikes).

    Gives the same result without building a (window x N) matrix (~1.6 GB on 2M samples).
    """
    arr = np.asarray(var, dtype=float)
    mask = ~np.isnan(arr)
    if spike_method.startswith("min"):
        base = _rolling(_rolling(arr[mask], window_size, "min"), window_size, "max")
    else:
        base = _rolling(arr[mask], window_size, "median")
    baseline = np.full(arr.shape, np.nan)
    baseline[mask] = base
    return baseline, arr - baseline


@register_step
class IsolateBBPSpikes(BaseStep, QCHandlingMixin):
    """
    Splits a backscatter record into a smooth baseline and isolated spikes (Briggs et
    al. 2011), written as ``<var>_BASELINE`` and ``<var>_SPIKES``.

    See :doc:`/user_guide/Pelagos-py_BBP`.

    Examples
    --------
    .. code-block:: yaml

        - name: "Isolate BBP Spikes"
          parameters:
            apply_to: "BBP700"
            window_size: 7
            method: "median"
          diagnostics: false
    """

    step_name = "Isolate BBP Spikes"
    required_variables = ["TIME"]
    provided_variables = []
    variable_parameters = ["apply_to"]
    uses_data_subset = True

    parameter_schema = {
        "apply_to": {
            "type": str,
            "default": "BBP700",
            "description": "Name of the variable to filter.",
        },
        "window_size": {
            "type": int,
            "default": 7,
            "description": "Median/minmax filter window size in samples (7-point filter of Briggs et al. 2011).",
        },
        "method": {
            "type": str,
            "default": "median",
            "description": "Filter method used to determine the baseline.",
        },
    }

    @classmethod
    def output_variables(cls, parameters):
        apply_to = parameters.get(
            "apply_to", cls.parameter_schema["apply_to"]["default"]
        )
        return [f"{raw_name(apply_to)}_BASELINE", f"{raw_name(apply_to)}_SPIKES"]

    def run(self):
        self.filter_qc()
        self.source = self.resolve(self.apply_to)
        base = raw_name(self.apply_to)

        # Flagged samples are left out of the despike so they cannot drag their
        # neighbours' rolling baseline; they get no baseline and are flagged missing (9)
        # below. Add an Interpolate Data step first for a gap-free baseline.
        usable = self.data[self.source].where(self.calculation_mask([self.source]))

        self.baseline, self.spikes = despike(
            usable, self.window_size, spike_method=self.method
        )

        attrs = dict(usable.attrs)
        attrs["history"] = attrs.get("history", "") + (
            f"Isolate BBP Spikes: despike(window_size={self.window_size}, "
            f"spike_method='{self.method}'), Briggs et al. (2011) as in glidertools.cleaning.despike;\n"
        )
        self.data[f"{base}_BASELINE"] = (usable.dims, self.baseline, attrs)
        self.data[f"{base}_SPIKES"] = (usable.dims, self.spikes, dict(attrs))

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        self.generate_qc(
            {
                f"{base}_BASELINE_QC": [f"{self.source}_QC"],
                f"{base}_SPIKES_QC": [f"{self.source}_QC"],
            }
        )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"].update(self.data)
        return self.context

    diagnostic_figures = {
        "baseline_spikes": ("Raw, baseline and isolated spikes", True)
    }

    def draw_figure(self, name):
        raw = self.data[self.source]
        time = self.data["TIME"]

        fig, axes = fig_spec.new_fig(nrows=2, sharex=True, height_ratios=(2, 1))
        ax1, ax2 = axes[0][0], axes[1][0]

        # Panel 1: raw and baseline time series.
        ax1.plot(
            time[~np.isnan(raw)],
            raw[~np.isnan(raw)],
            ls="--",
            color=fig_spec.FLAGGED,
            label="Raw",
        )
        ax1.plot(
            time[~np.isnan(self.baseline)],
            self.baseline[~np.isnan(self.baseline)],
            color=fig_spec.CATEGORY[1],
            alpha=fig_spec.ALPHA,
            label="Baseline",
        )

        # Panel 2: isolated spike points.
        fig_spec.points(
            ax2,
            time[~np.isnan(self.spikes)],
            self.spikes[~np.isnan(self.spikes)],
            color=fig_spec.CATEGORY[2],
            label="Spikes",
        )

        ylabel = fig_spec.axis_label(
            self.source, self.data[self.source].attrs.get("units")
        )
        for ax in (ax1, ax2):
            fig_spec.date_axis(ax, which="x", index=time.values)
            fig_spec.legend(ax)
        fig_spec.style_axes(ax1, ylabel=ylabel)
        fig_spec.style_axes(ax2, xlabel="Time", ylabel=ylabel)

        fig_spec.finish(fig, suptitle=f"{self.source}: Baseline Timeseries & Spikes")
        return fig
