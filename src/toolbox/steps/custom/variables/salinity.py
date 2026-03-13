# This file is part of the NOC Autonomy Toolbox.
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

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
from tqdm import tqdm
import xarray as xr
import numpy as np
import gsw


def running_average_nan(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Estimate running average mean
    """

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd for symmetry.")

    pad_size = window_size // 2  # Symmetric padding
    padded = np.pad(arr, pad_size, mode="reflect")  # Edge handling

    kernel = np.ones(window_size)
    # Compute weighted sums while ignoring NaNs
    sum_vals = np.convolve(np.nan_to_num(padded), kernel, mode="valid")
    count_vals = np.convolve(~np.isnan(padded), kernel, mode="valid")

    # Compute the moving average, handling NaNs properly
    avg = np.divide(sum_vals, count_vals, where=(count_vals != 0))
    avg[count_vals == 0] = np.nan  # Set to NaN where all values were NaN

    return avg


def compute_optimal_lag(profile_data, filter_window_size, time_col):
    """
    Calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
    Mimimize the standard deviation of the difference between a lagged CNDC and a high-pass filtered CNDC.
    The optimal lag is returned. The lag is chosen from -2 to 2s every 0.1s.
    This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.
    This correction is described in Woo (2019) but the mimimization is done between salinity and high-pass filtered salinity (as done by RBR, https://bitbucket.org/rbr/pyrsktools/src/master/pyrsktools) instead of comparing downcast vs upcast.

    Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/ 10.26198/5c997b5fdc9bd).


    Parameters
    ----------
    self.tsr: xarray.Dataset with raw CTD dataset for one single profile, which should contain:
        - TIME_CTD, sci_ctd41cp_timestamp, [numpy.datetime64]
        - PRES: pressure [dbar]
        - CNDC: conductivity [S/m]
        - TEMP: in-situ temperature [de C]

    windowLength: Window length over which the high-pass filter of conductivity is applied, 21 by default.

    Returns
    -------
    self.tsr: with lags.

    """

    # remove any rows where conductivity is bad (nan)
    profile_data = profile_data[
        [time_col,
         "CNDC",
         "PRES",
         "TEMP"]
    ].dropna(dim="N_MEASUREMENTS", subset=["CNDC"])

    # Find the elapsed time in seconds from the start of the profile
    t0 = profile_data[time_col].values[0]
    profile_data["ELAPSED_TIME[s]"] = (profile_data[time_col] - t0).dt.total_seconds()

    # Creates a callable function that predicts what CNDC would be at any given time
    conductivity_from_time = interpolate.interp1d(
        profile_data["ELAPSED_TIME[s]"].values,
        profile_data["CNDC"].values,
        bounds_error=False
    )

    # Specify the range time lags that the optimum will be found from. Column indexes are: (lag value, lag score)
    time_lags = np.array(
        [np.linspace(-2, 2, 41),
         np.full(41, np.nan)]
    ).T

    # For each lag find its score and add it to the time_lags array
    for i, lag in enumerate(time_lags[:, 0].copy()):
        # Apply the time shift
        time_shifted_conductivity = conductivity_from_time(
            profile_data["ELAPSED_TIME[s]"] + lag
        )
        # Derive salinity with the time shifted CNDC (spiking will be minimized when CNDC and TEMP are aligned)
        PSAL = gsw.conversions.SP_from_C(
            time_shifted_conductivity,
            profile_data["TEMP"],
            profile_data["PRES"]
        )

        # Smooth the salinity profile (to remove spiking)
        PSAL_Smooth = running_average_nan(PSAL, filter_window_size)

        # Subtracting the raw and smoothed salinity gives an idication of "spikiness".
        PSAL_Diff = PSAL - PSAL_Smooth

        # More spiky data will have higher standard deviation - which is used to score the effectiveness of the applied lag
        time_lags[i, 1] = np.nanstd(PSAL_Diff)

    # return the time lag which has the lowerst score (standard deviation)
    best_score_index = np.argmin(time_lags[:, 1])
    return time_lags[best_score_index, 0]


@register_step
class AdjustSalinity(BaseStep, QCHandlingMixin):
    step_name = "Salinity Adjustment"

    def run(self):
        """
        Apply the thermal-lag correction for Salinity presented in Morrison et al 1994.
        The temperature is estimated inside the conductivity cell to estimate Salinity.
        This is based on eq.5 of Morrison et al. (1994), which doesn't require to know the sensitivity of temperature to conductivity (eq.2 of Morrison et al. 1994).
        No attempt is done yet to minimize the coefficients alpha/tau in T/S space, as in Morrison et al. (1994) or Garau et al. (2011).
        The fixed coefficients (alpha and tau) presented in Morrison et al. (1994) are used.
        These coefficients should be valid for pumped SeaBird CTsail as described in Woo (2019) by using their flow rate in the conductivity cell.
        This function should further be adapted to unpumped CTD by taking into account the glider velocity through the water based on the pitch angle or a hydrodynamic flight model.

        Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/10.26198/5c997b5fdc9bd).

        Config Example
        --------------
          - name: "ADJ: Salinity"
            parameters:
              filter_window_size: 21
              plot_profiles_in_range: [100, 150]
            diagnostics: false

        Parameters
        -----------

        self.tsr: xarray.Dataset with raw CTD dataset, which should contain:
            - time, sci_m_present_time, [numpy.datetime64]
            - PRES: pressure [dbar]
            - CNDC: conductivity [S/c]
            - TEMP: in-situ temperature [deg C]
            - LON: longitude
            - LAT: latitude

        Returns
        -------
            Nil - serves on self in-place
                MUST APPLY self.data to self.context["data"] to save the changes

        """


        self.log(f"Running adjustment...")
        # TODO: TIME_CTD checking

        # Required for plotting later
        self.data_copy = self.data.copy(deep=True)

        # Check if TIME_CTD exists
        self.time_col = "TIME_CTD"
        if self.time_col not in self.data:
            self.log("TIME_CTD cound not be found. Defaulting to TIME instead.")
            self.time_col = "TIME"

        # Filter user-specified flags
        self.filter_qc()

        # Correct conductivity-temperature response time misalignment (C-T Lag)
        self.correct_ct_lag()

        # Correct thermal mass error
        self.correct_thermal_lag()

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        self.display_CTLag()
        self.display_adj_profiles()

    def correct_ct_lag(self):
        """
        For the full deployment, calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
        If more than 300 profiles are present, the optimal lag is estimated every 10 profiles.
        Display the optimal conductivity time lag calculated for each profile, estimate the median of this lag, and apply this median lag to corrected variables (CNDC_ADJ/PSAL_ADJ).
        This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.


        Parameters
        ----------
        self.tsr: xarray.Dataset with raw CTD dataset, which should contain:
            - PROFILE_NUMBER
            - TIME_CTD, sci_ctd41cp_timestamp, [numpy.datetime64]
            - PRES: pressure [dbar]
            - CNDC: conductivity [mS/cm]
            - TEMP: in-situ temperature [de C]

        windowLength: Window length over which the high-pass filter of conductivity is applied, 21 by default.

        Returns
        -------
        self.tsr: with tau and prof_i.

        """

        # Estimate the CT lag every profile or 10 profiles for more than 300 profiles.
        # Note that profile_numbers is not a list of consecutive integers as some profiles may have been filtered out.
        profile_numbers = np.unique(self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values)
        if len(profile_numbers) > 300:
            profile_numbers = profile_numbers[::10]

        # Making a place to store intermediate products. The two column dimentions are (profile number, time lag)
        self.per_profile_optimal_lag = np.full((len(profile_numbers), 2), np.nan)

        # TODO: The following could be optimized using xarray groupby() applying a user defined CTLag function
        # Loop through all good profiles and store the optimal C-T lag for each.
        for i, profile_number in enumerate(tqdm(profile_numbers, colour="green", desc='\033[97mProgress\033[0m', unit="prof")):
            profile = self.data.where((self.data["PROFILE_NUMBER"] == profile_number), drop=True)
            if len(profile[self.time_col]) > 3 * self.filter_window_size:
                optimal_lag = compute_optimal_lag(profile, self.filter_window_size, self.time_col)
                self.per_profile_optimal_lag[i, :] = [profile_number, optimal_lag]

        # Find median optimal time lag across all profiles
        median_lag = np.nanmedian(self.per_profile_optimal_lag[:, 1])
        
        # Get a nanless subset of CNDC data
        nan_mask = self.data["CNDC"].isnull()
        data_subset = self.data[[self.time_col, "CNDC"]].where(~nan_mask, drop=True)

        # Find the elapsed time in seconds
        t0 = data_subset[self.time_col].values[0]
        data_subset["ELAPSED_TIME[s]"] = (data_subset[self.time_col] - t0).dt.total_seconds()
        
        # Resample the data using a shifted time
        CNDC_from_TIME = interpolate.interp1d(
            data_subset["ELAPSED_TIME[s]"], 
            data_subset["CNDC"], 
            bounds_error=False
        )
        data_subset["CNDC"][:] = CNDC_from_TIME(data_subset["ELAPSED_TIME[s]"] + median_lag)
        
        # Reinsert the time-shifted data back into self.data
        self.data["CNDC"][~nan_mask] = data_subset["CNDC"]

    def correct_thermal_lag(self):

        nan_mask = self.data["TEMP"].isnull()
        data_subset = self.data[[self.time_col, "TEMP", "PRES"]].where(~nan_mask, drop=True)

        # Find the elapsed time in seconds
        t0 = data_subset[self.time_col].values[0]
        data_subset["ELAPSED_TIME[s]"] = (data_subset[self.time_col] - t0).dt.total_seconds()

        # TODO: Convert to xarray interpolation as interp1d doesn't get updated any more
        # Define a function that can estimate TEMP at any time point
        TEMP_from_TIME = interpolate.interp1d(
            data_subset["ELAPSED_TIME[s]"], 
            data_subset["TEMP"], 
            bounds_error=False
        )
        
        # Resample the data onto a 1Hz sample rate timeseries
        TIME_1Hz_sampling = np.arange(0, data_subset["ELAPSED_TIME[s]"][-1], 1)
        TEMP_1Hz_sampling = TEMP_from_TIME(TIME_1Hz_sampling)
        n_resamples = len(TEMP_1Hz_sampling)

        # Set up the recursive filter defined in "CTD dynamic performance and corrections through gradients"
        # Tau and alpha are the fixed coefficients of Morison94 for unpumped cell.
        # alpha: initial amplitude of the temperature error for a unit step change in ambient temperature [without unit].
        alpha_offset = 0.0135
        alpha_slope = 0.0264
        # tau = beta^-1: time constant of the error, the e-folding time of the temperature error [s].
        tau_offset = 7.1499
        tau_slope = 2.7858
        # flow_rate: The flow rate in the conductivity cell from Woo (2019).
        flow_rate = 0.4867

        tau = tau_offset + tau_slope / np.sqrt(flow_rate)
        alpha = alpha_offset + alpha_slope / flow_rate

        # Set the filter coefficients
        nyquist_frequency = 1/2  # Nyquist frequency for 1 Hz sampling (= sample frequency / 2)
        a = 4 * nyquist_frequency * alpha * tau / (1 + 4 * nyquist_frequency * tau)
        b = 1 - (2 * a / alpha)

        # Apply the filter
        TEMP_correction = np.full(n_resamples, 0.0)
        for i in range(1, n_resamples):
            TEMP_correction[i] = -b * TEMP_correction[i - 1] + a * (TEMP_1Hz_sampling[i] - TEMP_1Hz_sampling[i - 1])
        corrected_TEMP_1Hz_sampling = TEMP_1Hz_sampling - TEMP_correction

        # Resample the TEMP back onto the original time sampling
        corrected_TEMP_from_TIME = interpolate.interp1d(
            TIME_1Hz_sampling, 
            corrected_TEMP_1Hz_sampling, 
            bounds_error=False
        )
        data_subset["TEMP"][:] = corrected_TEMP_from_TIME(data_subset["ELAPSED_TIME[s]"])

        # Reinsert the corrected data back into self.data
        self.data["TEMP"][~nan_mask] = data_subset["TEMP"]

    def display_CTLag(self):
        # Display optimal CTlag for each profile
        mpl.use("tkagg")
        prof_min, prof_max = np.nanmin(self.per_profile_optimal_lag[:, 0]), np.nanmax(self.per_profile_optimal_lag[:, 0])
        tau_median = np.nanmedian(self.per_profile_optimal_lag[:, 1])

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            [prof_min, prof_max],
            [tau_median, tau_median],
            c="indianred",
            linestyle="--",
            linewidth=2,
            label=f"Median CTlag: {tau_median:.2f}s",
        )
        ax.plot([prof_min, prof_max], [0, 0], "k")
        ax.scatter(self.per_profile_optimal_lag[:, 0], self.per_profile_optimal_lag[:, 1], c="k")
        ax.legend(prop={"weight": "bold"}, labelcolor="indianred")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axis([prof_min, prof_max, -1, 1])
        ax.set_ylabel(
            "CTlag [s]\n < 0: delay Cond by CTlag\n > 0: advance Cond by CTlag",
            fontweight="bold",
        )
        ax.set_xlabel("Profile Index", fontweight="bold")

        fig.tight_layout()
        plt.show(block=True)

    def display_adj_profiles(self):
        """
        Display profiles for ~20 mid profiles of:
            (1) PSAL: raw salinity
            (2) PSAL_ADJ: salinity corrected from CTlag
            (3) PSAL_ADJ: salinity with the thermal lag correction
            (4) difference between raw and ADJ (CTlag + thermal lag correction) salinity and temperature

        """
        self.log("Displaying salinity profiles.")
        mpl.use("tkagg")

        # Get corrected and uncorrected profiles
        uncorrected_profiles = self.data_copy.where(
            (self.data["PROFILE_NUMBER"] > self.plot_profiles_in_range[0]) &
            (self.data["PROFILE_NUMBER"] < self.plot_profiles_in_range[1]),
            drop=True
        )
        corrected_profiles = self.data.where(
            (self.data["PROFILE_NUMBER"] > self.plot_profiles_in_range[0]) &
            (self.data["PROFILE_NUMBER"] < self.plot_profiles_in_range[1]),
            drop=True
        )

        fig, axs = plt.subplots(ncols=2, figsize=(8, 8), sharex=True, sharey=True)

        for ax, data, title in zip(axs, [uncorrected_profiles, corrected_profiles], ["Uncorrected", "Corrected"]):
            for direction, col, label in zip([-1, 1], ["r", "b"], ["Descending", "Ascending"]):
                plot_data = data[["DEPTH", "CNDC", "TEMP", "PRES", "PROFILE_DIRECTION"]].where(
                    data["PROFILE_DIRECTION"] == direction
                )
                plot_data["PRAC_SALINITY"] = gsw.conversions.SP_from_C(
                    plot_data["CNDC"],
                    plot_data["TEMP"],
                    plot_data["PRES"],
                )

                ax.plot(
                    plot_data["PRAC_SALINITY"],
                    plot_data["DEPTH"],
                    marker="o",
                    ls="",
                    c=col,
                    label=label
                )

            ax.set(
                xlabel="Practical Salinity",
                ylabel="Depth",
                title=title
            )
            ax.legend(loc="upper right")

        fig.tight_layout()
        plt.show(block=True)