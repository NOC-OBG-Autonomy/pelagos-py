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
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib as mpl
from scipy import interpolate
from scipy import signal
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
    avg = np.divide(
        sum_vals, 
        count_vals, 
        out=np.full_like(sum_vals, np.nan, dtype=float), 
        where=(count_vals != 0)
    )

    return avg


def compute_optimal_lag(profile_data, filter_window_size, time_col, return_cost_data=False):
    """
    Calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
    Minimise the standard deviation of the difference between a lagged CNDC and a high-pass filtered CNDC.
    The optimal lag is returned. The lag is chosen from -2 to 2s every 0.1s.
    This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.
    This correction is described in Woo (2019) but the minimisation is done between salinity and high-pass filtered salinity (as done by RBR, https://bitbucket.org/rbr/pyrsktools/src/master/pyrsktools) instead of comparing downcast vs upcast.

    Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/ 10.26198/5c997b5fdc9bd).

    Parameters
    ----------
    profile_data: xarray.Dataset with raw CTD dataset for one single profile, which should contain:
        - TIME_CTD, sci_ctd41cp_timestamp, [numpy.datetime64]
        - PRES: pressure [dbar]
        - CNDC: conductivity [S/m]
        - TEMP: in-situ temperature [deg C]
    filter_window_size: Window length over which the high-pass filter of conductivity is applied, 21 by default.
    time_col: Name of the time coordinate to use.
    return_cost_data: Boolean flag to return intermediate optimization data for diagnostics plotting.

    Returns
    -------
    best_lag: Optimal lag time in seconds.
    cost_data: (Optional) Dictionary containing optimisation curve data.
    """
    # Remove any rows where conductivity is bad (nan)
    profile_data = profile_data[
        [time_col,
         "CNDC",
         "PRES",
         "TEMP"]
    ].dropna(dim="N_MEASUREMENTS", subset=["CNDC"])

    if len(profile_data[time_col]) == 0:
        if return_cost_data:
            return np.nan, None
        return np.nan

    # Find the elapsed time in seconds from the start of the profile
    t0 = profile_data[time_col].values[0]
    elapsed_time = (profile_data[time_col] - t0).dt.total_seconds().values
    
    temp_vals = profile_data["TEMP"].values
    pres_vals = profile_data["PRES"].values

    # Creates a callable function that predicts what CNDC would be at any given time
    conductivity_from_time = interpolate.interp1d(
        elapsed_time,
        profile_data["CNDC"].values,
        bounds_error=False
    )

    # Specify the range time lags that the optimum will be found from. Column indexes are: (lag value, lag score)
    time_lags = np.array(
        [np.linspace(-2, 2, 41),
         np.full(41, np.nan)]
    ).T

    saved_psal = {}

    # For each lag find its score and add it to the time_lags array
    for i, lag in enumerate(time_lags[:, 0].copy()):
        # Apply the time shift
        time_shifted_conductivity = conductivity_from_time(elapsed_time + lag)
        
        # Scale if necessary
        cndc_scaled = time_shifted_conductivity * 10 if np.nanmax(time_shifted_conductivity) < 10 else time_shifted_conductivity
        
        # Derive salinity with the time shifted CNDC (spiking will be minimised when CNDC and TEMP are aligned)
        PSAL = gsw.conversions.SP_from_C(
            cndc_scaled,
            temp_vals,
            pres_vals
        )

        # Smooth the salinity profile (to remove spiking)
        PSAL_Smooth = running_average_nan(PSAL, filter_window_size)

        # Subtracting the raw and smoothed salinity gives an indication of "spikiness".
        PSAL_Diff = PSAL - PSAL_Smooth

        # More spiky data will have higher standard deviation - which is used to score the effectiveness of the applied lag
        time_lags[i, 1] = np.nanstd(PSAL_Diff)

        if return_cost_data:
            saved_psal[lag] = (PSAL, PSAL_Smooth)

    # Return the time lag which has the lowest score (standard deviation)
    best_score_index = np.argmin(time_lags[:, 1])
    best_lag = time_lags[best_score_index, 0]

    # Package diagnostic data if requested
    if return_cost_data:
        zero_idx = int(np.argmin(np.abs(time_lags[:, 0])))
        zero_lag = time_lags[zero_idx, 0]

        p_best, p_smooth_best = saved_psal[best_lag]
        p_zero, p_smooth_zero = saved_psal[zero_lag]

        cost_data = {
            "lags": time_lags[:, 0],
            "costs": time_lags[:, 1],
            "best_lag": best_lag,
            "zero_lag": zero_lag,
            "elapsed_time": elapsed_time,
            "resid_zero": p_zero - p_smooth_zero,
            "resid_best": p_best - p_smooth_best
        }
        return best_lag, cost_data

    return best_lag


@register_step
class AdjustSalinity(BaseStep, QCHandlingMixin):
    step_name = "Salinity Adjustment"
    required_variables = ["TIME", "PROFILE_NUMBER", "CNDC", "TEMP", "PRES"]
    provided_variables = []
    parameter_schema = {
        "filter_window_size": {
            "type": int,
            "default": 21,
            "description": "Window length over which the running average filter is applied to salinity for smoothing during the CT lag optimisation."
        },
        "qc_handling_settings": {
            "type": dict,
            "default": {
                "flag_filter_settings": {
                    "CNDC": [3, 4, 9],
                    "TEMP": [3, 4, 9],
                    "PROFILE_NUMBER": [3, 4, 9]
                },
                "reconstruction_behaviour": "reinsert",
                "flag_mapping": {
                    0: 5,
                    1: 5,
                    2: 5
                }
            },
            "description": "Rules for filtering bad data before adjustment and assigning new flags after reconstruction. Handled by QCHandlingMixin."
        }
    }
    
    def run(self):
        """
        Apply the thermal-lag correction for Salinity presented in Morrison et al 1994.
        The temperature is estimated inside the conductivity cell to estimate Salinity.
        This is based on eq.5 of Morrison et al. (1994), which doesn't require to know the sensitivity of temperature to conductivity (eq.2 of Morrison et al. 1994).
        No attempt is done yet to minimise the coefficients alpha/tau in T/S space, as in Morrison et al. (1994) or Garau et al. (2011).
        The fixed coefficients (alpha and tau) presented in Morrison et al. (1994) are used.
        These coefficients should be valid for pumped SeaBird CTsail as described in Woo (2019) by using their flow rate in the conductivity cell.
        This function should further be adapted to unpumped CTD by taking into account the glider velocity through the water based on the pitch angle or a hydrodynamic flight model.

        Woo, L.M. (2019). Delayed Mode QA/QC Best Practice Manual Version 2.0. Integrated Marine Observing System. DOI: 10.26198/5c997b5fdc9bd (http://dx.doi.org/10.26198/5c997b5fdc9bd).

        Config Example
        --------------
          - name: "Salinity Adjustment"
            parameters:
              filter_window_size: 21
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

        # Required for plotting later
        self.data_copy = self.data.copy(deep=True)

        # Check if TIME_CTD exists
        self.time_col = "TIME_CTD"
        if self.time_col not in self.data:
            self.log("TIME_CTD could not be found. Defaulting to TIME instead.")
            self.time_col = "TIME"

        # Filter user-specified flags
        self.filter_qc()

        # Correct conductivity-temperature response time misalignment (C-T Lag)
        self.correct_ct_lag()

        # Correct thermal mass error
        self.correct_thermal_lag()

        # Reconstruct and update QC flags based on adjustments
        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context
        
    def correct_ct_lag(self):
        """
        Calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes.
        Selects a random sample of up to 50 profiles per direction (Upcast, Downcast, Transect).
        Estimates the median of this lag per direction and applies it to correct the CNDC variables.
        This correction should reduce salinity spikes that result from the misalignment between conductivity and temperature sensors and from the difference in sensor response times.
        """
        profile_numbers = np.unique(self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values)

        # Making a place to store intermediate products. Column dimensions: (profile number, time lag, direction)
        self.per_profile_optimal_lag = np.full((len(profile_numbers), 3), np.nan)
        self._ct_cost_data = None 

        prof_arr = self.data["PROFILE_NUMBER"].values
        dir_arr = self.data["PROFILE_DIRECTION"].values
        time_arr = self.data[self.time_col].values
        cndc_arr = self.data["CNDC"].values

        # Randomly permute to ensure uniform sampling across the dataset
        indices = np.random.permutation(len(profile_numbers))
        
        processed_counts = {-1: 0, 1: 0, 0: 0}
        max_profiles = 50

        # Loop through sampled profiles and store the optimal C-T lag per direction
        for i in tqdm(indices, colour="green", desc='\033[97mCT Lag Progress\033[0m', unit="prof"):
            if all(count >= max_profiles for count in processed_counts.values()):
                break
                
            profile_number = profile_numbers[i]
            prof_indices = np.where(prof_arr == profile_number)[0]
            
            if len(prof_indices) == 0:
                continue
            
            dir_subset = dir_arr[prof_indices]
            valid_dirs = dir_subset[~np.isnan(dir_subset)]
            prof_direction = valid_dirs[0] if len(valid_dirs) > 0 else np.nan
            
            if np.isnan(prof_direction) or prof_direction not in processed_counts:
                continue
                
            if processed_counts[prof_direction] >= max_profiles:
                continue
            
            prof_times = time_arr[prof_indices]
            prof_cndc = cndc_arr[prof_indices]
            
            valid_mask = (~np.isnat(prof_times)) & (~np.isnan(prof_cndc))
            valid_times = prof_times[valid_mask]
            
            if len(valid_times) > 0:
                duration = valid_times[-1] - valid_times[0]
                
                # Check for sufficient profile duration and length
                if duration >= np.timedelta64(1, 'h') and len(valid_times) > 3 * getattr(self, "filter_window_size", 21):
                    profile = self.data.isel(N_MEASUREMENTS=prof_indices)
                    
                    if getattr(self, 'diagnostics', False) and self._ct_cost_data is None:
                        optimal_lag, cost_data = compute_optimal_lag(profile, getattr(self, "filter_window_size", 21), self.time_col, return_cost_data=True)
                        self._ct_cost_data = cost_data
                    else:
                        optimal_lag = compute_optimal_lag(profile, getattr(self, "filter_window_size", 21), self.time_col)
                    
                    self.per_profile_optimal_lag[i, :] = [profile_number, optimal_lag, prof_direction]
                    processed_counts[prof_direction] += 1

        valid_data_mask = (~np.isnan(cndc_arr)) & (~np.isnat(time_arr))
        
        if not np.any(valid_data_mask):
            self.log("No valid CNDC data found. Skipping CT lag correction.")
            return

        # Find median optimal time lag across profiles by direction
        self.ct_lag_medians = {}
        for d in [-1, 1, 0]:
            mask = (self.per_profile_optimal_lag[:, 2] == d) & (~np.isnan(self.per_profile_optimal_lag[:, 1]))
            dir_lags = self.per_profile_optimal_lag[mask, 1]
            if len(dir_lags) > 0:
                self.ct_lag_medians[d] = np.median(dir_lags)
            else:
                self.ct_lag_medians[d] = 0.0

        valid_times = time_arr[valid_data_mask]
        valid_cndc = cndc_arr[valid_data_mask]
        valid_dirs = dir_arr[valid_data_mask]

        # Find the elapsed time in seconds
        t0 = valid_times[0]
        elapsed_time = (valid_times - t0) / np.timedelta64(1, 's')
        
        # Resample the data using a shifted time function
        CNDC_from_TIME = interpolate.interp1d(
            elapsed_time, 
            valid_cndc, 
            bounds_error=False
        )

        corrected_cndc = valid_cndc.copy()

        # Apply the specific median shift to each directional subset
        for d in [-1, 1, 0]:
            dir_mask = valid_dirs == d
            if np.any(dir_mask):
                shifted_time = elapsed_time[dir_mask] + self.ct_lag_medians[d]
                corrected_cndc[dir_mask] = CNDC_from_TIME(shifted_time)
        
        # Reinsert the time-shifted data back into self.data
        final_cndc = cndc_arr.copy()
        final_cndc[valid_data_mask] = corrected_cndc
        self.data["CNDC"].values = final_cndc

    def correct_thermal_lag(self):
        """
        Apply the thermal-lag correction for Salinity presented in Morrison et al 1994.
        Uses a recursive filter algorithm across distinct directional casts to estimate
        temperature inside the conductivity cell.
        """
        corrected_temp_array = np.full(len(self.data["TEMP"]), np.nan)
        profile_numbers = np.unique(self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values)

        self.filter_params = {}
        self._thermal_scatter_data = None

        prof_arr = self.data["PROFILE_NUMBER"].values
        dir_arr = self.data["PROFILE_DIRECTION"].values
        temp_arr = self.data["TEMP"].values
        time_arr = self.data[self.time_col].values

        valid_mask = (~np.isnan(temp_arr)) & (~np.isnan(time_arr))
        
        for prof in tqdm(profile_numbers, colour="blue", desc='\033[97mThermal Lag Progress\033[0m', unit="prof"):
            for direction in [-1, 1, 0]:
                
                mask = (prof_arr == prof) & (dir_arr == direction) & valid_mask
                indices = np.where(mask)[0]
                
                if len(indices) < 5:  
                    continue
                    
                cast_times = time_arr[indices]
                cast_temps = temp_arr[indices]
                
                t0 = cast_times[0]
                elapsed_time = (cast_times - t0) / np.timedelta64(1, 's')

                TEMP_from_TIME = interpolate.interp1d(elapsed_time, cast_temps, bounds_error=False, fill_value="extrapolate")
                
                TIME_1Hz_sampling = np.arange(0, elapsed_time[-1], 1)
                if len(TIME_1Hz_sampling) < 2:
                    continue
                    
                TEMP_1Hz_sampling = TEMP_from_TIME(TIME_1Hz_sampling)

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
                
                self.filter_params = {"alpha": alpha, "tau": tau}

                # Set the filter coefficients
                nyquist_frequency = 1/2 
                a = 4 * nyquist_frequency * alpha * tau / (1 + 4 * nyquist_frequency * tau)
                b = 1 - (2 * a / alpha)

                # Apply the filter using scipy
                delta_TEMP = np.zeros_like(TEMP_1Hz_sampling)
                delta_TEMP[1:] = np.diff(TEMP_1Hz_sampling)
                TEMP_correction = signal.lfilter([a], [1, b], delta_TEMP)
                
                corrected_TEMP_1Hz_sampling = TEMP_1Hz_sampling - TEMP_correction

                corrected_TEMP_from_TIME = interpolate.interp1d(
                    TIME_1Hz_sampling,
                    corrected_TEMP_1Hz_sampling,
                    bounds_error=False,
                    fill_value="extrapolate"
                )

                corrected_temp_array[indices] = corrected_TEMP_from_TIME(elapsed_time)

                # Capture diagnostic data for plotting later
                if (
                    getattr(self, "diagnostics", False)
                    and self._thermal_scatter_data is None
                    and direction in (-1, 1)
                    and TIME_1Hz_sampling[-1] >= 3600
                    and np.nanmax(TEMP_1Hz_sampling) - np.nanmin(TEMP_1Hz_sampling) >= 1.0
                ):
                    self._thermal_scatter_data = {
                        "dT_dt": np.gradient(TEMP_1Hz_sampling, TIME_1Hz_sampling),
                        "correction": TEMP_correction
                    }

        # Reinsert the corrected data back into self.data
        final_temp = np.where(np.isnan(corrected_temp_array), self.data["TEMP"].values, corrected_temp_array)
        self.data["TEMP"][:] = final_temp

    def generate_diagnostics(self):
        """
        Displays a comprehensive diagnostics dashboard detailing applied adjustments
        to conductivity and temperature, along with overall impacts on the dataset.
        """
        # --- Friendly Configuration Variables ---
        FIG_SIZE = (12, 7)
        DPI = 120
        
        # Colours
        COLOUR_RAW_T = "indianred"
        COLOUR_CORR_T = "darkred"
        COLOUR_RAW_C = "steelblue"
        COLOUR_CORR_C = "darkblue"
        COLOUR_BEST = "darkorange"
        COLOUR_ZERO = "black"
        COLOUR_SMOOTH = "dimgrey"
        COLOUR_SCATTER = "tab:purple"
        COLOUR_UNUSED = "lightgrey"
        
        # Text Styles
        TITLE_SIZE = 9
        LABEL_SIZE = 8

        # --- Data Preparation ---
        prof_arr = self.data["PROFILE_NUMBER"].values
        unique_profs = np.unique(prof_arr[~np.isnan(prof_arr)])
        
        # Build a master QC mask for plotting to ignore bad data (flags 3, 4, 9)
        plot_qc_mask = xr.ones_like(self.data_copy["PROFILE_NUMBER"], dtype=bool)
        for var in ["TEMP", "CNDC", "PRES", "DEPTH", self.time_col]:
            qc_col = f"{var}_QC"
            if qc_col in self.data_copy.data_vars:
                plot_qc_mask = plot_qc_mask & ~self.data_copy[qc_col].isin([3, 4, 9])
        
        # Extract the exact profiles that were randomly selected and processed
        valid_lags = self.per_profile_optimal_lag[~np.isnan(self.per_profile_optimal_lag[:, 1])]
        processed_profs = valid_lags[:, 0]
        processed_dirs = valid_lags[:, 2]
        prof_dir_map = dict(zip(processed_profs, processed_dirs))
        
        present_dirs = [d for d in [-1, 1, 0] if d in processed_dirs]

        # Select a sample profile from the processed pool for the cost curves
        if len(processed_profs) > 0:
            sample_prof = processed_profs[len(processed_profs) // 2]
        else:
            sample_prof = unique_profs[0] if len(unique_profs) > 0 else np.nan

        # --- Main Figure Setup ---
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, constrained_layout=True)
        gs = fig.add_gridspec(2, 3)
        
        ax_lag = fig.add_subplot(gs[0, 0])
        ax_cost = fig.add_subplot(gs[0, 1])
        ax_scatter = fig.add_subplot(gs[0, 2])
        
        ax_dur = fig.add_subplot(gs[1, 0])
        ax_sal = fig.add_subplot(gs[1, 1])
        ax_mech = fig.add_subplot(gs[1, 2])

        colours_dir = {-1: "forestgreen", 1: "royalblue", 0: "mediumorchid"}
        labels_dir = {-1: "Downcast", 1: "Upcast", 0: "Transect"}

        # (1) Row 1, Col 1: Applied Lag Distribution
        rng = np.random.default_rng(0)
        for pos, d in enumerate(present_dirs):
            mask_lag = (self.per_profile_optimal_lag[:, 2] == d) & (~np.isnan(self.per_profile_optimal_lag[:, 1]))
            data = self.per_profile_optimal_lag[mask_lag, 1]
            if len(data) > 0:
                ax_lag.scatter(rng.normal(pos, 0.05, len(data)), data, c=colours_dir[d], s=8, alpha=0.4)
                ax_lag.boxplot([data], positions=[pos], widths=0.4, patch_artist=True, showfliers=False,
                               medianprops={"color": "black"}, boxprops={"facecolor": "none", "edgecolor": colours_dir[d]})
                
        if present_dirs:
            ax_lag.set_xticks(range(len(present_dirs)))
            ax_lag.set_xticklabels([labels_dir[d] for d in present_dirs])
            
        ax_lag.set_title("Dataset Lag Distribution", fontsize=TITLE_SIZE)
        ax_lag.set_ylabel("Optimal Lag (s)", fontsize=LABEL_SIZE)
        ax_lag.tick_params(axis="both", labelsize=LABEL_SIZE)
        ax_lag.grid(True, alpha=0.2, axis="y")

        # (2) Row 1, Col 2: CT Lag Cost Curve (Example Profile)
        if self._ct_cost_data:
            c = self._ct_cost_data
            ax_cost.plot(c["lags"], c["costs"], "o-", color=COLOUR_SMOOTH, lw=1, ms=3)
            ax_cost.axvline(c["best_lag"], color=COLOUR_BEST, ls="--", label=f"Best: {c['best_lag']:.2f}s")
            ax_cost.set_xlabel("Trial Lag (s)", fontsize=LABEL_SIZE)
            ax_cost.set_ylabel("std(PSAL - smooth)", fontsize=LABEL_SIZE)
            ax_cost.set_title(f"Optimal CT Lag Search (Profile {sample_prof:.0f})", fontsize=TITLE_SIZE)
            ax_cost.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_cost.legend(fontsize=7)
            ax_cost.grid(True, alpha=0.2)

        # (3) Row 1, Col 3: Thermal Scatter & Parameters Legend
        if self._thermal_scatter_data:
            ts = self._thermal_scatter_data
            finite = np.isfinite(ts["correction"]) & np.isfinite(ts["dT_dt"])
            ax_scatter.scatter(ts["dT_dt"][finite], ts["correction"][finite], s=4, alpha=0.3, color=COLOUR_SCATTER)
            ax_scatter.set_xlabel("dT/dt (°C/s)", fontsize=LABEL_SIZE)
            ax_scatter.set_ylabel("Corr Amplitude (°C)", fontsize=LABEL_SIZE)
            ax_scatter.set_title(f"Thermal Mass Verification (Profile {sample_prof:.0f})", fontsize=TITLE_SIZE)
            ax_scatter.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_scatter.grid(True, alpha=0.2)

            alpha_val = self.filter_params.get("alpha", np.nan)
            tau_val = self.filter_params.get("tau", np.nan)
            param_text = f"Flow Velocity: ~0.49 m/s\nAlpha (α): {alpha_val:.4f}\nTau (τ): {tau_val:.2f} s"
            ax_scatter.text(0.05, 0.95, param_text, transform=ax_scatter.transAxes, fontsize=7,
                            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#ccc"))

        # (4) Row 2, Col 1: Profile Durations
        durations_hrs = np.zeros(len(unique_profs))
        t_arr = self.data_copy[self.time_col].values
        bar_colours = []
        
        for i, p in enumerate(unique_profs):
            p_idx = np.where(prof_arr == p)[0]
            valid_p_t = t_arr[p_idx][~np.isnat(t_arr[p_idx])]
            if len(valid_p_t) > 0:
                durations_hrs[i] = (valid_p_t[-1] - valid_p_t[0]) / np.timedelta64(1, "h")
            
            if p in prof_dir_map:
                bar_colours.append(colours_dir[prof_dir_map[p]])
            else:
                bar_colours.append(COLOUR_UNUSED)

        ax_dur.bar(unique_profs, durations_hrs, color=bar_colours, alpha=0.8, width=1.0)
        ax_dur.axhline(1.0, color="black", ls="--", lw=1)
        
        legend_elements = [Patch(facecolor=COLOUR_UNUSED, edgecolor="none", label="Unused")]
        for d in present_dirs:
            legend_elements.append(Patch(facecolor=colours_dir[d], edgecolor="none", label=f"Used {labels_dir[d]}"))
            
        ax_dur.set_ylabel("Hours", fontsize=LABEL_SIZE)
        ax_dur.set_xlabel("Profile Number", fontsize=LABEL_SIZE)
        ax_dur.set_title("Profile Selection & Duration", fontsize=TITLE_SIZE)
        ax_dur.tick_params(axis="both", labelsize=LABEL_SIZE)
        ax_dur.grid(True, alpha=0.2, axis="y")
        ax_dur.legend(handles=legend_elements, fontsize=7, loc="upper right")

        # (5) Row 2, Col 2: Combined Salinity Profiles (Sampled Profiles Only)
        mask_range = self.data_copy["PROFILE_NUMBER"].isin(processed_profs)
        
        # Apply the QC mask here to filter out bad points from the plots
        uncorr = self.data_copy.where(mask_range & plot_qc_mask, drop=True)
        corr = self.data.where(mask_range & plot_qc_mask, drop=True)

        if len(uncorr["DEPTH"].dropna(dim="N_MEASUREMENTS")) > 0:
            c_raw = uncorr["CNDC"].values
            c_new = corr["CNDC"].values
            c_raw = c_raw * 10 if np.nanmax(c_raw) < 10 else c_raw
            c_new = c_new * 10 if np.nanmax(c_new) < 10 else c_new
            
            p_raw = gsw.conversions.SP_from_C(c_raw, uncorr["TEMP"].values, uncorr["PRES"].values)
            p_new = gsw.conversions.SP_from_C(c_new, corr["TEMP"].values, uncorr["PRES"].values)
            
            ax_sal.plot(p_raw, uncorr["DEPTH"].values, c="grey", ls="", marker=".", ms=1, alpha=0.3, label="Raw")
            
            sal_legend = [mlines.Line2D([], [], color="grey", marker=".", ls="", markersize=4, label="Raw (All)")]
            for d in present_dirs:
                m_dir = uncorr["PROFILE_DIRECTION"] == d
                if np.any(m_dir):
                    ax_sal.plot(p_new[m_dir], uncorr["DEPTH"].values[m_dir], c=colours_dir[d], ls="", marker=".", ms=1.5, alpha=0.7)
                    sal_legend.append(mlines.Line2D([], [], color=colours_dir[d], marker=".", ls="", markersize=4, label=f"Corr {labels_dir[d]}"))
            
            ax_sal.set_title("Combined Directional Result", fontsize=TITLE_SIZE)
            ax_sal.set_xlabel("Practical Salinity", fontsize=LABEL_SIZE)
            ax_sal.set_ylabel("Depth (m)", fontsize=LABEL_SIZE)
            ax_sal.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_sal.invert_yaxis()
            ax_sal.grid(True, alpha=0.2)
            ax_sal.legend(handles=sal_legend, fontsize=7, loc="lower right")

        # (6) Row 2, Col 3: Whole Dataset Mechanics
        t_all = self.data_copy[self.time_col].values
        
        # Include the QC mask in the valid data check
        valid_t = ~np.isnat(t_all) & ~np.isnan(self.data_copy["TEMP"].values) & ~np.isnan(self.data_copy["CNDC"].values) & plot_qc_mask.values
        sub_step = max(1, np.sum(valid_t) // 50000)

        t_valid = t_all[valid_t][::sub_step]
        if len(t_valid) > 0:
            elapsed_days = (t_valid - t_valid[0]) / np.timedelta64(1, "D")

            temp_raw_all = self.data_copy["TEMP"].values[valid_t][::sub_step]
            temp_corr_all = self.data["TEMP"].values[valid_t][::sub_step]
            cndc_raw_all = self.data_copy["CNDC"].values[valid_t][::sub_step]
            cndc_corr_all = self.data["CNDC"].values[valid_t][::sub_step]

            cndc_raw_all = cndc_raw_all * 10 if np.nanmax(cndc_raw_all) < 10 else cndc_raw_all
            cndc_corr_all = cndc_corr_all * 10 if np.nanmax(cndc_corr_all) < 10 else cndc_corr_all

            ax_mech_c = ax_mech.twinx()
            
            ax_mech.plot(elapsed_days, temp_raw_all, color=COLOUR_RAW_T, marker=".", ls="", ms=1, alpha=0.2, label="Raw Temp")
            ax_mech.plot(elapsed_days, temp_corr_all, color=COLOUR_CORR_T, marker=".", ls="", ms=1, alpha=0.6, label="Corr Temp")
            
            ax_mech_c.plot(elapsed_days, cndc_raw_all, color=COLOUR_RAW_C, marker=".", ls="", ms=1, alpha=0.2, label="Raw CNDC")
            ax_mech_c.plot(elapsed_days, cndc_corr_all, color=COLOUR_CORR_C, marker=".", ls="", ms=1, alpha=0.6, label="Corr CNDC")

            ax_mech.set_xlabel("Elapsed Time (Days)", fontsize=LABEL_SIZE)
            ax_mech.set_ylabel("Temperature (°C)", fontsize=LABEL_SIZE)
            ax_mech_c.set_ylabel("Conductivity (mS/cm)", fontsize=LABEL_SIZE)
            ax_mech.set_title("Dataset-Wide Adjustments", fontsize=TITLE_SIZE)
            ax_mech.tick_params(axis="both", labelsize=LABEL_SIZE)
            ax_mech_c.tick_params(axis="y", labelsize=LABEL_SIZE)
            
            lines1, labels1 = ax_mech.get_legend_handles_labels()
            lines2, labels2 = ax_mech_c.get_legend_handles_labels()
            
            leg_handles = []
            for line in lines1 + lines2:
                leg_handles.append(mlines.Line2D([], [], color=line.get_color(), marker=".", ls="", markersize=6))
                
            ax_mech.legend(leg_handles, labels1 + labels2, loc="best", fontsize=7)
            ax_mech.grid(True, alpha=0.2)

        # Final Render
        fig.suptitle("Salinity Adjustment Diagnostics Dashboard", fontsize=11, fontweight="bold")
        plt.show(block=True)