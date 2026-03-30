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

    # Compute the moving average, handling NaNs properly. 
    # Providing an 'out' array prevents uninitialised memory warnings.
    avg = np.divide(
        sum_vals, 
        count_vals, 
        out=np.full_like(sum_vals, np.nan, dtype=float), 
        where=(count_vals != 0)
    )

    return avg

def compute_optimal_lag(profile_data, filter_window_size, time_col):
    """
    Calculate the optimal conductivity time lag relative to temperature to reduce salinity spikes for each glider profile.
    """

    profile_data = profile_data[
        [time_col,
         "CNDC",
         "PRES",
         "TEMP"]
    ].dropna(dim="N_MEASUREMENTS", subset=["CNDC"])

    if len(profile_data[time_col]) == 0:
        return np.nan

    t0 = profile_data[time_col].values[0]
    elapsed_time = (profile_data[time_col] - t0).dt.total_seconds().values
    
    temp_vals = profile_data["TEMP"].values
    pres_vals = profile_data["PRES"].values

    conductivity_from_time = interpolate.interp1d(
        elapsed_time,
        profile_data["CNDC"].values,
        bounds_error=False
    )

    time_lags = np.array(
        [np.linspace(-2, 2, 41),
         np.full(41, np.nan)]
    ).T

    for i, lag in enumerate(time_lags[:, 0].copy()):
        time_shifted_conductivity = conductivity_from_time(elapsed_time + lag)
        
        cndc_scaled = time_shifted_conductivity * 10 if np.nanmax(time_shifted_conductivity) < 10 else time_shifted_conductivity
        
        PSAL = gsw.conversions.SP_from_C(
            cndc_scaled,
            temp_vals,
            pres_vals
        )

        PSAL_Smooth = running_average_nan(PSAL, filter_window_size)
        PSAL_Diff = PSAL - PSAL_Smooth
        time_lags[i, 1] = np.nanstd(PSAL_Diff)

    best_score_index = np.argmin(time_lags[:, 1])
    return time_lags[best_score_index, 0]

@register_step
class AdjustSalinity(BaseStep, QCHandlingMixin):
    step_name = "Salinity Adjustment"
    required_variables = ["TIME", "PROFILE_NUMBER", "CNDC", "TEMP", "PRES"]
    provided_variables = []

    def run(self):
        self.log(f"Running adjustment...")

        self.data_copy = self.data.copy(deep=True)

        self.time_col = "TIME_CTD"
        if self.time_col not in self.data:
            self.log("TIME_CTD cound not be found. Defaulting to TIME instead.")
            self.time_col = "TIME"

        self.filter_qc()

        self.correct_ct_lag()
        self.correct_thermal_lag()

        self.reconstruct_data()
        self.update_qc()

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context


    def correct_ct_lag(self):
        profile_numbers = np.unique(self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values)

        self.per_profile_optimal_lag = np.full((len(profile_numbers), 3), np.nan)

        # Extract numpy arrays for fast filtering to avoid Xarray subsetting overhead
        prof_arr = self.data["PROFILE_NUMBER"].values
        dir_arr = self.data["PROFILE_DIRECTION"].values
        time_arr = self.data[self.time_col].values
        cndc_arr = self.data["CNDC"].values

        # Shuffle indices to randomly sample profiles across the dataset
        indices = np.random.permutation(len(profile_numbers))
        
        processed_counts = {-1: 0, 1: 0, 0: 0}
        max_profiles = 50

        for i in tqdm(indices, colour="green", desc='\033[97mCT Lag Progress\033[0m', unit="prof"):
            # Break the loop entirely once we have enough profiles to save computation
            if all(count >= max_profiles for count in processed_counts.values()):
                break
                
            profile_number = profile_numbers[i]
            
            # Fast numpy subsetting
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
            valid_times = prof_times[~np.isnat(prof_times)]
            
            if len(valid_times) > 0:
                duration = valid_times[-1] - valid_times[0]
                
                # Check if profile lasted >= 30 minutes and has enough points for the filter
                if duration >= np.timedelta64(30, 'm') and len(valid_times) > 3 * self.filter_window_size:
                    # Only subset Xarray when we know we are calculating the lag
                    profile = self.data.isel(N_MEASUREMENTS=prof_indices)
                    optimal_lag = compute_optimal_lag(profile, self.filter_window_size, self.time_col)
                    
                    self.per_profile_optimal_lag[i, :] = [profile_number, optimal_lag, prof_direction]
                    processed_counts[prof_direction] += 1

        # Use numpy mask to extract valid data for interpolation
        valid_data_mask = (~np.isnan(cndc_arr)) & (~np.isnat(time_arr))
        
        if not np.any(valid_data_mask):
            self.log("No valid CNDC data found. Skipping CT lag correction.")
            return

        # Calculate medians for Downcast (-1), Upcast (1), and Transect (0)
        self.ct_lag_medians = {}

        for d in [-1, 1, 0]:
            mask = (self.per_profile_optimal_lag[:, 2] == d) & (~np.isnan(self.per_profile_optimal_lag[:, 1]))
            dir_lags = self.per_profile_optimal_lag[mask, 1]
            if len(dir_lags) > 0:
                self.ct_lag_medians[d] = np.median(dir_lags)
            else:
                self.ct_lag_medians[d] = 0.0  # Default to 0 if no valid profiles survive the 30min check

        valid_times = time_arr[valid_data_mask]
        valid_cndc = cndc_arr[valid_data_mask]
        valid_dirs = dir_arr[valid_data_mask]

        t0 = valid_times[0]
        elapsed_time = (valid_times - t0) / np.timedelta64(1, 's')
        
        CNDC_from_TIME = interpolate.interp1d(
            elapsed_time, 
            valid_cndc, 
            bounds_error=False
        )

        corrected_cndc = valid_cndc.copy()

        for d in [-1, 1, 0]:
            dir_mask = valid_dirs == d
            if np.any(dir_mask):
                shifted_time = elapsed_time[dir_mask] + self.ct_lag_medians[d]
                corrected_cndc[dir_mask] = CNDC_from_TIME(shifted_time)
        
        # Apply corrections back to the main array directly
        final_cndc = cndc_arr.copy()
        final_cndc[valid_data_mask] = corrected_cndc
        self.data["CNDC"].values = final_cndc


    def correct_thermal_lag(self):
        """
        Applies thermal mass correction independently to the downcast, transect, and upcast.
        """
        import scipy.signal

        corrected_temp_array = np.full(len(self.data["TEMP"]), np.nan)
        profile_numbers = np.unique(self.data["PROFILE_NUMBER"].dropna(dim="N_MEASUREMENTS").values)
        
        self.filter_params = {}
        
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

                alpha_offset = 0.0135
                alpha_slope = 0.0264
                tau_offset = 7.1499
                tau_slope = 2.7858
                flow_rate = 0.4867

                tau = tau_offset + tau_slope / np.sqrt(flow_rate)
                alpha = alpha_offset + alpha_slope / flow_rate
                
                self.filter_params = {"alpha": alpha, "tau": tau}

                nyquist_frequency = 1/2 
                a = 4 * nyquist_frequency * alpha * tau / (1 + 4 * nyquist_frequency * tau)
                b = 1 - (2 * a / alpha)

                delta_TEMP = np.zeros_like(TEMP_1Hz_sampling)
                delta_TEMP[1:] = np.diff(TEMP_1Hz_sampling)
                TEMP_correction = scipy.signal.lfilter([a], [1, b], delta_TEMP)
                
                corrected_TEMP_1Hz_sampling = TEMP_1Hz_sampling - TEMP_correction

                corrected_TEMP_from_TIME = interpolate.interp1d(
                    TIME_1Hz_sampling, 
                    corrected_TEMP_1Hz_sampling, 
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                
                corrected_temp_array[indices] = corrected_TEMP_from_TIME(elapsed_time)

        final_temp = np.where(np.isnan(corrected_temp_array), self.data["TEMP"].values, corrected_temp_array)
        self.data["TEMP"][:] = final_temp

    def generate_diagnostics(self):
        COLOUR_RAW = "indianred"
        COLOUR_CORRECTED = "steelblue"
        
        mpl.use("tkagg")
        
        fig = plt.figure(figsize=(15, 5), dpi=150)
        gs = fig.add_gridspec(1, 6, wspace=0.4)
        ax_lag = fig.add_subplot(gs[0, 0:2])
        ax_therm_down = fig.add_subplot(gs[0, 2])
        ax_therm_trans = fig.add_subplot(gs[0, 3], sharey=ax_therm_down)
        ax_therm_up = fig.add_subplot(gs[0, 4], sharey=ax_therm_down)
        ax_prof = fig.add_subplot(gs[0, 5])

        plt.setp(ax_therm_trans.get_yticklabels(), visible=False)
        plt.setp(ax_therm_up.get_yticklabels(), visible=False)

        # Handle purely empty profiles gently
        if np.all(np.isnan(self.per_profile_optimal_lag[:, 0])):
            prof_min, prof_max = 0, 1
        else:
            prof_min = np.nanmin(self.per_profile_optimal_lag[:, 0])
            prof_max = np.nanmax(self.per_profile_optimal_lag[:, 0])

        colours_dir = {-1: "forestgreen", 1: "royalblue", 0: "mediumorchid"}
        labels_dir = {-1: "Down", 1: "Up", 0: "Trans"}

        for d in [-1, 1, 0]:
            if d in self.ct_lag_medians:
                med_val = self.ct_lag_medians[d]
                ax_lag.plot([prof_min, prof_max], [med_val, med_val], 
                            c=colours_dir[d], ls="--", lw=2, label=f"{labels_dir[d]}: {med_val:.2f}s")

            mask = self.per_profile_optimal_lag[:, 2] == d
            ax_lag.scatter(self.per_profile_optimal_lag[mask, 0], self.per_profile_optimal_lag[mask, 1], 
                           c=colours_dir[d], s=10, alpha=0.6)

        ax_lag.plot([prof_min, prof_max], [0, 0], "k-", lw=1)
        ax_lag.set_xlabel("Profile Index")
        ax_lag.set_ylabel("CTlag (s)")
        ax_lag.set_title("Optimal C-T Lag per Profile")
        ax_lag.legend(loc="upper right", fontsize=8)
        ax_lag.grid(True, alpha=0.3)

        p_start, p_end = self.plot_profiles_in_range
        mask_range = (self.data_copy["PROFILE_NUMBER"] >= p_start) & (self.data_copy["PROFILE_NUMBER"] <= p_end)
        
        uncorrected = self.data_copy.where(mask_range, drop=True)
        corrected = self.data.where(mask_range, drop=True)

        if len(uncorrected["DEPTH"].dropna(dim="N_MEASUREMENTS")) == 0:
            ax_therm_down.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax_prof.text(0.5, 0.5, "No Data", ha="center", va="center")
        else:
            delta_temp = uncorrected["TEMP"] - corrected["TEMP"]
            
            alpha_val = self.filter_params.get('alpha', 0.0)
            tau_val = self.filter_params.get('tau', 0.0)
            
            ax_therm_trans.set_title(f"Thermal Error (°C)\nalpha={alpha_val:.3f}, tau={tau_val:.1f}s\n\nTransect", fontsize=9)
            ax_therm_down.set_title("\n\nDowncast", fontsize=9)
            ax_therm_up.set_title("\n\nUpcast", fontsize=9)

            for direction, ax_therm, marker in zip([-1, 0, 1], [ax_therm_down, ax_therm_trans, ax_therm_up], ["o", "s", "x"]):
                mask_dir = uncorrected["PROFILE_DIRECTION"] == direction
                if np.any(mask_dir):
                    ax_therm.plot(delta_temp[mask_dir], uncorrected["DEPTH"][mask_dir], 
                                  ls="", marker=marker, markersize=3, alpha=0.5, c=colours_dir[direction])
                
                ax_therm.axvline(0, color="black", ls="--", alpha=0.5)
                ax_therm.grid(True, alpha=0.3)

            # Invert the shared y-axis so 0 is at the top
            ax_therm_down.invert_yaxis()
            ax_therm_down.set_ylabel("Depth (m)")

            cndc_raw_max = uncorrected["CNDC"].max(skipna=True).values
            cndc_raw = uncorrected["CNDC"] * 10 if not np.isnan(cndc_raw_max) and cndc_raw_max < 10 else uncorrected["CNDC"]
            temp_raw = uncorrected["TEMP"]
            
            cndc_lagged_max = corrected["CNDC"].max(skipna=True).values
            cndc_lagged = corrected["CNDC"] * 10 if not np.isnan(cndc_lagged_max) and cndc_lagged_max < 10 else corrected["CNDC"]
            temp_therm = corrected["TEMP"]
            
            pres = uncorrected["PRES"]
            
            psal_raw = gsw.conversions.SP_from_C(cndc_raw, temp_raw, pres)
            psal_full = gsw.conversions.SP_from_C(cndc_lagged, temp_therm, pres)

            for direction, marker in zip([-1, 1, 0], ["o", "x", "s"]):
                mask_dir = uncorrected["PROFILE_DIRECTION"] == direction
                if np.any(mask_dir):
                    label_raw = "Raw" if direction == -1 else None
                    label_corr = "Corrected" if direction == -1 else None
                    
                    ax_prof.plot(psal_raw[mask_dir], uncorrected["DEPTH"][mask_dir], 
                                 c=COLOUR_RAW, ls="", marker=marker, markersize=2, alpha=0.4, label=label_raw)
                    
                    ax_prof.plot(psal_full[mask_dir], uncorrected["DEPTH"][mask_dir], 
                                 c=COLOUR_CORRECTED, ls="", marker=marker, markersize=2, alpha=0.6, label=label_corr)
            
            leg = ax_prof.legend(loc="lower left", fontsize=8)
            handles = getattr(leg, "legend_handles", getattr(leg, "legendHandles", []))
            for handle in handles:
                if hasattr(handle, "set_sizes"):
                    handle.set_sizes([20])
                    handle.set_alpha(1)

            # Invert the profile plot y-axis so 0 is at the top
            ax_prof.invert_yaxis()
            ax_prof.set_xlabel("Practical Salinity")
            ax_prof.set_title(f"Final Result ({p_start}-{p_end})", fontsize=10)
            ax_prof.grid(True, alpha=0.3)

        fig.subplots_adjust(top=0.85)
        plt.show(block=True)