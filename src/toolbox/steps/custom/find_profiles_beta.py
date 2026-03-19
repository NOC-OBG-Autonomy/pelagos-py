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

"""Class definition for exporting data steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter as tk
from scipy.signal import savgol_filter

# --- Tweakable Variables ---
DEFAULT_RESAMPLE_CADENCE = "1min"
DEFAULT_DEPTH_MEDIAN_WIN = 1
DEFAULT_DEPTH_MEAN_WIN = 2
DEFAULT_SAVGOL_WINDOW = 5
DEFAULT_SAVGOL_POLY = 2
DEFAULT_VELOCITY_THRESHOLD = 0.05
DEFAULT_MIN_VALID_DEPTH = -0.5
DEFAULT_MIN_PROFILE_DEPTH = 15
DEFAULT_MIN_POINTS_IN_PROFILE = 10
DEFAULT_MAX_DEPTH_GAP = 15

# Plotting Aesthetics
COLOUR_UP = "tab:blue"
COLOUR_DOWN = "tab:green"
COLOUR_TURNING = "tab:orange"
COLOUR_VELOCITY = "tab:red"
COLOUR_TREND = "black"
RAW_ALPHA = 0.6
MARKER_SIZE = 2
LINE_WIDTH = 1.5
# ---------------------------

def find_profiles_beta(df_sorted, cadence, med_win, mean_win, sg_win, sg_poly, vel_thresh, min_depth, min_prof_depth, min_points, max_gap, depth_col):
    """
    Identifies vertical profiles using centered smoothing, Savitzky-Golay velocity 
    filtering, and secondary segment validation.

    This function processes depth-time data to identify discrete profiling events. 
    It applies a multi-stage smoothing process to velocity, identifies potential 
    profile boundaries, and then performs a secondary validation pass to split 
    profiles containing large data gaps or remove segments that do not meet 
    physical requirements.

    Parameters
    ----------
    df_sorted : pandas.DataFrame
        Input dataframe containing time-indexed depth measurements.
    cadence : str
        Resampling frequency (e.g., '1min') used for initial smoothing and 
        velocity calculations.
    med_win, mean_win : int
        Rolling window sizes for median and mean smoothing applied to the 
        resampled depth data.
    sg_win, sg_poly : int
        Window size and polynomial order for the Savitzky-Golay filter 
        applied to vertical velocity.
    vel_thresh : float
        Velocity threshold (m/s) below which the glider is considered to be 
        turning or at a standstill.
    min_depth : float
        Depth threshold (m) below which data is considered surface noise 
        and excluded from profiles.
    min_prof_depth : float
        Minimum total vertical distance (m) a segment must cover to be 
        validated as a profile.
    min_points : int
        Minimum number of raw data points required within a segment to be 
        validated as a profile.
    max_gap : float
        Maximum allowable vertical gap (m) between consecutive points within 
         a single profile. Exceeding this splits the profile.
    depth_col : str
        The name of the column containing depth/pressure data.

    Returns
    -------
    df_out : pandas.DataFrame
        The original dataframe with added columns:
        - 'PROFILE_ID': Unique integer ID for each profile (-1 for non-profile).
        - 'DIRECTION': 1 for Ascending, -1 for Descending, NaN for Turning.
        - 'GRADIENT': Average vertical velocity (m/s) calculated via linear fit.
        - 'is_turning': Boolean indicating turning points or gaps.
    df_smooth : pandas.DataFrame
        The resampled/smoothed diagnostic data used for processing.
    """
    df = df_sorted[depth_col].resample(cadence).mean().to_frame()
    df[depth_col] = df[depth_col].interpolate(method='linear')

    df["SMOOTH_DEPTH"] = (
        df[depth_col]
        .rolling(window=med_win, center=True).median()
        .rolling(window=mean_win, center=True).mean()
    )

    dt = pd.Timedelta(cadence).total_seconds()
    df["RAW_VEL"] = np.gradient(df["SMOOTH_DEPTH"]) / dt
    df["RAW_VEL"] = df["RAW_VEL"].fillna(0)
    
    df["SMOOTH_VELOCITY"] = savgol_filter(df["RAW_VEL"], sg_win, sg_poly)
    vel_crosses_zero = (df["SMOOTH_VELOCITY"] * df["SMOOTH_VELOCITY"].shift(1)) < 0
    
    df["is_turning"] = (
        (df["SMOOTH_VELOCITY"].abs() <= vel_thresh) | 
        (df["SMOOTH_DEPTH"] < min_depth) |
        vel_crosses_zero
    )

    is_profile = ~df["is_turning"]
    profile_starts = is_profile & ~is_profile.shift(1, fill_value=False)
    df["PROFILE_ID"] = profile_starts.cumsum()
    df.loc[df["is_turning"], "PROFILE_ID"] = np.nan

    df_features = df[["PROFILE_ID", "is_turning", "SMOOTH_VELOCITY"]]
    
    df_out = pd.merge_asof(
        df_sorted, 
        df_features, 
        left_index=True, 
        right_index=True, 
        direction="nearest", 
        tolerance=pd.Timedelta(cadence)
    )

    df_out["VALID_PROFILE"] = np.nan
    df_out["DIRECTION"] = np.nan
    df_out["GRADIENT"] = np.nan
    
    valid_pid_counter = 1
    
    for pid, group in df_out.dropna(subset=["PROFILE_ID"]).groupby("PROFILE_ID"):
        depth_diffs = group[depth_col].diff().abs()
        sub_groups = (depth_diffs > max_gap).fillna(False).cumsum()
        
        for sub_id, sub_group in group.groupby(sub_groups):
            depth_span = sub_group[depth_col].max() - sub_group[depth_col].min()
            point_count = len(sub_group)
            
            if depth_span >= min_prof_depth and point_count >= min_points:
                df_out.loc[sub_group.index, "VALID_PROFILE"] = valid_pid_counter
                x = (sub_group.index - sub_group.index[0]).total_seconds().values
                
                if len(x) > 1:
                    m, _ = np.polyfit(x, sub_group[depth_col].values, 1)
                    df_out.loc[sub_group.index, "GRADIENT"] = m
                    df_out.loc[sub_group.index, "DIRECTION"] = 1 if m < 0 else -1
                    
                valid_pid_counter += 1
            else:
                df_out.loc[sub_group.index, "is_turning"] = True

    df_out = df_out.drop(columns=["PROFILE_ID"])
    df_out = df_out.rename(columns={"VALID_PROFILE": "PROFILE_ID"})
    df_out["PROFILE_ID"] = df_out["PROFILE_ID"].fillna(-1)

    return df_out, df


@register_step
class FindProfilesBetaStep(BaseStep, QCHandlingMixin):
    step_name = "Find Profiles Beta"

    def run(self):
        self.log("Attempting to designate profile numbers, directions, and gradients")
        self.filter_qc()

        self.depth_col = self.parameters.get("depth_column")
        if not self.depth_col:
            if "PRES_ENG" in self.data.variables:
                self.depth_col = "PRES_ENG"
                self.log("Automatically selected PRES_ENG as depth variable.")
            elif "PRES" in self.data.variables:
                self.depth_col = "PRES"
                self.log("PRES_ENG not found. Falling back to PRES.")
            else:
                raise ValueError("Neither PRES_ENG nor PRES variables found in the dataset.")
        elif self.depth_col not in self.data.variables:
            raise ValueError(f"Specified depth column '{self.depth_col}' not found in the dataset.")

        self.cadence = self.parameters.get("resample_cadence", DEFAULT_RESAMPLE_CADENCE)
        self.med_win = self.parameters.get("depth_median_win", DEFAULT_DEPTH_MEDIAN_WIN)
        self.mean_win = self.parameters.get("depth_mean_win", DEFAULT_DEPTH_MEAN_WIN)
        self.sg_win = self.parameters.get("savgol_window", DEFAULT_SAVGOL_WINDOW)
        self.sg_poly = self.parameters.get("savgol_poly", DEFAULT_SAVGOL_POLY)
        self.vel_thresh = self.parameters.get("velocity_threshold", DEFAULT_VELOCITY_THRESHOLD)
        self.min_depth = self.parameters.get("min_valid_depth", DEFAULT_MIN_VALID_DEPTH)
        self.min_prof_depth = self.parameters.get("min_profile_depth", DEFAULT_MIN_PROFILE_DEPTH)
        self.min_points = self.parameters.get("min_points_in_profile", DEFAULT_MIN_POINTS_IN_PROFILE)
        self.max_gap = self.parameters.get("max_depth_gap", DEFAULT_MAX_DEPTH_GAP)

        if self.depth_col == "PRES_ENG" and "PRES" in self.data.variables:
            pres_max = float(self.data["PRES"].max())
            eng_max = float(self.data["PRES_ENG"].max())
            ratio = pres_max / eng_max if eng_max != 0 else 1
            if 8 < ratio < 12:
                self.log("Detected PRES_ENG 10x bug. Scaling PRES_ENG by 10.")
                self.data["PRES_ENG"] = self.data["PRES_ENG"] * 10

        if self.diagnostics:
            self.log("Generating diagnostics")
            root = self.generate_diagnostics()
            root.mainloop()

        df_raw = self.data[["TIME", self.depth_col]].to_dataframe().reset_index()
        df_sorted = df_raw.dropna(subset=[self.depth_col, "TIME"]).sort_values("TIME").set_index("TIME")

        df_out, _ = find_profiles_beta(
            df_sorted, self.cadence, self.med_win, self.mean_win, 
            self.sg_win, self.sg_poly, self.vel_thresh, self.min_depth,
            self.min_prof_depth, self.min_points, self.max_gap, self.depth_col
        )

        df_out = df_out.reset_index()
        df_final = df_raw.merge(
            df_out[["N_MEASUREMENTS", "PROFILE_ID", "DIRECTION", "GRADIENT"]], 
            on="N_MEASUREMENTS", 
            how="left"
        )

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), df_final["PROFILE_ID"].fillna(-1).to_numpy())
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. #=-1 indicates no profile, #>=0 are profiles.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": -1,
            "valid_max": np.inf,
        }

        self.data["PROFILE_DIRECTION"] = (("N_MEASUREMENTS",), df_final["DIRECTION"].to_numpy())
        self.data.PROFILE_DIRECTION.attrs = {
            "long_name": "Profile Direction (-1: Descending, 1: Ascending, NaN: Not Profile)",
            "units": "None",
        }

        self.data["PROFILE_GRADIENT"] = (("N_MEASUREMENTS",), df_final["GRADIENT"].to_numpy())
        self.data.PROFILE_GRADIENT.attrs = {
            "long_name": "Profile Vertical Gradient",
            "units": "m/s",
        }

        self.generate_qc({
            "PROFILE_NUMBER_QC": ["TIME_QC", f"{self.depth_col}_QC"],
            "PROFILE_DIRECTION_QC": ["TIME_QC", f"{self.depth_col}_QC"],
            "PROFILE_GRADIENT_QC": ["TIME_QC", f"{self.depth_col}_QC"]
        })

        self.context["data"] = self.data
        return self.context


    def generate_diagnostics(self):
        def generate_plot():
            mpl.use("TkAgg")

            df_raw = self.data[["TIME", self.depth_col]].to_dataframe().reset_index()
            df_sorted = df_raw.dropna(subset=[self.depth_col, "TIME"]).sort_values("TIME").set_index("TIME")

            df_out, df_smooth = find_profiles_beta(
                df_sorted, self.cadence, self.med_win, self.mean_win, 
                self.sg_win, self.sg_poly, self.vel_thresh, self.min_depth,
                self.min_prof_depth, self.min_points, self.max_gap, self.depth_col
            )

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})

            up_mask = df_out["DIRECTION"] == 1
            down_mask = df_out["DIRECTION"] == -1
            turn_mask = df_out["PROFILE_ID"] == -1

            ax1.plot(df_out[turn_mask].index, -df_out[turn_mask][self.depth_col], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_TURNING, alpha=RAW_ALPHA, label="Turning")
            ax1.plot(df_out[up_mask].index, -df_out[up_mask][self.depth_col], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_UP, alpha=RAW_ALPHA, label="Ascending (+1)")
            ax1.plot(df_out[down_mask].index, -df_out[down_mask][self.depth_col], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_DOWN, alpha=RAW_ALPHA, label="Descending (-1)")
            
            ax1.set_ylabel(self.depth_col)
            ax1.set_title("Profile Classification")
            ax1.legend(loc="upper right", markerscale=5)

            ax2.plot(df_smooth.index, df_smooth["SMOOTH_VELOCITY"], color=COLOUR_VELOCITY, lw=LINE_WIDTH, label="Smoothed Velocity")
            ax2.axhline(self.vel_thresh, color=COLOUR_TURNING, lw=0.8, ls="--", alpha=0.5)
            ax2.axhline(-self.vel_thresh, color=COLOUR_TURNING, lw=0.8, ls="--", alpha=0.5)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set_ylabel("Velocity")
            ax2.legend(loc="upper right")

            ax3.plot(df_out.index, df_out["PROFILE_ID"], color="gray")
            ax3.set_ylabel("Profile ID")
            ax3.set_xlabel("Time")

            plt.tight_layout()
            plt.show(block=False)

        root = tk.Tk()
        root.title("Parameter Adjustment")
        root.geometry("450x300")
        entries = {}

        params = [
            ("Depth Column", "depth_column", self.depth_col),
            ("Velocity Threshold", "velocity_threshold", self.vel_thresh),
            ("Median Window", "depth_median_win", self.med_win),
            ("Mean Window", "depth_mean_win", self.mean_win),
            ("Savgol Window", "savgol_window", self.sg_win),
            ("Savgol Poly", "savgol_poly", self.sg_poly),
            ("Min Valid Depth", "min_valid_depth", self.min_depth),
            ("Min Profile Depth", "min_profile_depth", self.min_prof_depth),
            ("Min Points", "min_points_in_profile", self.min_points),
            ("Max Depth Gap", "max_depth_gap", self.max_gap)
        ]

        for i, (label_text, key, val) in enumerate(params):
            tk.Label(root, text=label_text).grid(row=i//2, column=(i%2)*2, sticky="e", padx=5, pady=2)
            entry = tk.Entry(root, width=12)
            entry.insert(0, str(val))
            entry.grid(row=i//2, column=(i%2)*2+1, padx=5, pady=2)
            entries[key] = entry

        def on_cancel(event=None):
            plt.close('all')
            root.quit()
            root.destroy()

        def on_regenerate(event=None):
            self.depth_col = entries["depth_column"].get()
            self.vel_thresh = float(entries["velocity_threshold"].get())
            self.med_win = int(entries["depth_median_win"].get())
            self.mean_win = int(entries["depth_mean_win"].get())
            self.sg_win = int(entries["savgol_window"].get())
            self.sg_poly = int(entries["savgol_poly"].get())
            self.min_depth = float(entries["min_valid_depth"].get())
            self.min_prof_depth = float(entries["min_profile_depth"].get())
            self.min_points = int(entries["min_points_in_profile"].get())
            self.max_gap = float(entries["max_depth_gap"].get())
            
            plt.close('all')
            generate_plot()

        def on_save(event=None):
            self.update_parameters(
                depth_column=self.depth_col,
                velocity_threshold=self.vel_thresh,
                depth_median_win=self.med_win,
                depth_mean_win=self.mean_win,
                savgol_window=self.sg_win,
                savgol_poly=self.sg_poly,
                min_valid_depth=self.min_depth,
                min_profile_depth=self.min_prof_depth,
                min_points_in_profile=self.min_points,
                max_depth_gap=self.max_gap
            )
            plt.close('all')
            root.quit()
            root.destroy()

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=(len(params)//2)+1, column=0, columnspan=4, pady=15)

        tk.Button(btn_frame, text="Regenerate", command=on_regenerate).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save", command=on_save).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)

        root.bind('<Return>', on_regenerate)
        root.bind('<Escape>', on_cancel)
        root.bind('<Command-s>', on_save)
        root.bind('<Control-s>', on_save)

        generate_plot()
        return root