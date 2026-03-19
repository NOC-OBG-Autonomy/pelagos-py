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

# --- Configurable Defaults ---
DEFAULT_RESAMPLE_CADENCE = "30s"
DEFAULT_GRADIENT_THRESHOLDS = [0.033, -0.033]
DEFAULT_FILTER_WINDOW_SIZES = [1, 2]
DEFAULT_EDGE_SQUEEZE = 0
DEFAULT_DIVE_SCALE = 15

# --- Fixed Variables ---
FIXED_SAVGOL_WINDOW = 5
FIXED_SAVGOL_POLY = 2
FIXED_MIN_VALID_DEPTH = -0.5
FIXED_MIN_POINTS_IN_PROFILE = 10

# --- Plot Aesthetics ---
COLOUR_UP = "tab:blue"
COLOUR_DOWN = "tab:green"
COLOUR_TURNING = "tab:orange"
COLOUR_VELOCITY = "tab:red"
MARKER_SIZE = 2
LINE_WIDTH = 1.5
# ---------------------------

def _parse_windows(win_sizes, cadence):
    cadence_sec = pd.Timedelta(cadence).total_seconds()
    parsed = []
    for w in win_sizes:
        if isinstance(w, str):
            try:
                w_sec = pd.Timedelta(w).total_seconds()
                parsed.append(max(1, int(round(w_sec / cadence_sec))))
            except ValueError:
                parsed.append(int(w))
        else:
            parsed.append(int(w))
    return parsed

def find_profiles_beta(df_sorted, cadence, filter_win_sizes, gradient_thresholds, edge_squeeze, dive_scale, depth_col):
    """
    Identifies vertical profiles using compound smoothing, velocity filtering, 
    and morphological edge squeezing.

    This function processes depth-time data to identify discrete profiling events. 
    It applies median and mean filters, calculates vertical velocity, and isolates 
    periods where the instrument is actively profiling based on asymmetric 
    gradient thresholds. An edge-squeezing mechanism reclaims data points at the 
    boundaries of turning regions to maximise profile length.

    Parameters
    ----------
    df_sorted : pandas.DataFrame
        Input dataframe containing time-indexed depth measurements.
    cadence : str
        Resampling frequency (e.g., '30s') used to regularise the time series.
    filter_win_sizes : list
        Two-element list defining the rolling median and mean window sizes. 
        Accepts integer row counts or time duration strings (e.g., ['20s', '10s']) 
        which are automatically parsed relative to the cadence.
    gradient_thresholds : list
        Two-element list [positive_threshold, negative_threshold] defining the 
        vertical velocity bounds (m/s). Velocities between these bounds are 
        classified as turning or at a standstill.
    edge_squeeze : int
        Number of points to iteratively erode from the edges of identified turning 
        regions to push more data into the adjacent ascending/descending profiles.
    dive_scale : float
        Minimum total vertical distance (m) a segment must cover to be validated 
        as a profile. Also acts as the maximum allowable gap between points.
    depth_col : str
        Name of the column containing depth or pressure data.

    Returns
    -------
    df_out : pandas.DataFrame
        Dataframe aligned to the original indices with added columns:
        - 'PROFILE_ID': Unique integer ID for each profile (NaN for turning).
        - 'DIRECTION': 1 for Ascending, -1 for Descending, NaN for turning.
        - 'GRADIENT': Average vertical velocity (m/s) via linear fit.
    df : pandas.DataFrame
        The resampled and smoothed diagnostic dataframe.
    """
    df = df_sorted[depth_col].resample(cadence).mean().to_frame()
    df[depth_col] = df[depth_col].interpolate(method='linear')

    windows = _parse_windows(filter_win_sizes, cadence)
    med_win, mean_win = windows[0], windows[1]
    
    df["SMOOTH_DEPTH"] = (
        df[depth_col]
        .rolling(window=med_win, center=True).median()
        .rolling(window=mean_win, center=True).mean()
    )

    dt = pd.Timedelta(cadence).total_seconds()
    df["RAW_VEL"] = np.gradient(df["SMOOTH_DEPTH"]) / dt
    df["RAW_VEL"] = df["RAW_VEL"].fillna(0)
    
    df["SMOOTH_VELOCITY"] = savgol_filter(df["RAW_VEL"], FIXED_SAVGOL_WINDOW, FIXED_SAVGOL_POLY)
    vel_crosses_zero = (df["SMOOTH_VELOCITY"] * df["SMOOTH_VELOCITY"].shift(1)) < 0
    
    pos_grad, neg_grad = gradient_thresholds
    df["is_turning"] = (
        ((df["SMOOTH_VELOCITY"] >= neg_grad) & (df["SMOOTH_VELOCITY"] <= pos_grad)) | 
        (df["SMOOTH_DEPTH"] < FIXED_MIN_VALID_DEPTH) |
        vel_crosses_zero
    )

    turn_mask = df["is_turning"].to_numpy(copy=True)
    if edge_squeeze > 0:
        for _ in range(int(edge_squeeze)):
            shifted_left = np.roll(turn_mask, -1)
            shifted_right = np.roll(turn_mask, 1)
            shifted_left[-1] = turn_mask[-1]
            shifted_right[0] = turn_mask[0]
            
            is_edge = turn_mask & (~shifted_left | ~shifted_right)
            is_single = turn_mask & ~shifted_left & ~shifted_right
            
            to_erode = is_edge & ~is_single
            turn_mask[to_erode] = False
            
    df["is_turning"] = turn_mask

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
        sub_groups = (depth_diffs > dive_scale).fillna(False).cumsum()
        
        for sub_id, sub_group in group.groupby(sub_groups):
            depth_span = sub_group[depth_col].max() - sub_group[depth_col].min()
            point_count = len(sub_group)
            
            if depth_span >= dive_scale and point_count >= FIXED_MIN_POINTS_IN_PROFILE:
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

    return df_out, df


@register_step
class FindProfilesBetaStep(BaseStep, QCHandlingMixin):
    step_name = "Find Profiles Beta"

    def run(self):
        self.log("Attempting to designate profile numbers, directions, and gradients")
        self.filter_qc()

        retired_params = [
            "savgol_window", "savgol_poly", "min_valid_depth", 
            "min_points_in_profile", "min_profile_depth", "max_depth_gap"
        ]
        for param in retired_params:
            if param in self.parameters:
                self.log(f"Notification: Parameter '{param}' is no longer required and will be ignored.")

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
        self.gradient_thresholds = self.parameters.get("gradient_thresholds", DEFAULT_GRADIENT_THRESHOLDS)
        self.filter_win_sizes = self.parameters.get("filter_window_sizes", DEFAULT_FILTER_WINDOW_SIZES)
        self.edge_squeeze = self.parameters.get("edge_squeeze", DEFAULT_EDGE_SQUEEZE)
        
        self.dive_scale = self.parameters.get("dive_scale", DEFAULT_DIVE_SCALE)
        if "dive_scale" not in self.parameters and "min_profile_depth" in self.parameters:
            self.dive_scale = self.parameters.get("min_profile_depth")

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
            df_sorted, self.cadence, self.filter_win_sizes, 
            self.gradient_thresholds, self.edge_squeeze,
            self.dive_scale, self.depth_col
        )

        df_out = df_out.reset_index()
        df_final = df_raw.merge(
            df_out[["N_MEASUREMENTS", "PROFILE_ID", "DIRECTION", "GRADIENT"]], 
            on="N_MEASUREMENTS", 
            how="left"
        )

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), df_final["PROFILE_ID"].to_numpy())
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. NaN indicates no profile, #>=1 are profiles.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": 1,
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
                df_sorted, self.cadence, self.filter_win_sizes, 
                self.gradient_thresholds, self.edge_squeeze,
                self.dive_scale, self.depth_col
            )

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})

            up_mask = df_out["DIRECTION"] == 1
            down_mask = df_out["DIRECTION"] == -1
            turn_mask = df_out["PROFILE_ID"].isna()

            ax1.plot(df_out[turn_mask].index, -df_out[turn_mask][self.depth_col], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_TURNING, alpha=0.6, label="Turning")
            ax1.plot(df_out[up_mask].index, -df_out[up_mask][self.depth_col], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_UP, alpha=0.6, label="Ascending (+1)")
            ax1.plot(df_out[down_mask].index, -df_out[down_mask][self.depth_col], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_DOWN, alpha=0.6, label="Descending (-1)")
            
            ax1.set_ylabel(self.depth_col)
            ax1.set_title("Profile Classification")
            ax1.legend(loc="upper right", markerscale=5)

            ax2.plot(df_smooth.index, df_smooth["SMOOTH_VELOCITY"], color=COLOUR_VELOCITY, lw=LINE_WIDTH, label="Smoothed Velocity")
            ax2.axhline(self.gradient_thresholds[0], color=COLOUR_TURNING, lw=0.8, ls="--", alpha=0.5)
            ax2.axhline(self.gradient_thresholds[1], color=COLOUR_TURNING, lw=0.8, ls="--", alpha=0.5)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set_ylabel("Velocity")
            ax2.legend(loc="upper right")

            ax3.plot(df_out.index, df_out["PROFILE_ID"], color="gray", marker=".", ls="", ms=MARKER_SIZE)
            ax3.set_ylabel("Profile ID")
            ax3.set_xlabel("Time")

            plt.tight_layout()
            plt.show(block=False)

        root = tk.Tk()
        root.title("Parameter Adjustment")
        
        entries = {}

        tk.Label(root, text="Depth Column").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ent_depth = tk.Entry(root, width=12)
        ent_depth.insert(0, self.depth_col)
        ent_depth.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        entries["depth_column"] = ent_depth

        tk.Label(root, text="Cadence").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        ent_cadence = tk.Entry(root, width=8)
        ent_cadence.insert(0, self.cadence)
        ent_cadence.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        entries["resample_cadence"] = ent_cadence

        tk.Label(root, text="Gradient Thresholds").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ent_grad_pos = tk.Entry(root, width=6)
        ent_grad_pos.insert(0, str(self.gradient_thresholds[0]))
        ent_grad_pos.grid(row=1, column=1, sticky="w", padx=5)
        entries["grad_pos"] = ent_grad_pos
        
        ent_grad_neg = tk.Entry(root, width=6)
        ent_grad_neg.insert(0, str(self.gradient_thresholds[1]))
        ent_grad_neg.grid(row=1, column=2, sticky="w", padx=5)
        entries["grad_neg"] = ent_grad_neg

        tk.Label(root, text="Filter Window Sizes").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ent_win_med = tk.Entry(root, width=6)
        ent_win_med.insert(0, str(self.filter_win_sizes[0]))
        ent_win_med.grid(row=2, column=1, sticky="w", padx=5)
        entries["win_med"] = ent_win_med

        ent_win_mean = tk.Entry(root, width=6)
        ent_win_mean.insert(0, str(self.filter_win_sizes[1]))
        ent_win_mean.grid(row=2, column=2, sticky="w", padx=5)
        entries["win_mean"] = ent_win_mean

        tk.Label(root, text="Edge Squeeze").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        ent_squeeze = tk.Entry(root, width=6)
        ent_squeeze.insert(0, str(self.edge_squeeze))
        ent_squeeze.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        entries["edge_squeeze"] = ent_squeeze

        tk.Label(root, text="Dive Scale").grid(row=3, column=2, sticky="e", padx=5, pady=2)
        ent_scale = tk.Entry(root, width=6)
        ent_scale.insert(0, str(self.dive_scale))
        ent_scale.grid(row=3, column=3, sticky="w", padx=5, pady=2)
        entries["dive_scale"] = ent_scale

        def on_cancel():
            plt.close('all')
            root.quit()
            root.destroy()

        def on_regenerate():
            self.depth_col = entries["depth_column"].get()
            self.cadence = entries["resample_cadence"].get()
            self.gradient_thresholds = [float(entries["grad_pos"].get()), float(entries["grad_neg"].get())]
            
            med_val = entries["win_med"].get()
            mean_val = entries["win_mean"].get()
            self.filter_win_sizes = [
                med_val if not med_val.isdigit() else int(med_val), 
                mean_val if not mean_val.isdigit() else int(mean_val)
            ]
            
            self.edge_squeeze = int(entries["edge_squeeze"].get())
            self.dive_scale = float(entries["dive_scale"].get())
            
            plt.close('all')
            generate_plot()

        def on_save():
            self.update_parameters(
                depth_column=self.depth_col,
                resample_cadence=self.cadence,
                gradient_thresholds=self.gradient_thresholds,
                filter_window_sizes=self.filter_win_sizes,
                edge_squeeze=self.edge_squeeze,
                dive_scale=self.dive_scale
            )
            plt.close('all')
            root.quit()
            root.destroy()

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=4, column=0, columnspan=4, pady=15)

        tk.Button(btn_frame, text="Regenerate", command=on_regenerate).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save", command=on_save).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)

        generate_plot()
        return root