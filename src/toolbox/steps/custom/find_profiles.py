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

"""Class definition for finding vertical and horizontal profiles in depth data."""

from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import tkinter as tk
from scipy.signal import savgol_filter


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

def find_profiles(df_sorted, cadence, filter_win_sizes, gradient_thresholds, horiz_grad_thresh, dive_scale, min_horizontal_duration, surfacing_depth, inflection_accel_threshold, depth_col, has_water_vel):
    """
    Identifies and classifies vertical and horizontal profiles from depth-time data.
    Also derives continuous cycle numbers and scientific phase flags.
    """
    df = df_sorted[depth_col].resample(cadence).mean().to_frame()
    df[depth_col] = df[depth_col].interpolate(method='linear')

    if len(df) < 5:
        df_out = df_sorted.copy()
        df_out["PROFILE_ID"] = np.nan
        df_out["DIRECTION"] = np.nan
        df_out["GRADIENT"] = np.nan
        df_out["CYCLE"] = 1
        df_out["SCI_PHASE"] = 0
        
        df["SMOOTH_DEPTH"] = df[depth_col]
        df["SMOOTH_VELOCITY"] = 0.0
        df["SMOOTH_VELOCITY_HORIZ"] = 0.0
        df["STATE"] = "turning"
        df["ACCEL"] = 0.0
        return df_out, df

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
    
    df["SMOOTH_VELOCITY"] = savgol_filter(df["RAW_VEL"], 5, 2)
    df["SMOOTH_VELOCITY_HORIZ"] = savgol_filter(df["RAW_VEL"], 3, 2)
    df["ACCEL"] = np.gradient(df["SMOOTH_VELOCITY"]) / dt
    
    vel_crosses_zero = (df["SMOOTH_VELOCITY"] * df["SMOOTH_VELOCITY"].shift(1)) < 0
    pos_grad, neg_grad = gradient_thresholds

    df["STATE"] = "turning"
    df.loc[df["SMOOTH_VELOCITY"] > pos_grad, "STATE"] = "down"
    df.loc[df["SMOOTH_VELOCITY"] < neg_grad, "STATE"] = "up"
    df.loc[(df["SMOOTH_VELOCITY_HORIZ"].abs() <= horiz_grad_thresh) & (df["SMOOTH_DEPTH"] > surfacing_depth), "STATE"] = "horizontal"
    df.loc[(df["SMOOTH_DEPTH"] < -0.5) | vel_crosses_zero, "STATE"] = "turning"
    
    df["is_turning"] = (
        ((df["SMOOTH_VELOCITY"] >= neg_grad) & (df["SMOOTH_VELOCITY"] <= pos_grad)) | 
        (df["SMOOTH_DEPTH"] < -0.5) |
        vel_crosses_zero |
        df["SMOOTH_DEPTH"].isna()
    ).fillna(True).astype(bool)

    is_profile = ~df["is_turning"]
    profile_starts = is_profile & ~is_profile.shift(1, fill_value=False)
    df["PROFILE_ID"] = profile_starts.cumsum()
    df.loc[df["is_turning"], "PROFILE_ID"] = np.nan

    surf_mask = df["SMOOTH_DEPTH"] <= surfacing_depth
    down_mask = df["STATE"] == "down"
    
    state_subset = df.loc[surf_mask | down_mask]
    is_new_cycle = (state_subset["STATE"] == "down") & (surf_mask.loc[state_subset.index].shift(1) == True)
    
    cycle_trigger = pd.Series(0, index=df.index)
    cycle_trigger.loc[state_subset[is_new_cycle].index] = 1
    df["CYCLE"] = cycle_trigger.cumsum() + 1

    df_features = df[["PROFILE_ID", "is_turning", "SMOOTH_VELOCITY", "SMOOTH_VELOCITY_HORIZ", "SMOOTH_DEPTH", "STATE", "CYCLE", "ACCEL"]]
    
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
    df_out["is_failed_profile"] = False
    
    valid_pid_counter = 1
    
    for pid, group in df_out.dropna(subset=["PROFILE_ID"]).groupby("PROFILE_ID"):
        depth_span = group[depth_col].max() - group[depth_col].min()
        point_count = len(group)
        
        if depth_span >= dive_scale and point_count >= 2:
            df_out.loc[group.index, "VALID_PROFILE"] = valid_pid_counter
            x = (group.index - group.index[0]).total_seconds().values
            
            if len(x) > 1:
                m, _ = np.polyfit(x, group[depth_col].values, 1)
                df_out.loc[group.index, "GRADIENT"] = m
                df_out.loc[group.index, "DIRECTION"] = 1 if m < 0 else -1
                
            valid_pid_counter += 1
        else:
            df_out.loc[group.index, "is_failed_profile"] = True
            df_out.loc[group.index, "is_turning"] = False

    unassigned_mask = df_out["VALID_PROFILE"].isna()
    df_out["is_horiz_candidate"] = False
    df_out.loc[unassigned_mask, "is_horiz_candidate"] = (
        (df_out.loc[unassigned_mask, "SMOOTH_VELOCITY_HORIZ"].abs() <= horiz_grad_thresh) & 
        (df_out.loc[unassigned_mask, "SMOOTH_DEPTH"] > surfacing_depth)
    )

    horiz_groups = (~df_out["is_horiz_candidate"]).cumsum()
    duration_threshold = pd.Timedelta(min_horizontal_duration)

    for sub_id, sub_group in df_out[df_out["is_horiz_candidate"]].groupby(horiz_groups):
        if len(sub_group) < 2:
            df_out.loc[sub_group.index, "is_failed_profile"] = True
            df_out.loc[sub_group.index, "is_turning"] = False
            continue
            
        time_span = sub_group.index[-1] - sub_group.index[0]
        
        if time_span >= duration_threshold:
            df_out.loc[sub_group.index, "VALID_PROFILE"] = valid_pid_counter
            x = (sub_group.index - sub_group.index[0]).total_seconds().values
            
            if len(x) > 1:
                m, _ = np.polyfit(x, sub_group[depth_col].values, 1)
                df_out.loc[sub_group.index, "GRADIENT"] = m
            else:
                df_out.loc[sub_group.index, "GRADIENT"] = 0.0
                
            df_out.loc[sub_group.index, "DIRECTION"] = 0
            df_out.loc[sub_group.index, "is_turning"] = False
            valid_pid_counter += 1
        else:
            df_out.loc[sub_group.index, "is_failed_profile"] = True
            df_out.loc[sub_group.index, "is_turning"] = False

    valid_mask = df_out["VALID_PROFILE"].notna()
    profile_transitions = valid_mask & (df_out["VALID_PROFILE"] != df_out["VALID_PROFILE"].shift(1))
    
    df_out["CHRONO_ID"] = profile_transitions.cumsum()
    df_out.loc[~valid_mask, "CHRONO_ID"] = np.nan
    
    df_out = df_out.drop(columns=["PROFILE_ID", "is_horiz_candidate", "VALID_PROFILE"])
    df_out = df_out.rename(columns={"CHRONO_ID": "PROFILE_ID"})

    df_out["SCI_PHASE"] = 0 
    
    surfacing_mask = df_out["SMOOTH_DEPTH"] <= surfacing_depth
    df_out.loc[surfacing_mask, "SCI_PHASE"] = 3
    
    df_out.loc[(df_out["DIRECTION"] == 1) & (df_out["SCI_PHASE"] == 0), "SCI_PHASE"] = 1
    df_out.loc[(df_out["DIRECTION"] == -1) & (df_out["SCI_PHASE"] == 0), "SCI_PHASE"] = 2
    
    horiz_pids = df_out.loc[(df_out["DIRECTION"] == 0) & (df_out["SCI_PHASE"] == 0), "PROFILE_ID"].dropna().unique()
    for pid in horiz_pids:
        mask = (df_out["PROFILE_ID"] == pid) & (df_out["SCI_PHASE"] == 0)
        segment = df_out[mask]
        
        if segment.empty:
            continue
            
        has_enough_vel = False
        if has_water_vel and "WATER_VELOC_FINAL_U" in segment.columns and "WATER_VELOC_FINAL_V" in segment.columns:
            if segment["WATER_VELOC_FINAL_U"].count() >= 5 and segment["WATER_VELOC_FINAL_V"].count() >= 5:
                has_enough_vel = True
                
        if has_enough_vel:
            df_out.loc[mask, "SCI_PHASE"] = 4
        else:
            duration = segment.index[-1] - segment.index[0]
            if duration > pd.Timedelta("10min"):
                df_out.loc[mask, "SCI_PHASE"] = 6
            else:
                df_out.loc[mask, "SCI_PHASE"] = 7
                
    failed_mask = df_out["is_failed_profile"] & (df_out["SCI_PHASE"] == 0)
    df_out.loc[failed_mask, "SCI_PHASE"] = 7
    
    turning_mask = df_out["is_turning"] & (df_out["SCI_PHASE"] == 0)
    df_out.loc[turning_mask, "SCI_PHASE"] = 5 
    
    # Overwrite transition with inflection if velocity is within turning thresholds
    vel_mask = (df_out["SCI_PHASE"] == 7) & (df_out["SMOOTH_VELOCITY"] >= neg_grad) & (df_out["SMOOTH_VELOCITY"] <= pos_grad)
    df_out.loc[vel_mask, "SCI_PHASE"] = 5

    # Overwrite transition with inflection if acceleration is extreme
    accel_mask = (df_out["SCI_PHASE"] == 7) & (df_out["ACCEL"].abs() > inflection_accel_threshold)
    df_out.loc[accel_mask, "SCI_PHASE"] = 5

    return df_out, df


@register_step
class FindProfilesStep(BaseStep, QCHandlingMixin):
    """
    Identifies and classifies vertical and horizontal profiles from depth-time data.
    Derives continuous cycle numbers and assigns scientific phase flags.

    Parameters
    ----------
    depth_column : str, optional
        Name of the dataset variable to use for vertical depth calculations. Defaults to "PRES".
    resample_cadence : str, optional
        Time string for regularising the data via interpolation prior to calculation.
    gradient_thresholds : list, optional
        List of [positive, negative] gradient thresholds for identifying descending/ascending motion.
    horiz_gradient_threshold : float, optional
        Threshold for gradient variance to qualify a phase as horizontal.
    filter_window_sizes : list, optional
        List of sizes [median, mean] for rolling windows to smooth profile gradients.
    dive_scale : float, optional
        Minimum vertical depth required to qualify as a valid vertical dive.
    min_horizontal_duration : str, optional
        Minimum time required at a fixed depth to classify as a valid horizontal phase.
    surfacing_depth : float, optional
        Maximum operational depth boundary indicating the platform is surfaced.
    inflection_accel_threshold : float, optional
        Acceleration threshold beyond which a transition phase becomes inflection.
    """
    
    step_name = "Find Profiles"
    required_variables = ["TIME"]
    provided_variables = ["PROFILE_NUMBER", "CYCLE", "SCI_PHASE"]

    parameter_schema = {
        "depth_column": {
            "type": str,
            "default": "PRES",
            "description": "Depth or pressure column name. Defaults to PRES."
        },
        "resample_cadence": {
            "type": str,
            "default": "30s",
            "description": "Time cadence to resample for feature extraction."
        },
        "gradient_thresholds": {
            "type": list,
            "default": [0.033, -0.033],
            "description": "Positive and negative velocity thresholds."
        },
        "horiz_gradient_threshold": {
            "type": float,
            "default": 0.01,
            "description": "Velocity threshold for horizontal phase."
        },
        "filter_window_sizes": {
            "type": list,
            "default": [1, 2],
            "description": "Window sizes for median and mean smoothing."
        },
        "dive_scale": {
            "type": float,
            "default": 15.0,
            "description": "Minimum depth span to be considered a profile."
        },
        "min_horizontal_duration": {
            "type": str,
            "default": "20min",
            "description": "Minimum continuous duration to be classed as horizontal."
        },
        "surfacing_depth": {
            "type": float,
            "default": 0.8,
            "description": "Maximum depth indicating the platform is surfaced."
        },
        "inflection_accel_threshold": {
            "type": float,
            "default": 0.002,
            "description": "Acceleration threshold beyond which a transition phase becomes inflection."
        }
    }

    def run(self):
        self.log("Attempting to designate profile numbers, cycles, directions, and phases")
        self.check_data()
        self.filter_qc()

        self.depth_col = getattr(self, "depth_column", "PRES")
        if self.depth_col not in self.data.variables:
            raise ValueError(f"Specified depth column '{self.depth_col}' not found in the dataset.")

        self.cadence = getattr(self, "resample_cadence", "30s")
        self.gradient_thresholds = getattr(self, "gradient_thresholds", [0.033, -0.033])
        self.horiz_grad_thresh = getattr(self, "horiz_gradient_threshold", 0.01)
        self.filter_win_sizes = getattr(self, "filter_window_sizes", [1, 2])
        self.dive_scale = getattr(self, "dive_scale", 15.0)
        self.min_horizontal_duration = getattr(self, "min_horizontal_duration", "20min")
        self.surfacing_depth = getattr(self, "surfacing_depth", 0.8)
        self.inflection_accel_threshold = getattr(self, "inflection_accel_threshold", 0.002)

        self.has_water_vel = ("WATER_VELOC_FINAL_U" in self.data.variables and 
                              "WATER_VELOC_FINAL_V" in self.data.variables)
        
        if not self.has_water_vel:
            self.log("Warning: WATER_VELOC_FINAL_U and/or WATER_VELOC_FINAL_V not found. Parking will default to Propelled or Transition.")

        if self.diagnostics:
            root = self.generate_diagnostics()
            root.mainloop()

        cols_to_extract = ["TIME", self.depth_col]
        if self.has_water_vel:
            cols_to_extract.extend(["WATER_VELOC_FINAL_U", "WATER_VELOC_FINAL_V"])

        df_raw = self.data[cols_to_extract].to_dataframe().reset_index()
        df_sorted = df_raw.dropna(subset=[self.depth_col, "TIME"]).sort_values("TIME").set_index("TIME")

        df_out, _ = find_profiles(
            df_sorted, self.cadence, self.filter_win_sizes, 
            self.gradient_thresholds, self.horiz_grad_thresh, self.dive_scale, 
            self.min_horizontal_duration, self.surfacing_depth, self.inflection_accel_threshold, 
            self.depth_col, self.has_water_vel
        )

        df_out = df_out.reset_index()
        df_final = df_raw.merge(
            df_out[["N_MEASUREMENTS", "PROFILE_ID", "DIRECTION", "GRADIENT", "CYCLE", "SCI_PHASE"]], 
            on="N_MEASUREMENTS", 
            how="left"
        )
        
        df_final["SCI_PHASE"] = df_final["SCI_PHASE"].fillna(0).astype(int)

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), df_final["PROFILE_ID"].to_numpy())
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. NaN indicates no profile.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["PROFILE_DIRECTION"] = (("N_MEASUREMENTS",), df_final["DIRECTION"].to_numpy())
        self.data.PROFILE_DIRECTION.attrs = {
            "long_name": "Profile Direction (-1: Descending, 0: Horizontal, 1: Ascending, NaN: Not Profile)",
            "units": "None",
        }

        self.data["PROFILE_GRADIENT"] = (("N_MEASUREMENTS",), df_final["GRADIENT"].to_numpy())
        self.data.PROFILE_GRADIENT.attrs = {
            "long_name": "Profile Vertical Gradient",
            "units": "m/s",
        }

        self.data["CYCLE"] = (("N_MEASUREMENTS",), df_final["CYCLE"].to_numpy())
        self.data.CYCLE.attrs = {
            "long_name": "Continuous cycle number derived from surfacing points",
            "units": "None",
            "standard_name": "Cycle Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["SCI_PHASE"] = (("N_MEASUREMENTS",), df_final["SCI_PHASE"].to_numpy())
        self.data.SCI_PHASE.attrs = {
            "long_name": "Scientific Phase Classification",
            "units": "None",
            "valid_min": 0,
            "valid_max": 7,
            "flag_values": "0, 1, 2, 3, 4, 5, 6, 7",
            "flag_meanings": "unknown ascent descent surfacing parking inflection propelled transition"
        }

        self.generate_qc({
            "PROFILE_NUMBER_QC": ["TIME_QC", f"{self.depth_col}_QC"],
            "PROFILE_DIRECTION_QC": ["TIME_QC", f"{self.depth_col}_QC"],
            "PROFILE_GRADIENT_QC": ["TIME_QC", f"{self.depth_col}_QC"],
            "CYCLE_QC": ["TIME_QC", f"{self.depth_col}_QC"],
            "SCI_PHASE_QC": ["TIME_QC", f"{self.depth_col}_QC"]
        })

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        def generate_plot():
            mpl.use("TkAgg")

            cols_to_extract = ["TIME", self.depth_col]
            if self.has_water_vel:
                cols_to_extract.extend(["WATER_VELOC_FINAL_U", "WATER_VELOC_FINAL_V"])

            df_raw = self.data[cols_to_extract].to_dataframe().reset_index()
            df_sorted = df_raw.dropna(subset=[self.depth_col, "TIME"]).sort_values("TIME").set_index("TIME")
            
            df_out, df_smooth = find_profiles(
                df_sorted, self.cadence, self.filter_win_sizes, self.gradient_thresholds, self.horiz_grad_thresh, 
                self.dive_scale, self.min_horizontal_duration, self.surfacing_depth, self.inflection_accel_threshold, 
                self.depth_col, self.has_water_vel
            )

            fig_main, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})

            points = np.array([mdates.date2num(df_smooth.index), -df_smooth["SMOOTH_DEPTH"].values]).T.reshape(-1, 1, 2)
            c_map = {"up": "tab:blue", "down": "tab:green", "horizontal": "tab:purple", "turning": "tab:orange"}
            ax1.add_collection(LineCollection(np.concatenate([points[:-1], points[1:]], axis=1), colors=[c_map[state] for state in df_smooth["STATE"].iloc[:-1]], linewidths=1.5, zorder=0, alpha=0.7))

            phase_colours = {0: "tab:gray", 1: "tab:blue", 2: "tab:green", 3: "tab:cyan", 4: "tab:purple", 5: "tab:orange", 6: "tab:pink", 7: "tab:brown"}
            phase_names = {0: "Unknown", 1: "Ascent", 2: "Descent", 3: "Surfacing", 4: "Parking", 5: "Inflection", 6: "Propelled", 7: "Transition"}

            for p in sorted(df_out["SCI_PHASE"].dropna().unique()):
                mask = df_out["SCI_PHASE"] == p
                ax1.plot(df_out[mask].index, -df_out[mask][self.depth_col], marker=".", ls="", ms=4, color=phase_colours.get(p, "tab:gray"), zorder=3)

            ax1.legend([Line2D([0], [0], marker='.', color='w', markerfacecolor=phase_colours.get(p, "tab:gray"), markersize=8) for p in phase_names], [phase_names.get(p, "Unknown") for p in phase_names], loc="upper right")
            ax1.set(ylabel=self.depth_col, title="Scientific Phase Overlay")

            ax2.plot(df_smooth.index, df_smooth["SMOOTH_VELOCITY"], color="tab:red", lw=1.5, label="Smoothed Velocity (Vert)")
            for thresh in self.gradient_thresholds: ax2.axhline(thresh, color="tab:orange", lw=0.8, ls="--", alpha=0.5)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set(ylabel="Velocity")
            ax2.legend(loc="upper right")

            ax3.plot(df_out.index, df_out["PROFILE_ID"], color="gray", marker=".", ls="", ms=2, label="Profile ID")
            ax3.plot(df_out.index, df_out["CYCLE"], color="tab:red", marker=".", ls="", ms=2, label="Cycle Number")
            ax3.set(ylabel="ID / Cycle", xlabel="Time")
            ax3.legend(loc="upper left")

            fig_main.tight_layout()
            fig_main.show()

        root = tk.Tk()
        root.title("Parameter Adjustment")
        entries = {}

        ui_fields = [
            ("Cadence", "resample_cadence", self.cadence, 0, 0), 
            ("Vert Grad +", "grad_pos", self.gradient_thresholds[0], 0, 2), 
            ("Vert Grad -", "grad_neg", self.gradient_thresholds[1], 0, 4),
            ("Win Med", "win_med", self.filter_win_sizes[0], 1, 0), 
            ("Win Mean", "win_mean", self.filter_win_sizes[1], 1, 2), 
            ("Dive Scale", "dive_scale", self.dive_scale, 1, 4),
            ("Horiz Grad", "horiz_gradient_threshold", self.horiz_grad_thresh, 2, 0), 
            ("Horiz Dur.", "min_horizontal_duration", self.min_horizontal_duration, 2, 2),
            ("Surfacing Dep.", "surfacing_depth", self.surfacing_depth, 2, 4),
            ("Accel Thresh.", "inflection_accel_threshold", self.inflection_accel_threshold, 3, 0)
        ]

        for lbl, key, val, r, c in ui_fields:
            tk.Label(root, text=lbl).grid(row=r, column=c, sticky="e", padx=2, pady=1)
            ent = tk.Entry(root, width=8)
            ent.insert(0, str(val))
            ent.grid(row=r, column=c+1, sticky="w", padx=2, pady=1)
            entries[key] = ent

        root.bind("<Down>", lambda e: e.widget.tk_focusNext().focus() or "break")
        root.bind("<Up>", lambda e: e.widget.tk_focusPrev().focus() or "break")

        def close_all(event=None):
            plt.close('all')
            root.quit()
            root.destroy()

        def on_regen(event=None):
            self.cadence = entries["resample_cadence"].get()
            self.min_horizontal_duration = entries["min_horizontal_duration"].get()
            self.gradient_thresholds = [float(entries["grad_pos"].get()), float(entries["grad_neg"].get())]
            self.horiz_grad_thresh = float(entries["horiz_gradient_threshold"].get())
            self.dive_scale = float(entries["dive_scale"].get())
            self.surfacing_depth = float(entries["surfacing_depth"].get())
            self.inflection_accel_threshold = float(entries["inflection_accel_threshold"].get())
            
            w_med, w_mean = entries["win_med"].get(), entries["win_mean"].get()
            self.filter_win_sizes = [int(w_med) if w_med.isdigit() else w_med, int(w_mean) if w_mean.isdigit() else w_mean]
            
            plt.close('all')
            generate_plot()

        def on_save(event=None):
            self.update_parameters(
                resample_cadence=self.cadence, gradient_thresholds=self.gradient_thresholds, 
                horiz_gradient_threshold=self.horiz_grad_thresh, filter_window_sizes=self.filter_win_sizes,
                dive_scale=self.dive_scale, min_horizontal_duration=self.min_horizontal_duration,
                surfacing_depth=self.surfacing_depth, inflection_accel_threshold=self.inflection_accel_threshold
            )
            close_all()

        for key in ["<Return>", "<Control-s>", "<Command-s>"]: root.bind(key, on_save)
        root.bind("<Escape>", close_all)

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=4, column=0, columnspan=6, pady=10)

        for text, cmd in [("Regenerate", on_regen), ("Save", on_save), ("Cancel", close_all)]:
            tk.Button(btn_frame, text=text, command=cmd).pack(side="left", padx=5)

        generate_plot()
        return root