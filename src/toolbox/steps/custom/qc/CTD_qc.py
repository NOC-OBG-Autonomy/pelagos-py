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

"""QC step that flags CTD fill values, corrects CNDC S/m to mS/cm, and cross-flags statistical outliers.

This QC could become redundant if all incoming data was fixed at source"""

#### Mandatory imports ####
import numpy as np
import xarray as xr
from toolbox.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib
import matplotlib.pyplot as plt

# Diagnostic plot settings
PLOT_SIZE = (10, 8)
COLOR_RAW = "#b2bec3"        # Light grey for original data
COLOR_CORRECTED = "#0984e3"  # Blue for scaled/valid data
COLOR_FLAGGED = "#d63031"    # Red for flagged zeros (9)
COLOR_OUTLIER = "#e17055"    # Orange for statistical outliers (4)
MARKER_SIZE = 1.5
PLOT_ALPHA = 0.7
SIGMA_THRESHOLD = 5.0        # Number of standard deviations for anomaly detection


@register_qc
class ctd_qc(BaseQC):
    """
    Target Variable: PRES, TEMP, CNDC
    Flag Number: 9 (missing/fill value), 4 (gross outlier)
    Variables Flagged: PRES, TEMP, CNDC
    
    Checks for fill values of exactly 0.000 across the CTD variables 
    and flags them as 9. Evaluates valid CNDC data, applies unit scaling 
    if auto_scale is True (S/m to mS/cm), and identifies statistical outliers 
    (> 5 std deviations from the mean). Flags these outliers as 4 across 
    all three CTD variables simultaneously.
    """

    qc_name = "ctd qc"
    expected_parameters = {"auto_scale": bool}
    required_variables = ["PRES", "TEMP", "CNDC"]
    qc_outputs = ["PRES_QC", "TEMP_QC", "CNDC_QC"]

    def __init__(self, data, **kwargs):
        if data is not None:
            self.data = data
            # Store pure copies of the raw values strictly for the diagnostic plot
            self._raw_data = {
                "PRES": data["PRES"].values.copy(),
                "TEMP": data["TEMP"].values.copy(),
                "CNDC": data["CNDC"].values.copy()
            }
            
        # Default to True for standard ocean deployments
        self.auto_scale = kwargs.get("auto_scale", True)
        self.scaled = False

    def return_qc(self):
        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})
        qc_arrays = {}

        # Step 1: Handle Zeros and Unit Scaling
        for var in self.required_variables:
            vals = self.data[var].values
            qc = xr.zeros_like(self.data[var], dtype=int)

            # Flag exact 0.000 values as 9
            zero_mask = (vals == 0.0)
            qc = xr.where(zero_mask, 9, qc)

            # Safely check for unit scaling (CNDC only)
            if var == "CNDC":
                valid_mask = ~zero_mask & ~np.isnan(vals)
                
                if self.auto_scale and np.any(valid_mask):
                    max_val = np.max(vals[valid_mask])
                    current_units = str(self.data[var].attrs.get("units", "")).strip().lower()
                    
                    # Trust the metadata if it explicitly states it is already in mS/cm
                    already_mscm = current_units in ["ms/cm", "ms cm-1", "millisiemens/cm", "milli-siemens/cm"]

                    if not already_mscm and max_val < 10.0:
                        self.scaled = True
                        print("      [ctd qc] Converting CNDC from S/m to mS/cm for GSW calculations...")
                        
                        # Multiply valid data by 10 in place
                        vals[valid_mask] = vals[valid_mask] * 10.0
                        self.data[var].values = vals
                        
                        # Update attributes to reflect the change
                        self.data[var].attrs["units"] = "mS/cm"

            qc_arrays[var] = qc

        # Step 2: Cross-flagging 5-Sigma CNDC Outliers
        cndc_vals = self.data["CNDC"].values
        # Only evaluate points that are not NaN and not already flagged as 9 (zero)
        cndc_valid_mask = (qc_arrays["CNDC"] != 9) & ~np.isnan(cndc_vals)

        if np.any(cndc_valid_mask):
            cndc_mean = np.nanmean(cndc_vals[cndc_valid_mask])
            cndc_std = np.nanstd(cndc_vals[cndc_valid_mask])
            
            # Prevent dividing by zero or flagging noise if std is exactly 0
            if cndc_std > 0:
                # Create mask for values beyond the sigma threshold
                outlier_mask = np.abs(cndc_vals - cndc_mean) > (SIGMA_THRESHOLD * cndc_std)
                # Ensure we only flag points that were valid to begin with
                outlier_mask = outlier_mask & cndc_valid_mask
                
                outlier_count = np.sum(outlier_mask)
                if outlier_count > 0:
                    print(f"      [ctd qc] Found {outlier_count} CNDC anomalies (>{SIGMA_THRESHOLD} std). Cross-flagging triad as bad (4).")
                    
                    # Propagate the outlier flag (4) to PRES, TEMP, and CNDC
                    for var in self.required_variables:
                        qc_arrays[var] = xr.where(outlier_mask & (qc_arrays[var] == 0), 4, qc_arrays[var])

        # Step 3: Write to output
        for var in self.required_variables:
            self.flags[f"{var}_QC"] = qc_arrays[var]

        return self.flags

    def plot_diagnostics(self):
        if "TIME" not in self.data:
            return

        matplotlib.use("tkagg")
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=PLOT_SIZE, dpi=150)
        
        time_data = self.data["TIME"].values

        for ax, var in zip(axes, self.required_variables):
            raw_vals = self._raw_data[var]
            valid_vals = self.data[var].values
            qc_vals = self.flags[f"{var}_QC"].values

            # Plot raw data as a background trace if scaled (CNDC only)
            if var == "CNDC" and self.scaled:
                ax.plot(
                    time_data,
                    raw_vals,
                    marker="o",
                    ls="",
                    color=COLOR_RAW,
                    markersize=MARKER_SIZE,
                    alpha=PLOT_ALPHA,
                    label="Raw (S/m)"
                )

            # Plot the valid, active dataset
            label_str = f"Valid {var}"
            if var == "CNDC" and self.scaled:
                label_str += " (mS/cm)"

            ax.plot(
                time_data,
                valid_vals,
                marker="o",
                ls="",
                color=COLOR_CORRECTED,
                markersize=MARKER_SIZE,
                alpha=PLOT_ALPHA,
                label=label_str
            )

            # Overlay the flagged 5-sigma outliers (Flag 4)
            outlier_mask = (qc_vals == 4)
            if np.any(outlier_mask):
                ax.plot(
                    time_data[outlier_mask],
                    raw_vals[outlier_mask] if var == "CNDC" and self.scaled else valid_vals[outlier_mask],
                    marker="d",  # Diamond marker to distinguish from zeros
                    ls="",
                    color=COLOR_OUTLIER,
                    markersize=MARKER_SIZE + 2.0,
                    label=f"Outliers (>{SIGMA_THRESHOLD}σ)"
                )

            # Overlay the flagged zero values (Flag 9)
            zero_mask = (qc_vals == 9)
            if np.any(zero_mask):
                ax.plot(
                    time_data[zero_mask],
                    raw_vals[zero_mask],
                    marker="x",
                    ls="",
                    color=COLOR_FLAGGED,
                    markersize=MARKER_SIZE + 1.5,
                    label="Flagged Zeros (9)"
                )

            ax.set_ylabel(var, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=8)
            
            # Invert y-axis for pressure so the ocean surface is at the top of the plot
            if var == "PRES":
                ax.invert_yaxis()
            
            # Place legend to the right to avoid overlapping data
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                framealpha=0.9,
                fancybox=True
            )

        # Configure plot title based on actions taken
        title = "CTD Zero Flagging, Unit Verification & Anomaly Detection"
        if self.scaled:
            title += "\n(CNDC Magnitude Shifted: x10 to mS/cm)"
        elif not self.auto_scale:
            title += "\n(CNDC Auto-scale Disabled)"

        fig.suptitle(title, fontsize=10, fontweight="bold")
        axes[-1].set_xlabel("Time", fontsize=8)

        fig.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show(block=True)