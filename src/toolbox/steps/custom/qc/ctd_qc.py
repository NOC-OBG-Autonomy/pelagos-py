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

"""QC step that flags CTD fill values, corrects CNDC S/m to mS/cm, and cross-flags statistical outliers."""

from toolbox.steps.base_qc import BaseQC, register_qc
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt


@register_qc
class ctd_qc(BaseQC):
    """
    Target Variable: PRES, TEMP, CNDC
    Flag Number: 9 (missing/fill value), 4 (gross outlier)
    Variables Flagged: PRES, TEMP, CNDC
    """

    qc_name = "ctd qc"
    required_variables = ["PRES", "TEMP", "CNDC"]
    qc_outputs = ["PRES_QC", "TEMP_QC", "CNDC_QC"]

    parameter_schema = {
        "auto_scale": {
            "type": bool,
            "default": True,
            "description": "Automatically scale CNDC from S/m to mS/cm if max < 10.0"
        }
    }

    def __init__(self, data, **kwargs):
        self.expected_parameters = {k: v["type"] for k, v in self.parameter_schema.items()}
        super().__init__(data, **kwargs)
        
        if data is not None:
            self.data = data
            self._raw_data = {
                "PRES": data["PRES"].values.copy(),
                "TEMP": data["TEMP"].values.copy(),
                "CNDC": data["CNDC"].values.copy()
            }
            
        self.auto_scale = kwargs.get("auto_scale", self.parameter_schema["auto_scale"]["default"])
        self.scaled = False

    def return_qc(self):
        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})
        qc_arrays = {}

        for var in self.required_variables:
            vals = self.data[var].values
            qc = xr.zeros_like(self.data[var], dtype=int)

            zero_mask = (vals == 0.0)
            qc = xr.where(zero_mask, 9, qc)

            if var == "CNDC":
                valid_mask = ~zero_mask & ~np.isnan(vals)
                
                if self.auto_scale and np.any(valid_mask):
                    max_val = np.max(vals[valid_mask])
                    current_units = str(self.data[var].attrs.get("units", "")).strip().lower()
                    already_mscm = current_units in ["ms/cm", "ms cm-1", "millisiemens/cm", "milli-siemens/cm"]

                    if not already_mscm and max_val < 10.0:
                        self.scaled = True
                        self.log("Converting CNDC from S/m to mS/cm for GSW calculations...")
                        
                        vals[valid_mask] = vals[valid_mask] * 10.0
                        self.data[var].values = vals
                        self.data[var].attrs["units"] = "mS/cm"

            qc_arrays[var] = qc

        cndc_vals = self.data["CNDC"].values
        cndc_valid_mask = (qc_arrays["CNDC"] != 9) & ~np.isnan(cndc_vals)

        if np.any(cndc_valid_mask):
            cndc_mean = np.nanmean(cndc_vals[cndc_valid_mask])
            cndc_std = np.nanstd(cndc_vals[cndc_valid_mask])
            
            if cndc_std > 0:
                outlier_mask = np.abs(cndc_vals - cndc_mean) > (5.0 * cndc_std)
                outlier_mask = outlier_mask & cndc_valid_mask
                
                outlier_count = np.sum(outlier_mask)
                if outlier_count > 0:
                    self.log(f"Found {outlier_count} CNDC anomalies (>5.0 std). Cross-flagging triad as bad (4).")
                    
                    for var in self.required_variables:
                        qc_arrays[var] = xr.where(outlier_mask & (qc_arrays[var] == 0), 4, qc_arrays[var])

        for var in self.required_variables:
            self.flags[f"{var}_QC"] = qc_arrays[var]

        return self.flags

    def plot_diagnostics(self):
        if "TIME" not in self.data:
            return

        matplotlib.use("tkagg")
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8), dpi=150)
        
        time_data = self.data["TIME"].values

        for ax, var in zip(axes, self.required_variables):
            raw_vals = self._raw_data[var]
            valid_vals = self.data[var].values
            qc_vals = self.flags[f"{var}_QC"].values

            if var == "CNDC" and self.scaled:
                ax.plot(
                    time_data, raw_vals, marker="o", ls="", color="#b2bec3",
                    markersize=1.5, alpha=0.7, label="Raw (S/m)"
                )

            label_str = f"Valid {var} (mS/cm)" if var == "CNDC" and self.scaled else f"Valid {var}"

            ax.plot(
                time_data, valid_vals, marker="o", ls="", color="#0984e3",
                markersize=1.5, alpha=0.7, label=label_str
            )

            outlier_mask = (qc_vals == 4)
            if np.any(outlier_mask):
                ax.plot(
                    time_data[outlier_mask], 
                    raw_vals[outlier_mask] if var == "CNDC" and self.scaled else valid_vals[outlier_mask],
                    marker="d", ls="", color="#e17055", markersize=3.5, label="Outliers (>5.0σ)"
                )

            zero_mask = (qc_vals == 9)
            if np.any(zero_mask):
                ax.plot(
                    time_data[zero_mask], raw_vals[zero_mask], marker="x",
                    ls="", color="#d63031", markersize=3.0, label="Flagged Zeros (9)"
                )

            ax.set_ylabel(var, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=8)
            
            if var == "PRES":
                ax.invert_yaxis()
            
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9, fancybox=True)

        title = "CTD Zero Flagging, Unit Verification & Anomaly Detection"
        if self.scaled:
            title += "\n(CNDC Magnitude Shifted: x10 to mS/cm)"
        elif not self.auto_scale:
            title += "\n(CNDC Auto-scale Disabled)"

        fig.suptitle(title, fontsize=10, fontweight="bold")
        axes[-1].set_xlabel("Time", fontsize=8)

        fig.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show(block=True)