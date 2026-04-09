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
import polars as pl
import numpy as np
import gsw
import matplotlib
import matplotlib.pyplot as plt

# Diagnostic plot settings
PLOT_SIZE = (10, 8)  # Widened slightly to accommodate the external legend
PLOT_COLOURS = ["#00b894", "#0984e3", "#d63031", "#fdcb6e", "#6c5ce7", "#e84393", "#00cec9", "#e17055"]
FLAGGED_COLOUR = "#b2bec3" # Grey for flagged/bad data
MARKER_SIZE = 1
PLOT_ALPHA = 0.6


@register_step
class DeriveCTDVariables(BaseStep, QCHandlingMixin):
    """
    A processing step class for deriving oceanographic variables from CTD data.

    This class processes Conductivity, Temperature, and Depth (CTD) data to derive
    additional oceanographic variables such as salinity, density, and depth using
    the Gibbs SeaWater (GSW) Oceanographic Toolbox functions.

    Inherits from BaseStep and processes data stored in the context dictionary.

    Attributes:
        step_name (str): Identifier for this processing step ("Derive CTD")
    """

    step_name = "Derive CTD"
    required_variables = ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]
    provided_variables = ["DEPTH", "PRAC_SALINITY", "ABS_SALINITY", "CONS_TEMP", "DENSITY"]

    def run(self):
        """
        Execute the CTD variable derivation process. The following variables are
        required: ["TIME", "LATITUDE", "LONGITUDE", "CNDC", "PRES", "TEMP"]

        This method performs the following operations:
        1. Validates that data exists in the context
        2. Applies unit conversions to raw measurements
        3. Optionally interpolates missing position data
        4. Derives oceanographic variables using GSW functions
        5. Adds metadata to derived variables
        6. Updates the context with processed data

        Returns:
            dict: Updated context dictionary containing original and derived variables

        Raises:
            ValueError: If no data is found in the context
        """
        self.log("Processing CTD...")

        self.filter_qc()

        # Extract only the required variables that currently exist in the dataset
        available_vars = [col for col in self.required_variables if col in self.data]
        
        # nan_to_null is False to ensure underlying NaN structures are never altered
        df = pl.from_pandas(
            self.data[available_vars].to_dataframe(),
            nan_to_null=False,
        )

        gsw_function_calls = (
            ("DEPTH", gsw.z_from_p, ["PRES", "LATITUDE"]),
            ("PRAC_SALINITY", gsw.SP_from_C, ["CNDC", "TEMP", "PRES"]),
            (
                "ABS_SALINITY",
                gsw.SA_from_SP,
                ["PRAC_SALINITY", "PRES", "LONGITUDE", "LATITUDE"],
            ),
            ("CONS_TEMP", gsw.CT_from_t, ["ABS_SALINITY", "TEMP", "PRES"]),
            ("DENSITY", gsw.rho, ["ABS_SALINITY", "CONS_TEMP", "PRES"]),
        )

        variable_metadata = {
            "DEPTH": {
                "long_name": "Depth from surface (negative down as defined by TEOS-10)",
                "units": "m",
                "standard_name": "DEPTH",
                "valid_min": -10925,
                "valid_max": 1,
            },
            "PRAC_SALINITY": {
                "long_name": "Practical salinity",
                "units": "1",
                "standard_name": "PRAC_SALINITY",
                "valid_min": 2,
                "valid_max": 42,
            },
            "ABS_SALINITY": {
                "long_name": "Absolute salinity",
                "units": "g/kg",
                "standard_name": "ABS_SALINITY",
                "valid_min": 0,
                "valid_max": 1000,
            },
            "CONS_TEMP": {
                "long_name": "Conservative temperature",
                "units": "degC",
                "standard_name": "CONS_TEMP",
                "valid_min": -2,
                "valid_max": 102,
            },
            "DENSITY": {
                "long_name": "Density",
                "units": "kg/m3",
                "standard_name": "DENSITY",
                "valid_min": 900,
                "valid_max": 1100,
            },
        }

        for var_name, func, args in gsw_function_calls:
            if var_name not in self.to_derive:
                continue

            self.log(f"Deriving {var_name}...")

            # Validate that all required inputs exist for this specific calculation
            missing_args = [arg for arg in args if arg not in df.columns]
            if missing_args:
                self.log(f"Warning: Missing required variables {missing_args} for {var_name}. Skipping.")
                continue

            # Convert inputs back to pure numpy arrays for GSW
            input_arrays = [df[arg].to_numpy() for arg in args]
            derived_values = func(*input_arrays)

            df = df.with_columns(pl.Series(var_name, derived_values))

            self.data[var_name] = (("N_MEASUREMENTS",), derived_values)
            self.data[var_name].attrs = variable_metadata[var_name]

            # Safely generate QC by only passing source columns that actually exist
            source_qcs = [f"{arg}_QC" for arg in args if f"{arg}_QC" in self.data]
            if source_qcs:
                self.generate_qc({f"{var_name}_QC": source_qcs})

        if self.diagnostics:
            self.plot_diagnostics()

        self.reconstruct_data()
        self.update_qc()

        self.context["data"] = self.data
        return self.context

    def plot_diagnostics(self):
        if "TIME" not in self.data:
            return

        # Combine physical inputs and derived outputs, filtering for what actually exists
        target_variables = ["PRES", "CNDC", "TEMP"] + self.provided_variables
        plot_vars = [var for var in target_variables if var in self.data]
        
        if not plot_vars:
            return

        matplotlib.use("tkagg")
        n_vars = len(plot_vars)
        
        fig, axes = plt.subplots(n_vars, 1, sharex=True, figsize=PLOT_SIZE, dpi=150)
        
        if n_vars == 1:
            axes = [axes]

        time_data = self.data["TIME"].values

        for i, var_name in enumerate(plot_vars):
            ax = axes[i]
            colour = PLOT_COLOURS[i % len(PLOT_COLOURS)]
            data_vals = self.data[var_name].values
            
            # Extract units and format cleanly (ignore "1" or missing units)
            units = str(self.data[var_name].attrs.get("units", "")).strip()
            if units in ["1", "unknown", "None", ""]:
                unit_str = ""
            else:
                unit_str = f"\n[{units}]"

            # Determine QC status if the QC column exists
            qc_col = f"{var_name}_QC"
            if qc_col in self.data:
                qc_vals = self.data[qc_col].values
                # Treat 0 (No QC), 1, 2, 5, 8 as "Good" points
                good_mask = np.isin(qc_vals, [0, 1, 2, 5, 8])
                bad_mask = ~good_mask & ~np.isnan(data_vals)
                good_plot_mask = good_mask & ~np.isnan(data_vals)
                
                # Plot bad data first so it sits beneath good data
                if np.any(bad_mask):
                    ax.plot(
                        time_data[bad_mask],
                        data_vals[bad_mask],
                        ls="",
                        marker="o",
                        markersize=MARKER_SIZE,
                        alpha=PLOT_ALPHA,
                        c=FLAGGED_COLOUR,
                        zorder=1
                    )
                    
                # Plot good data on top
                if np.any(good_plot_mask):
                    ax.plot(
                        time_data[good_plot_mask],
                        data_vals[good_plot_mask],
                        ls="",
                        marker="o",
                        markersize=MARKER_SIZE,
                        alpha=PLOT_ALPHA,
                        c=colour,
                        zorder=2
                    )
                    
                # We calculate stats only on the good data for a cleaner representation
                stats_data = data_vals[good_plot_mask]
            else:
                # Fallback if no QC column exists
                ax.plot(
                    time_data,
                    data_vals,
                    ls="",
                    marker="o",
                    markersize=MARKER_SIZE,
                    alpha=PLOT_ALPHA,
                    c=colour,
                    zorder=2
                )
                stats_data = data_vals[~np.isnan(data_vals)]
            
            # Calculate robust statistics
            if len(stats_data) > 0:
                v_min = np.nanmin(stats_data)
                v_max = np.nanmax(stats_data)
                v_mean = np.nanmean(stats_data)
                v_std = np.nanstd(stats_data)
                
                # Add formatted statistical legend outside the plot area
                stat_text = f"Min: {v_min:.3f}\nMax: {v_max:.3f}\nMean: {v_mean:.3f}\nStd: {v_std:.3f}"
                ax.plot([], [], ls="", label=stat_text)
                ax.legend(
                    loc="center left", 
                    bbox_to_anchor=(1.01, 0.5), 
                    fontsize=6, 
                    framealpha=0.9, 
                    fancybox=True
                )

            ax.set_ylabel(f"{var_name}{unit_str}", fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=7)
            
            # Invert y-axis for pressure so the ocean surface is at the top of the plot
            if var_name == "PRES":
                ax.invert_yaxis()

        axes[-1].set_xlabel("Time", fontsize=8)
        fig.suptitle(f"{self.step_name} Diagnostics", fontsize=10, fontweight="bold")
        
        # Adjust layout to leave room on the right for the external legends
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        plt.show(block=True)