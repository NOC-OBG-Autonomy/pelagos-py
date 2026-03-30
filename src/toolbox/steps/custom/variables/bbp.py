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
from toolbox.utils.processing_utils import *
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import glidertools as gt


@register_step
class BBPFromBeta(BaseStep, QCHandlingMixin):

    step_name = "BBP from Beta"
    required_variables = ["TIME", "DEPTH", "TEMP", "PRAC_SALINITY"]
    provided_variables = []

    def run(self):
        """
        Example
        -------
        - name: "BBP from Beta"
          parameters:
            apply_to: "BETA_BACKSCATTERING700"
            output_as: "BBP700"
            theta: 124
            xfactor: 1.076
          diagnostics: false

        Returns
        -------

        """
        self.filter_qc()

        apply_to = getattr(self, "apply_to", None)
        output_as = getattr(self, "output_as", None)
        self.theta = getattr(self, "theta", 124)
        self.xfactor = getattr(self, "xfactor", 1.076)

        self.process_list = []
        if not apply_to:
            for var in self.data.data_vars:
                if str(var).startswith("BETA_BACKSCATTERING"):
                    suffix = str(var).replace("BETA_BACKSCATTERING", "")
                    self.process_list.append((str(var), f"BBP{suffix}"))
        else:
            self.process_list.append((apply_to, output_as))

        # Get the required variables and interpolate DEPTH, TEMP and PRAC_SALINITY
        self.data_subset = self.data[
            ["TIME",
             "PROFILE_NUMBER",
             "DEPTH",
             "TEMP",
             "PRAC_SALINITY"]
        ]

        for var in ["DEPTH", "TEMP", "PRAC_SALINITY"]:
            self.data_subset[var][:] = interpolate_nans(
                self.data_subset[var],
                self.data_subset["TIME"]
            )

        # Apply the correction for all identified variables
        for current_apply_to, current_output_as in self.process_list:
            wavelength = 700
            suffix = current_apply_to.replace("BETA_BACKSCATTERING", "")
            if suffix.isdigit():
                wavelength = int(suffix)

            bbp_corrected = gt.flo_functions.flo_bback_total(
                self.data[current_apply_to],
                self.data_subset["TEMP"],
                self.data_subset["PRAC_SALINITY"],
                self.theta,
                wavelength,
                self.xfactor)

            # Stitch back into the data
            self.data[current_output_as] = bbp_corrected

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        for current_apply_to, current_output_as in self.process_list:
            if current_apply_to != current_output_as:
                self.generate_qc({f"{current_output_as}_QC": [f"{current_apply_to}_QC"]})
            else:
                self.log_warn(f"'apply_to' and 'output_as' are the same. This will cause {current_apply_to} to be overwritten.")

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        # --- Configurable Plot Variables ---
        COLOUR_BETA = "indianred"
        COLOUR_BBP = "steelblue"
        PLOT_SIZE = (12, 6)
        MARKER_SIZE = 2
        
        mpl.use("tkagg")
        
        for current_apply_to, current_output_as in self.process_list:
            fig, (ax_box, ax_scatter) = plt.subplots(1, 2, figsize=PLOT_SIZE, dpi=150)
            
            # Clean both datasets for the boxplot to remove extreme outliers
            beta_clean = remove_outliers(self.data[current_apply_to])
            bbp_clean = remove_outliers(self.data[current_output_as])

            # Panel 1: Boxplots
            bplot = ax_box.boxplot(
                [beta_clean, bbp_clean],
                vert=True, patch_artist=True,
                labels=["Raw Beta", "Particulate BBP"]
            )
            
            bplot['boxes'][0].set_facecolor(COLOUR_BETA)
            bplot['boxes'][1].set_facecolor(COLOUR_BBP)
            
            ax_box.set_title(f"1. Data Distribution Comparison ({current_apply_to})")
            ax_box.set_ylabel("Value")
            ax_box.grid(True, linestyle="--", alpha=0.6)

            # Add statistical text box
            stats_text = (
                f"Raw Beta Mean: {np.nanmean(beta_clean):.6f}\n"
                f"Particulate BBP Mean: {np.nanmean(bbp_clean):.6f}\n"
                f"Scale Multiplier: ~{np.nanmean(bbp_clean)/np.nanmean(beta_clean):.1f}x"
            )
            ax_box.text(0.05, 0.95, stats_text, transform=ax_box.transAxes, 
                        va='top', bbox=dict(facecolor='white', alpha=0.8))

            # Panel 2: Converted BBP over Depth
            ax_scatter.plot(self.data[current_output_as], self.data["DEPTH"], 
                     ls="", marker=".", c=COLOUR_BBP, markersize=MARKER_SIZE, alpha=0.3)
            
            # Force 0m to the top of the axis
            depth_min = float(self.data["DEPTH"].min(skipna=True))
            ax_scatter.set_ylim(depth_min, 0)
            
            ax_scatter.set_xlabel(f"Particulate Backscatter ({current_output_as})")
            ax_scatter.set_ylabel("Depth (m)")
            ax_scatter.set_title("2. Converted Profile")
            ax_scatter.grid(True, alpha=0.3)
            
            fig.suptitle(f"Optical Backscatter Conversion Statistics: {current_apply_to} -> {current_output_as}")
            fig.tight_layout()
            plt.show(block=True)


@register_step
class IsolateBBPSpikes(BaseStep, QCHandlingMixin):

    step_name = "Isolate BBP Spikes"
    required_variables = ["TIME", "DEPTH"] 
    provided_variables = []

    def run(self):
        """
         Example
         -------
         - name: "Isolate BBP Spikes"
           parameters:
             apply_to: "BBP700"
             window_size: 50
             method: "median"
           diagnostics: false

         Returns
         -------

         """
        self.filter_qc()

        apply_to = getattr(self, "apply_to", None)
        self.window_size = getattr(self, "window_size", 50)
        self.method = getattr(self, "method", "median")

        self.process_list = []
        if not apply_to:
            for var in self.data.data_vars:
                if str(var).startswith("BBP") and not str(var).endswith(("QC", "BASELINE", "SPIKES")):
                    self.process_list.append(str(var))
        else:
            self.process_list.append(apply_to)

        self.baselines = {}
        self.spikes = {}

        for current_apply_to in self.process_list:
            baseline, spikes = gt.cleaning.despike(
                self.data[current_apply_to],
                self.window_size,
                spike_method=self.method
            )

            self.data[f"{current_apply_to}_BASELINE"] = baseline
            self.data[f"{current_apply_to}_SPIKES"] = spikes
            
            self.baselines[current_apply_to] = baseline
            self.spikes[current_apply_to] = spikes

        self.reconstruct_data()
        self.update_qc()

        # Generate QC if a new variable is added. Otherwise warn the user that input is being overwritten.
        for current_apply_to in self.process_list:
            self.generate_qc(
                {
                    f"{current_apply_to}_BASELINE_QC": [f"{current_apply_to}_QC"],
                    f"{current_apply_to}_SPIKES_QC": [f"{current_apply_to}_QC"],
                }
            )

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        # --- Configurable Plot Variables ---
        COLOUR_RAW = "lightgrey"
        COLOUR_BASELINE = "steelblue"
        COLOUR_SPIKES = "indianred"
        PLOT_SIZE = (12, 6)
        MARKER_SIZE = 3
        
        mpl.use("tkagg")
        
        time_vals = self.data["TIME"].values
        depth_vals = self.data["DEPTH"].values
        
        for current_apply_to in self.process_list:
            raw = self.data[current_apply_to].values
            baseline = self.baselines[current_apply_to]
            spikes = self.spikes[current_apply_to]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_SIZE, dpi=150)
            
            # Panel 1: Raw vs Baseline over Time
            ax1.plot(time_vals, raw, ls="", marker=".", c=COLOUR_RAW, markersize=MARKER_SIZE, label="Raw BBP")
            ax1.plot(time_vals, baseline, ls="", marker=".", c=COLOUR_BASELINE, markersize=MARKER_SIZE, alpha=0.7, label="Baseline (Background)")
            ax1.set_ylabel(current_apply_to)
            ax1.set_xlabel("Time")
            ax1.set_title("1. Baseline Extraction")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Spikes over Depth
            ax2.plot(spikes, depth_vals, ls="", marker=".", c=COLOUR_SPIKES, markersize=MARKER_SIZE, alpha=0.5, label="Isolated Spikes")
            
            # Add statistical text box
            valid_raw = np.sum(~np.isnan(raw))
            valid_spikes = np.sum(~np.isnan(spikes) & (spikes != 0))
            
            stats_text = (
                f"Total Data Points: {valid_raw}\n"
                f"Spikes Isolated: {valid_spikes}\n"
                f"Max Spike Value: {np.nanmax(spikes):.5f}"
            )
            ax2.text(0.05, 0.05, stats_text, transform=ax2.transAxes, 
                     va='bottom', bbox=dict(facecolor='white', alpha=0.8))
            
            # Force 0m to the top of the axis
            depth_min = float(np.nanmin(depth_vals))
            ax2.set_ylim(depth_min, 0)
            
            ax2.set_xlabel("Spike Magnitude")
            ax2.set_ylabel("Depth (m)")
            ax2.set_title("2. Marine Snow / Spike Depth Distribution")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(f"{current_apply_to} Despiking Diagnostics")
            fig.tight_layout()
            plt.show(block=True)