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

"""QC test(s) for flagging stuck, static, or otherwise unchanged data (which should be changing)."""

#### Mandatory imports ####
import numpy as np
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib


@register_qc
class stuck_value_qc(BaseQC):
    """
    Target Variable: Any
    Flag Number: 4 (bad)
    Variables Flagged: Any
    Checks that successive measurements are not frozen.

    EXAMPLE
    -------
    - name: "Apply QC"
      parameters:
        qc_settings: {
            "stuck value qc": {
              "variables": {"PRES": 4, "LATITUDE": 100},
              "also_flag": {"PRES": ["CNDC", "TEMP"], "LATITUDE": ["LONGITUDE"]},
              "plot": ["PRES", "LATITUDE"]
            }
        }
      diagnostics: true
    """

    qc_name = "stuck value qc"
    dynamic = True

    # Define the schema here so BaseQC automatically extracts and assigns them
    expected_parameters = {
        "variables": {},
        "also_flag": {},
        "plot": []
    }

    def __init__(self, data, **kwargs):
        # 1. Let BaseQC handle data copying, logging setup, and parameter validation
        super().__init__(data, **kwargs)

        # 2. Safety check for the required dynamic config
        if not getattr(self, "variables", None):
            raise KeyError(f"'variables' is required but missing from {self.qc_name} settings")

        # 3. Dynamically construct required variables based on user config
        self.required_variables = list(self.variables.keys())
        
        # 4. Dynamically construct output columns
        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.required_variables) | 
            set(f"{var}_QC" for var in sum(self.also_flag.values(), []))
        )

        self.flags = None

    def return_qc(self):
        # Subset the data
        self.data = self.data[self.required_variables]

        # Generate the variable-specific flags
        for var, n_stuck in self.variables.items():
            # remove nans
            var_data = self.data[var].dropna(dim="N_MEASUREMENTS")

            # Calculate forward (step=1) and backward (step=-1) differences across the variable
            backward_diff = np.diff(var_data, append=0)
            forward_diff = np.diff(var_data[::-1], append=0)[::-1]

            # When either diff is 0 at a given index, then the value is stuck
            stuck_value_mask = (backward_diff == 0) | (forward_diff == 0)

            # Handle edge cases
            for index, step in zip([0, -1], [1, -1]):
                stuck_value_mask[index] = var_data[index] == var_data[index + step]

            # The remaining processing has to be in int dtype
            stuck_value_mask = stuck_value_mask.astype(int)

            # Find transitions between stuck and unstuck
            switching_points = np.diff(np.concatenate([[0], stuck_value_mask, [0]]))
            starts = np.where(switching_points == 1)[0]
            ends = np.where(switching_points == -1)[0]

            # Replace the value of each element in a group of stuck values with the length of that group
            for start, end in zip(starts, ends):
                stuck_value_mask[start:end] = end - start

            # Convert the stuck values mask into flags
            bad_values = stuck_value_mask > n_stuck
            stuck_value_mask[bad_values] = 4
            stuck_value_mask[~bad_values] = 1

            # Insert the flags into the QC column
            nan_mask = np.isnan(self.data[var])
            self.data[f"{var}_QC"] = (["N_MEASUREMENTS"], np.where(nan_mask, 9, 1))
            self.data[f"{var}_QC"][np.where(~nan_mask)] = stuck_value_mask

            # Broadcast the QC found for var into variables specified by "also_flag"
            if extra_vars := self.also_flag.get(var):
                for extra_var in extra_vars:
                    self.data[f"{extra_var}_QC"] = self.data[f"{var}_QC"]

        # Select just the flags
        self.flags = self.data[
            [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
        ]

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # If not plots were specified
        if len(self.plot) == 0:
            self.log_warn("Diagnostics were called but no variables were specified for plotting.")
            return

        # Plot the QC output
        fig, axs = plt.subplots(
            nrows=len(self.plot), figsize=(8, 6), sharex=True, dpi=200
        )
        if len(self.plot) == 1:
            axs = [axs]

        for ax, var in zip(axs, self.plot):
            # Check that the user specified var exists in the test set
            if f"{var}_QC" not in self.qc_outputs:
                self.log_warn(f"Cannot plot {var}_QC as it was not included in this test.")
                continue

            for i in range(10):
                # Plot by flag number
                plot_data = self.data[[var, "N_MEASUREMENTS"]].where(
                    self.data[f"{var}_QC"] == i, drop=True
                )

                if len(plot_data[var]) == 0:
                    continue

                # Plot the data
                ax.plot(
                    plot_data["N_MEASUREMENTS"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )

            ax.set(
                xlabel="Index",
                ylabel=var,
                title=f"{var} Stuck Value Test",
            )

            ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)