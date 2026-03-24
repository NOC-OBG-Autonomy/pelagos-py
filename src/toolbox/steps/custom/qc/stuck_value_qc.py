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
            "stuck value test": {
              "variables": {"PRES": 4, "LATITUDE": 100},
              "also_flag": {"PRES": ["CNDC", "TEMP"], "LATITUDE": ["LONGITUDE"]},
              "plot": ["PRES", "LATITUDE"]
            }
        }
      diagnostics: true
    """

    qc_name = "stuck value qc"

    # Specify if test target variable is user-defined (if True, __init__ has to be redefined)
    dynamic = True

    def __init__(self, data, **kwargs):
        # Check the necessary kwargs are available
        required_kwargs = {"variables", "also_flag", "plot"}
        if not required_kwargs.issubset(set(kwargs.keys())):
            raise KeyError(
                f"{required_kwargs - set(kwargs.keys())} are missing from {self.qc_name} settings"
            )

        # Specify the tests paramters from kwargs (config)
        self.expected_parameters = {
            k: v for k, v in kwargs.items() if k in required_kwargs
        }
        self.required_variables = list(
            set(self.expected_parameters["variables"].keys())
        )
        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.required_variables)
            | set(
                f"{var}_QC"
                for var in sum(self.expected_parameters["also_flag"].values(), [])
            )
        )

        if data is not None:
            self.data = data.copy(deep=True)

        for k, v in self.expected_parameters.items():
            setattr(self, k, v)

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
            print(
                f"WARNING: In '{self.qc_name}', diagnostics were called but no variables were specified for plotting."
            )
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
                print(
                    f"WARNING: Cannot plot {var}_QC as it was not included in this test."
                )
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
