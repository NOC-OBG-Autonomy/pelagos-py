# This file is part of pelagos_py.
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

"""QC test(s) for flagging based on value ranges."""

#### Mandatory imports ####
import numpy as np
from pelagos_py.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib


@register_qc
class impossible_range_qc(BaseQC):
    """
    Target Variable: Any
    Flag Number: Any
    Variables Flagged: Any
    Checks that a meausurement is within a reasonable range.

    EXAMPLE
    -------
    ::

        - name: "Apply QC"
          parameters:
            qc_settings: {
                "impossible range qc": {
                  "variable_ranges": {"PRES": {3: [-2, 0], 4: [-999, -2]}, "LATITUDE": {4: [-90, 90]}},
                  "also_flag": {"PRES": ["CNDC", "TEMP"], "LATITUDE": ["LONGITUDE"]},
                  "plot": ["PRES", "LATITUDE"]
                  "test_depth_range": [-100, 0]  # OPTIONAL
                }
            }
          diagnostics: true
    """

    qc_name = "impossible range qc"

    # Specify if test target variable is user-defined (if True, __init__ has to be redefined)
    dynamic = True

    parameter_schema = {
        "variable_ranges": {
            "type": dict,
            "required": True,
            "description": "Per-variable {flag: [low, high]} ranges to flag within.",
        },
        "also_flag": {
            "type": dict,
            "default": {},
            "description": "Propagate a variable's flags onto other variables.",
        },
        "plot": {
            "type": list,
            "default": [],
            "description": "Variables to plot in diagnostics.",
        },
        "test_depth_range": {
            "type": list,
            "default": None,
            "description": "Optional [min, max] depth window to limit the checks to.",
        },
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.required_variables = list(set(self.variable_ranges.keys()))
        self.tested_variables = self.required_variables.copy()
        if self.test_depth_range is not None:
            self.required_variables.append("DEPTH")

        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.tested_variables)
            | set(f"{var}_QC" for var in sum(self.also_flag.values(), []))
        )

    def return_qc(self):
        # Subset the data
        self.data = self.data[self.required_variables]

        # If the user specified a depth range, limit the checks to that range
        if self.test_depth_range is not None:
            # TODO: -DEPTH
            depth_range_mask = (self.data["DEPTH"] >= self.test_depth_range[0]) & (
                self.data["DEPTH"] <= self.test_depth_range[1]
            )
        else:
            depth_range_mask = True

        # Make the empty QC columns
        for var in self.tested_variables:
            self.data[f"{var}_QC"] = (
                ["N_MEASUREMENTS"],
                np.full(len(self.data[var]), 0),
            )

        # Generate the variable-specific flags
        for var, meta in self.variable_ranges.items():
            for flag, bounds in meta.items():
                self.data[f"{var}_QC"] = xr.where(
                    (
                        depth_range_mask
                        & (self.data[var] > bounds[0])
                        & (self.data[var] < bounds[1])
                        & (self.data[f"{var}_QC"] == 0)
                    ),
                    flag,
                    0,
                )

            # Replace all remaining 0s (unchecked) with 1s (good)
            self.data[f"{var}_QC"] = xr.where(
                depth_range_mask & (self.data[f"{var}_QC"] == 0),
                1,
                self.data[f"{var}_QC"],
            )

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
                "WARNING: In 'range test' diagnostics were called but no plots were specified."
            )
            return

        # Plot the QC output
        fig, axs = plt.subplots(nrows=len(self.plot), figsize=(8, 6), dpi=200)
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

            for bounds in self.variable_ranges[var].values():
                for bound in bounds:
                    ax.axhline(bound, ls="--", c="k")

            ax.set(
                xlabel="Index",
                ylabel=var,
                title=f"{var} Range Test",
            )

            ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)
