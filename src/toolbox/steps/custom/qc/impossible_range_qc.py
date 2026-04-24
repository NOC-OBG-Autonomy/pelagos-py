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

"""QC test(s) for flagging based on value ranges."""

#### Mandatory imports ####
import numpy as np
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

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
    Checks that a measurement is within a reasonable range.

    EXAMPLE
    -------
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
    dynamic = True

    # Define the schema here so BaseQC automatically extracts and assigns them
    expected_parameters = {
        "variable_ranges": {},
        "also_flag": {},
        "plot": [],
        "test_depth_range": None
    }

    def __init__(self, data, **kwargs):
        # 1. Let BaseQC handle data copying, logging setup, and parameter validation
        super().__init__(data, **kwargs)

        # 2. Safety check for the required dynamic config
        if not self.variable_ranges:
            raise KeyError(f"'variable_ranges' is required but missing from {self.qc_name} settings")

        # 3. Dynamically construct required variables based on user config
        self.tested_variables = list(self.variable_ranges.keys())
        self.required_variables = self.tested_variables.copy()
        
        if self.test_depth_range is not None:
            self.required_variables.append("DEPTH")

        # 4. Dynamically construct output columns
        self.qc_outputs = list(
            set(f"{var}_QC" for var in self.tested_variables) | 
            set(f"{var}_QC" for var in sum(self.also_flag.values(), []))
        )

        self.flags = None

    def return_qc(self):
        self.data = self.data[self.required_variables]

        if self.test_depth_range is not None:
            # TODO: -DEPTH
            depth_range_mask = (self.data["DEPTH"] >= self.test_depth_range[0]) & (
                self.data["DEPTH"] <= self.test_depth_range[1]
            )
        else:
            depth_range_mask = True

        for var in self.tested_variables:
            self.data[f"{var}_QC"] = (
                ["N_MEASUREMENTS"],
                np.full(len(self.data[var]), 0),
            )

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

            self.data[f"{var}_QC"] = xr.where(
                depth_range_mask & (self.data[f"{var}_QC"] == 0),
                1,
                self.data[f"{var}_QC"],
            )

            if extra_vars := self.also_flag.get(var):
                for extra_var in extra_vars:
                    self.data[f"{extra_var}_QC"] = self.data[f"{var}_QC"]

        self.flags = self.data[
            [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
        ]

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        if len(self.plot) == 0:
            self.log_warn("Diagnostics were called but no plots were specified in the config.")
            return

        fig, axs = plt.subplots(nrows=len(self.plot), figsize=(8, 6), dpi=200)
        if len(self.plot) == 1:
            axs = [axs]

        for ax, var in zip(axs, self.plot):
            if f"{var}_QC" not in self.qc_outputs:
                self.log_warn(f"Cannot plot {var}_QC as it was not included in this test.")
                continue

            for i in range(10):
                plot_data = self.data[[var, "N_MEASUREMENTS"]].where(
                    self.data[f"{var}_QC"] == i, drop=True
                )

                if len(plot_data[var]) == 0:
                    continue

                ax.plot(
                    plot_data["N_MEASUREMENTS"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )

            for bounds in self.variable_ranges.get(var, {}).values():
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