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

"""QC test to identify impossible locations in LATITUDE and LONGITUDE variables."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt


@register_qc
class impossible_location_qc(BaseQC):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the latitude and longitude are valid.
    """

    qc_name = "impossible location qc"
    expected_parameters = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Convert to pandas
        self.df = self.data[self.required_variables].to_dataframe()

        # Check LAT/LONG exist within expected bounds
        # TODO: Add optional bounds via parameters (such as Southern Hemisphere, for example)
        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            conditions = [
                self.df[label].isna(),
                (self.df[label] > bounds[0]) & (self.df[label] < bounds[1])
            ]
            # 9 for NaN, 1 for good, 4 for bad
            choices = [9, 1]
            self.df[f"{label}_QC"] = np.select(conditions, choices, default=4)

        # Convert back to xarray
        flags = self.df[[f"{col}_QC" for col in self.required_variables]]
        self.flags = xr.Dataset(
            data_vars={
                col: ("N_MEASUREMENTS", flags[col].values) for col in flags.columns
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, dpi=200)

        # Add a numeric row index for plotting on the x-axis
        self.df["row_index"] = np.arange(len(self.df))

        for ax, var, bounds in zip(
            axs, ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            for i in range(10):
                # Plot by flag number
                plot_data = self.df[self.df[f"{var}_QC"] == i]
                if plot_data.empty:
                    continue

                # Plot the data
                ax.plot(
                    plot_data["row_index"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )
            ax.set(
                xlabel="Index",
                ylabel=var,
            )
            ax.legend(title="Flags", loc="upper right")
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")

        fig.suptitle("Impossible Location Test")
        fig.tight_layout()
        plt.show(block=True)