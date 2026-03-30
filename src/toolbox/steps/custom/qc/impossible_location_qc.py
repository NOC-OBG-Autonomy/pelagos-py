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
    required_variables = []
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})

        if "LATITUDE" not in self.data or "LONGITUDE" not in self.data:
            print("Warning: LATITUDE or LONGITUDE missing. Skipping impossible location qc.")
            return self.flags

        # Check LAT/LONG exist within expected bounds
        # TODO: Add optional bounds via parameters (such as Southern Hemisphere, for example)
        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            var_data = self.data[label]
            
            qc_var = xr.where((var_data > bounds[0]) & (var_data < bounds[1]), 1, 4)
            qc_var = xr.where(var_data.isnull(), 9, qc_var)
            
            self.flags[f"{label}_QC"] = qc_var

        return self.flags

    def plot_diagnostics(self):
        if "LATITUDE" not in self.data or "LONGITUDE" not in self.data:
            return
            
        matplotlib.use("tkagg")
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharex=True, dpi=200)

        row_index = np.arange(self.data.sizes["N_MEASUREMENTS"])

        for ax, var, bounds in zip(
            axs, ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            for i in range(10):
                if f"{var}_QC" not in self.flags:
                    continue
                    
                mask = self.flags[f"{var}_QC"] == i
                if not mask.any():
                    continue

                ax.plot(
                    row_index[mask.values],
                    self.data[var].values[mask.values],
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