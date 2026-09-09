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

"""QC test to identify impossible locations in LATITUDE and LONGITUDE variables."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


@register_qc
class impossible_location_qc(BaseQC):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the latitude and longitude are valid.
    """

    qc_name = "impossible location qc"
    parameter_schema = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Check LAT/LONG exist within expected bounds
        # TODO: Add optional bounds via parameters (such as Southern Hemisphere, for example)
        flags = {}
        for label, bounds in zip(["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]):
            values = self.data[label].values
            in_bounds = (values > bounds[0]) & (values < bounds[1])
            flags[f"{label}_QC"] = np.where(np.isnan(values), 9, np.where(in_bounds, 1, 4))

        self.flags = xr.Dataset(
            data_vars={col: ("N_MEASUREMENTS", qc) for col, qc in flags.items()},
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        x = fig_spec.x_time(self.data)
        fig, axes = fig_spec.new_fig(nrows=2, sharex=True)

        for ax, var, bounds in zip(
            axes[:, 0], ["LATITUDE", "LONGITUDE"], [(-90, 90), (-180, 180)]
        ):
            fig_spec.flag_points(ax, x, self.data[var].values, self.flags[f"{var}_QC"].values)
            ylabel = fig_spec.axis_label(var, self.data[var].attrs.get("units"))
            fig_spec.style_axes(ax, ylabel=ylabel)
            fig_spec.legend(ax, title="Flags")
            for bound in bounds:
                ax.axhline(bound, ls="--", c="k")
        fig_spec.x_axis(axes[-1][0], x)

        fig_spec.finish(fig, suptitle="Impossible Location Test")
        plt.show(block=True)
