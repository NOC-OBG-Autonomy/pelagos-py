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

"""QC test to identify impossible speeds in glider data."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib.pyplot as plt
from pelagos_py.utils.processing_utils import interpolate_by_time
import xarray as xr
import numpy as np
import matplotlib
from pelagos_py.utils import fig_spec


@register_qc
class impossible_speed_qc(BaseQC):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: TIME, LATITUDE, LONGITUDE
    Checks that the the gliders horizontal speed stays below 3m/s
    """

    qc_name = "impossible speed qc"
    parameter_schema = {}
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        time = self.data["TIME"].values
        dt = np.diff(time.astype("datetime64[ns]").astype("int64"), prepend=np.int64(0)) * 1e-9
        dt[0] = np.nan  # no preceding sample

        # Degrees/second per axis, on gap-filled positions (non-finite -> interpolated)
        speeds = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for label in ["LATITUDE", "LONGITUDE"]:
                filled = interpolate_by_time(self.data[label].values, time)
                speeds[label] = np.diff(filled, prepend=np.nan) / dt
            absolute_speed = (speeds["LATITUDE"] ** 2 + speeds["LONGITUDE"] ** 2) ** 0.5

            # TODO: Does this need a flag for potentially bad data for cases where speed is inf?
            speed_is_valid = np.isfinite(absolute_speed) & (absolute_speed < 3)  # Speed threshold
        self.absolute_speed = absolute_speed  # for plot_diagnostics
        qc = np.where(speed_is_valid, 1, 4)

        self.flags = xr.Dataset(
            data_vars={
                f"{label}_QC": ("N_MEASUREMENTS", qc.copy())
                for label in ["LATITUDE", "LONGITUDE", "TIME"]
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        fig_spec.flag_points(
            ax, self.data["TIME"].values, self.absolute_speed, self.flags["LATITUDE_QC"].values
        )
        fig_spec.date_axis(ax, which="x")
        ax.set_ylim(0, 4)
        ax.axhline(3, ls="--", c="k")
        fig_spec.style_axes(
            ax, xlabel="Time", ylabel="Absolute Horizontal Speed (m/s)"
        )
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Impossible Speed Test")
        plt.show(block=True)
