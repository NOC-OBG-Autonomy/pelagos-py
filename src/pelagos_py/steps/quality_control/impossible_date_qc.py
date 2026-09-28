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

"""QC test to identify impossible dates in TIME variable."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from pelagos_py.utils import fig_spec


@register_qc
class impossible_date_qc(BaseQC):
    """
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    qc_name = "impossible date qc"
    parameter_schema = {}
    required_variables = ["TIME"]
    qc_outputs = ["TIME_QC"]

    def return_qc(self):
        time = self.data["TIME"].values

        # Check if any of the datetime stamps fall outside 1985 and the current datetime
        # TODO: Add optional bounds via parameters (such as known deployment dates, for example)
        in_range = (time > np.datetime64(datetime(1985, 1, 1))) & (
            time < np.datetime64(datetime.now())
        )
        time_qc = np.where(np.isnat(time), 9, np.where(in_range, 1, 4))

        self.flags = xr.Dataset(
            data_vars={"TIME_QC": ("N_MEASUREMENTS", time_qc)},
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        time = self.data["TIME"].values
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        fig_spec.flag_points(ax, np.arange(time.size), time, self.flags["TIME_QC"].values)
        fig_spec.date_axis(ax, which="y")
        fig_spec.style_axes(ax, xlabel="Index", ylabel="TIME")
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Impossible Date Test")
        plt.show(block=True)
