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

"""QC tests for assessing validity of a glider profile, based on different definitions of successful data."""

#### Mandatory imports ####
from pelagos_py.steps.base_qc import BaseQC, register_qc

#### Custom imports ####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
from pelagos_py.utils import fig_spec


@register_qc
class valid_profile_qc(BaseQC):
    """
    Flag whole profiles that are too short or never reach a target depth range.

    | **Target variable:** ``PROFILE_NUMBER``
    | **Variables flagged:** ``PROFILE_NUMBER``
    | **Flags applied:** 1 (good), 3 (probably bad), 4 (bad), 9 (missing)

    Each profile (a group of measurements sharing a ``PROFILE_NUMBER``, as produced
    by :doc:`Find Profiles <../processing/find_profiles/index>`) is assessed as a
    whole and every row in that profile receives the same ``PROFILE_NUMBER_QC``:

    - **9 (missing)** — the row has no profile (``PROFILE_NUMBER`` is NaN, e.g.
      surfacing rows or data gaps).
    - **4 (bad)** — the profile contains fewer than ``profile_length`` measurements.
    - **3 (probably bad)** — the profile is long enough but has no measurement whose
      ``DEPTH`` falls inside ``depth_range``.
    - **1 (good)** — the profile passes both checks.

    Only ``PROFILE_NUMBER_QC`` is written; the underlying data is never modified.

    Parameters
    ----------
    profile_length : int, optional
        Minimum number of measurements a profile must contain to be kept. Profiles
        shorter than this are flagged bad (4). Default ``100``.
    depth_range : tuple of float, optional
        ``(min, max)`` depth window (in the same units/sign convention as ``DEPTH``,
        i.e. positive downward) that a profile must reach into. A profile with no data
        inside this window is flagged probably bad (3). Default ``(0, 1000)``.

    Examples
    --------
    The check works with its defaults, so the minimal configuration sets no
    parameters:

    .. code-block:: yaml

        - name: "Apply QC"
          parameters:
            qc_settings:
              valid profile qc: {}

    Both parameters may be tuned — here profiles must be at least 50 points long and
    contain data somewhere between 1000 m depth and the surface:

    .. code-block:: yaml

        - name: "Apply QC"
          parameters:
            qc_settings:
              valid profile qc:
                profile_length: 50
                depth_range: [0, 1000]
          diagnostics: true  # plot DEPTH vs index, coloured by the resulting flag
    """

    qc_name = "valid profile qc"
    parameter_schema = {
        "profile_length": {
            "type": int,
            "default": 100,
            "description": "Minimum number of measurements a profile must contain to be kept.",
        },
        "depth_range": {
            "type": list,
            "default": (0, 1000),
            "description": "(min, max) depth window a profile must reach into.",
        },
    }
    required_variables = ["PROFILE_NUMBER", "DEPTH"]
    qc_outputs = ["PROFILE_NUMBER_QC"]

    def return_qc(self):
        profile_number = self.data["PROFILE_NUMBER"].values
        depth = self.data["DEPTH"].values
        lower, upper = self.depth_range

        # Per-sample: length of its profile, and whether the profile reaches the depth
        # range (samples without a profile number get NaN here and are flagged 9 below)
        by_profile = pd.Series((depth >= lower) & (depth <= upper)).groupby(profile_number)
        count = by_profile.transform("size").to_numpy(dtype=float)
        in_depth_range = by_profile.transform("any").fillna(False).to_numpy(dtype=bool)

        qc = np.select(
            [pd.isna(profile_number), count < self.profile_length, ~in_depth_range],
            [9, 4, 3],
            default=1,
        )

        self.flags = xr.Dataset(
            data_vars={"PROFILE_NUMBER_QC": ("N_MEASUREMENTS", qc)},
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        x = fig_spec.x_time(self.data)
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        fig_spec.flag_points(ax, x, self.data["DEPTH"].values, self.flags["PROFILE_NUMBER_QC"].values)
        fig_spec.style_axes(ax, ylabel="Pressure")
        fig_spec.x_axis(ax, x)
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Valid Profile Test")
        plt.show(block=True)
