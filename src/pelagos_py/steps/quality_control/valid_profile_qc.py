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

"""Flag whole profiles that are too short or never reach a target depth range."""

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
    | **Flags applied:** ``flag`` (default 4, bad)

    Every row of a profile (rows sharing a ``PROFILE_NUMBER`` from
    :doc:`Find Profiles <../processing/find_profiles/index>`) gets ``flag`` when the
    profile has fewer than ``min_length`` rows or, if ``depth_range`` is set, no
    ``depth_var`` sample inside it. Passing profiles are flagged 1; rows without a
    profile are left unchecked (0), so their existing flag is kept.

    Examples
    --------
    .. code-block:: yaml

        - name: "Apply QC"
          parameters:
            qc_settings:
              valid profile qc:
                min_length: 50          # rows per profile
                depth_range: [0, 1000]  # PRES window the profile must reach into
                flag: 4
          diagnostics: true
    """

    qc_name = "valid profile qc"
    parameter_schema = {
        "min_length": {
            "type": int,
            "default": 100,
            "description": "Minimum number of rows a profile must contain.",
        },
        "depth_range": {
            "type": list,
            "default": None,
            "description": "Optional [min, max] depth_var window a profile must have a sample in.",
        },
        "depth_var": {
            "type": str,
            "default": "PRES",
            "description": "Vertical-coordinate variable used for depth_range.",
        },
        "flag": {
            "type": int,
            "default": 4,
            "description": "QC flag given to every row of a failing profile.",
        },
    }
    dynamic = True
    required_variables = ["PROFILE_NUMBER"]
    qc_outputs = ["PROFILE_NUMBER_QC"]

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        if not 0 <= self.flag <= 9:
            raise ValueError(f"[{self.qc_name}] invalid QC flag {self.flag!r}; expected 0-9.")
        self.required_variables = ["PROFILE_NUMBER"] + ([self.depth_var] if self.depth_range else [])

    def return_qc(self):
        profile = pd.Series(self.data["PROFILE_NUMBER"].values)
        has_profile = profile.notna().to_numpy()
        bad = profile.groupby(profile).transform("size").to_numpy() < self.min_length
        if self.depth_range:
            lower, upper = self.depth_range
            depth = self.data[self.depth_var].values
            in_range = pd.Series((depth >= lower) & (depth <= upper))
            bad |= ~in_range.groupby(profile).transform("any").fillna(False).to_numpy(dtype=bool)

        qc = np.where(has_profile, np.where(bad, self.flag, 1), 0).astype(np.int8)
        self.flags = xr.Dataset(
            {"PROFILE_NUMBER_QC": ("N_MEASUREMENTS", qc)},
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        x = fig_spec.x_time(self.data)
        fig, axes = fig_spec.new_fig()
        ax = axes[0][0]
        y = self.data[self.depth_var]
        fig_spec.flag_points(ax, x, y.values, self.flags["PROFILE_NUMBER_QC"].values)
        fig_spec.style_axes(ax, ylabel=fig_spec.axis_label(self.depth_var, y.attrs.get("units")))
        ax.invert_yaxis()
        fig_spec.x_axis(ax, x)
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Valid Profile Test")
        plt.show(block=True)
