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
    Checks that the horizontal speed between GPS fixes stays below ``threshold`` (m/s).

    Only real fixes (finite LATITUDE and LONGITUDE) are used. Each fix is compared
    against the last *good* fix at least ``min_interval`` seconds earlier, so GPS
    jitter between fixes a few seconds apart (10 m in 1 s reads as 10 m/s) does
    not trip the test, and a single bad fix does not drag the next one down with it.

    Example::

        impossible speed qc:
          threshold: 3       # m/s
          min_interval: 60   # s
    """

    qc_name = "impossible speed qc"
    parameter_schema = {
        "threshold": {
            "type": [int, float],
            "default": 3,
            "description": "Maximum plausible horizontal speed (m/s).",
        },
        "min_interval": {
            "type": [int, float],
            "default": 60,
            "description": "Minimum seconds between the two fixes a speed is measured over.",
        },
    }
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        time = self.data["TIME"].values.astype("datetime64[ns]").astype("int64") * 1e-9
        lat, lon = self.data["LATITUDE"].values, self.data["LONGITUDE"].values
        has_fix = np.isfinite(lat) & np.isfinite(lon)
        fix = np.flatnonzero(has_fix)
        speed = np.full(lat.shape, np.nan)
        bad = np.zeros(lat.shape, dtype=bool)

        ref = None  # index of the last good fix used as the reference
        for i in fix:
            if ref is not None and time[i] - time[ref] >= self.min_interval:
                dlat = lat[i] - lat[ref]
                dlon = (lon[i] - lon[ref] + 180) % 360 - 180  # dateline-safe
                mid_lat = np.deg2rad((lat[i] + lat[ref]) / 2)
                speed[i] = np.hypot(dlat, dlon * np.cos(mid_lat)) * 111_195 / (time[i] - time[ref])
                bad[i] = speed[i] >= self.threshold
            if not bad[i] and (ref is None or time[i] - time[ref] >= self.min_interval):
                ref = i
        self.absolute_speed = speed  # for plot_diagnostics

        qc = np.where(bad, 4, 1)
        pos_qc = np.where(has_fix, qc, 9)
        self.flags = xr.Dataset(
            data_vars={
                "LATITUDE_QC": ("N_MEASUREMENTS", pos_qc),
                "LONGITUDE_QC": ("N_MEASUREMENTS", pos_qc.copy()),
                "TIME_QC": ("N_MEASUREMENTS", qc),
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        time = self.data["TIME"].values
        lat, lon = self.data["LATITUDE"].values, self.data["LONGITUDE"].values
        flags = self.flags["LATITUDE_QC"].values
        fig, axes = fig_spec.new_fig(1, 2, width_ratios=(3, 2))

        ax = axes[0][0]
        fig_spec.flag_points(ax, time, self.absolute_speed, flags)
        fig_spec.date_axis(ax, which="x", index=time)
        ax.set_ylim(bottom=0)
        ax.axhline(self.threshold, ls="--", c="k")
        fig_spec.style_axes(ax, xlabel="Time", ylabel="Speed since last good fix (m/s)")

        # Map of the fixes so a flagged jump can be seen against the track.
        ax = axes[0][1]
        fix = np.isfinite(lat) & np.isfinite(lon)
        if fix.sum():
            ax.plot(lon[fix], lat[fix], color="0.6", lw=0.6, zorder=0)
            fig_spec.flag_points(ax, lon, lat, flags)
            fig_spec.style_axes(ax, xlabel="LONGITUDE", ylabel="LATITUDE")
            fig_spec.coastlines(ax, fig_spec.map_extent(lon, lat, pad=0.15))
        fig_spec.legend(ax, title="Flags")
        fig_spec.finish(fig, suptitle="Impossible Speed Test")
        plt.show(block=True)
