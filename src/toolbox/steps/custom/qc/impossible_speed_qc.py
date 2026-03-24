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

"""QC test to identify impossible speeds in glider data."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib


@register_qc
class impossible_speed_qc(BaseQC):
    """
    Target Variable: TIME, LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: TIME, LATITUDE, LONGITUDE
    Checks that the the gliders horizontal speed stays below 3m/s
    """

    qc_name = "impossible speed qc"
    expected_parameters = {}
    required_variables = ["TIME", "LATITUDE", "LONGITUDE"]
    qc_outputs = ["TIME_QC", "LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Convert to pandas
        self.df = self.data[self.required_variables].to_dataframe()

        # Get time difference in seconds safely
        self.df["dt"] = self.df["TIME"].diff().dt.total_seconds()

        # Interpolate missing or infinite coordinates
        for label in ["LATITUDE", "LONGITUDE"]:
            self.df[label] = self.df[label].replace([np.inf, -np.inf], np.nan).interpolate()

        # Convert coordinates to radians for Haversine calculation
        lat_rad = np.radians(self.df["LATITUDE"])
        lon_rad = np.radians(self.df["LONGITUDE"])
        
        # Shift to get previous coordinates
        prev_lat_rad = lat_rad.shift(1)
        prev_lon_rad = lon_rad.shift(1)
        
        # Haversine formula
        a = (
            np.sin((lat_rad - prev_lat_rad) / 2)**2 +
            np.cos(prev_lat_rad) * np.cos(lat_rad) *
            np.sin((lon_rad - prev_lon_rad) / 2)**2
        )
        
        # Radius of Earth is approx 6371000 metres
        self.df["distance_m"] = 6371000.0 * 2 * np.arcsin(np.sqrt(a))

        # Calculate absolute speed in metres per second
        self.df["absolute_speed"] = self.df["distance_m"] / self.df["dt"]

        # First row will be NaN because there is no previous point to calculate speed.
        # We fill the first NaN with 0.0 so it passes the valid speed check.
        self.df["absolute_speed"] = self.df["absolute_speed"].fillna(0.0)

        # TODO: Does this need a flag for potentially bad data for cases where speed is inf?
        speed_is_valid = (
            (self.df["absolute_speed"] < 3.0)  #  Speed threshold
            & self.df["absolute_speed"].notna()
            & np.isfinite(self.df["absolute_speed"])
        )

        for label in ["LATITUDE", "LONGITUDE", "TIME"]:
            self.df[f"{label}_QC"] = np.where(speed_is_valid, 1, 4)

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
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        for i in range(10):
            # Plot by flag number
            plot_data = self.df[self.df["LATITUDE_QC"] == i]
            if plot_data.empty:
                continue

            # Plot the data
            ax.plot(
                plot_data["TIME"],
                plot_data["absolute_speed"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            title="Impossible Speed Test",
            xlabel="Time (s)",
            ylabel="Absolute Horizontal Speed (m/s)",
            ylim=(0, 4),
        )
        ax.axhline(3, ls="--", c="k")
        ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)