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

"""QC test that identifies glider positions not located on land and flags accordingly."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
from geodatasets import get_path
import matplotlib.pyplot as plt
import shapely as sh
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import geopandas


@register_qc
class position_on_land_qc(BaseQC):
    """
    Target Variable: LATITUDE, LONGITUDE
    Flag Number: 4 (bad data)
    Variables Flagged: LATITUDE, LONGITUDE
    Checks that the measurement location is not on land.
    """

    qc_name = "position on land qc"
    expected_parameters = {}
    required_variables = ["LATITUDE", "LONGITUDE"]
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        # Convert to pandas
        self.df = self.data[self.required_variables].to_dataframe()

        # Concat the polygons into a MultiPolygon object
        self.world = geopandas.read_file(get_path("naturalearth.land"))
        land_polygons = sh.ops.unary_union(self.world.geometry)

        # Check if lat, long coords fall within the area of the land polygons
        # shapely.contains_xy evaluates arrays quickly and returns a boolean array
        on_land_mask = sh.contains_xy(
            land_polygons, 
            self.df["LONGITUDE"].values, 
            self.df["LATITUDE"].values
        )

        # Apply flags: True (on land) -> 4, False (in water) -> 1
        self.df["LONGITUDE_QC"] = np.where(on_land_mask, 4, 1)
        
        # Add the flags to LATITUDE as well.
        self.df["LATITUDE_QC"] = self.df["LONGITUDE_QC"]

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
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        # Plot land boundaries
        self.world.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.3)

        for i in range(10):
            # Plot by flag number
            plot_data = self.df[self.df["LATITUDE_QC"] == i]
            if plot_data.empty:
                continue

            # Plot the data
            ax.plot(
                plot_data["LONGITUDE"],
                plot_data["LATITUDE"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            xlabel="Longitude",
            ylabel="Latitude",
            title="Position On Land Test",
        )
        ax.legend(title="Flags", loc="upper right")
        fig.tight_layout()
        plt.show(block=True)