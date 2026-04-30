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
    required_variables = []
    qc_outputs = ["LATITUDE_QC", "LONGITUDE_QC"]

    def return_qc(self):
        self.flags = xr.Dataset(coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]})

        if "LATITUDE" not in self.data or "LONGITUDE" not in self.data:
            self.log_warn("LATITUDE or LONGITUDE missing. Skipping position on land qc.")
            return self.flags

        # Concat the polygons into a MultiPolygon object
        self.world = geopandas.read_file(get_path("naturalearth.land"))
        land_polygons = sh.ops.unary_union(self.world.geometry)

        # Check if lat, long coords fall within the area of the land polygons
        # shapely.contains_xy evaluates arrays quickly and returns a boolean array
        on_land_mask = sh.contains_xy(
            land_polygons, 
            self.data["LONGITUDE"].values, 
            self.data["LATITUDE"].values
        )

        # Apply flags: True (on land) -> 4, False (in water) -> 1
        flag_values = np.where(on_land_mask, 4, 1)

        for col in ["LATITUDE", "LONGITUDE"]:
            self.flags[f"{col}_QC"] = ("N_MEASUREMENTS", flag_values)

        return self.flags

    def plot_diagnostics(self):
        if "LATITUDE" not in self.data or "LONGITUDE" not in self.data:
            return

        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

        # Plot land boundaries
        self.world.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.3)

        for i in range(10):
            if "LATITUDE_QC" not in self.flags:
                continue
                
            # Plot by flag number
            mask = self.flags["LATITUDE_QC"] == i
            if not mask.any():
                continue

            # Plot the data
            ax.plot(
                self.data["LONGITUDE"].values[mask.values],
                self.data["LATITUDE"].values[mask.values],
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