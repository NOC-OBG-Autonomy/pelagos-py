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
import polars as pl
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
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        self.df = self.df.with_columns(
            (pl.col("TIME").diff().cast(pl.Float64) * 1e-9).alias("dt")
        )
        for label in ["LATITUDE", "LONGITUDE"]:
            self.df = self.df.with_columns(
                pl.col(label)
                .replace([np.inf, -np.inf, np.nan], None)
                .interpolate_by("TIME")
                .diff()
                .alias(f"delta_{label}")
            )
            self.df = self.df.with_columns(
                (pl.col(f"delta_{label}") / pl.col("dt")).alias(f"{label}_speed")
            )
        # Define absolute speed
        self.df = self.df.with_columns(
            (
                (pl.col("LATITUDE_speed") ** 2 + pl.col("LONGITUDE_speed") ** 2) ** 0.5
            ).alias("absolute_speed")
        )

        # TODO: Does this need a flag for potentially bad data for cases where speed is inf?
        self.df = self.df.with_columns(
            (
                (pl.col("absolute_speed") < 3)  #  Speed threshold
                & pl.col("absolute_speed").is_not_null()
                & pl.col("absolute_speed").is_finite()
            ).alias("speed_is_valid")
        )

        for label in ["LATITUDE", "LONGITUDE", "TIME"]:
            self.df = self.df.with_columns(
                pl.when(pl.col("speed_is_valid"))
                .then(1)
                .otherwise(4)
                .alias(f"{label}_QC")
            )

        # Convert back to xarray
        flags = self.df.select(pl.col("^.*_QC$"))
        self.flags = xr.Dataset(
            data_vars={
                col: ("N_MEASUREMENTS", flags[col].to_numpy()) for col in flags.columns
            },
            coords={"N_MEASUREMENTS": self.data["N_MEASUREMENTS"]},
        )

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        for i in range(10):
            # Plot by flag number
            plot_data = self.df.filter(pl.col("LATITUDE_QC") == i)
            if len(plot_data) == 0:
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
