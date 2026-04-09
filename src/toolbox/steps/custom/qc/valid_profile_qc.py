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

"""QC tests for assessing validity of a glider profile, based on different definitions of successful data."""

#### Mandatory imports ####
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import polars as pl
import xarray as xr
import matplotlib


@register_qc
class valid_profile_qc(BaseQC):
    """
    Target Variable: PROFILE_NUMBER
    Flag Number: 4 (bad data), 3 (potentially bad)
    Variables Flagged: PROFILE_NUMBER
    Checks that each profile is of a certain length (in number of points)
    and contains points within a specified depth range.
    """

    qc_name = "valid profile qc"
    expected_parameters = {
        "profile_length": 1000,
        "depth_range": (-1000, 0),
    }
    required_variables = ["PROFILE_NUMBER", "DEPTH"]
    qc_outputs = ["PROFILE_NUMBER_QC"]

    def return_qc(self):
        # Convert to polars
        self.df = pl.from_pandas(
            self.data[self.required_variables].to_dataframe(), nan_to_null=False
        )

        # Check profiles are of a given length
        profile_lengths = self.df.group_by("PROFILE_NUMBER").count()
        self.df = self.df.join(profile_lengths, on="PROFILE_NUMBER", how="left")

        # Find profiles that have no data between the sepcified depth ranges
        profile_ranges = self.df.group_by("PROFILE_NUMBER").agg(
            (pl.col("DEPTH").is_between(*self.depth_range).any()).alias(
                "in_depth_range"
            )
        )
        self.df = self.df.join(profile_ranges, on="PROFILE_NUMBER", how="left")

        self.df = self.df.with_columns(
            pl.when(pl.col("PROFILE_NUMBER").is_nan())
            .then(9)
            .when(pl.col("count") < self.profile_length)
            .then(4)
            .when(pl.col("in_depth_range").not_())
            .then(3)
            .otherwise(1)
            .alias("PROFILE_NUMBER_QC")
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
            plot_data = self.df.with_row_index().filter(
                pl.col("PROFILE_NUMBER_QC") == i
            )
            if len(plot_data) == 0:
                continue

            # Plot the data
            ax.plot(
                plot_data["index"],
                plot_data["DEPTH"],
                c=flag_cols[i],
                ls="",
                marker="o",
                label=f"{i}",
            )

        ax.set(
            xlabel="Index",
            ylabel="Pressure",
            title="Valid Profile Test",
        )
        ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)
