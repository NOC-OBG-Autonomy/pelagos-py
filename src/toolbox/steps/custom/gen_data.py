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

"""Step for generating synthetic data for testing pipelines"""

import polars as pl
import xarray as xr
import numpy as np
from ..base_step import BaseStep, register_step
from datetime import date, timedelta


@register_step
class GenerateData(BaseStep):
    """
    Step for generating synthetic data for testing pipelines.
    
    Example config setup:

    """

    step_name = "Generate Data"
    required_variables = []
    provided_variables = ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"]

    def run(self):
        # Check if the data is already in the context
        if "data" in self.context:
            raise ValueError(
                "[Generate Data] WARNING: Data found in context. This will be replaced by generated data."
            )

        if self.gen_fixed_data:
            self.log("Generating fixed data")
            import itertools

            ncols = 2
            column_names = ["A", "B", "C"][:ncols]
            qc_values = np.array(list(itertools.product(range(10), repeat=ncols)))
            values = [[i] * int(10**ncols) for i in range(1, ncols + 1)]
            df = pl.DataFrame(
                {
                    **{col: values[i] for i, col in enumerate(column_names)},
                    **{
                        f"{col}_QC": qc_values[:, i]
                        for i, col in enumerate(column_names)
                    },
                }
            )

        else:
            self.log("Generating random data")
            # Load config parameters
            start_date, end_date, sample_period = self.parameters["sampling_info"]
            additional_variables = self.parameters["additional_variables"]
            user_value_limits = self.parameters["value_limits"]
            diagnostics = self.parameters["diagnostics"]

            # Add aditional variables
            variable_names = {"LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"}
            variable_names.update(additional_variables)

            # Define variable limits and update with user values
            variable_limits = {
                "LATITUDE": [-90, 90],  # Degrees
                "LONGITUDE": [-180, 180],  # Degrees
                "PRES": [0, 100],  # Bar
                "TEMP": [0, 20],  # Celcius
                "CNDC": [34, 35],  # S/m
            }
            variable_limits.update(user_value_limits)
            if diagnostics:
                self.log(f"[Generate Data] Variables: {variable_limits}")

            # Make time index for dataframe (df)
            df = pl.select(
                pl.datetime_range(
                    date(*map(int, start_date.split("-"))),
                    date(*map(int, end_date.split("-"))),
                    timedelta(seconds=sample_period),
                    time_unit="ns",
                ).alias("TIME")
            )
            data_length = len(df)

            # Generate random data for the remaining variables
            for variable_name in variable_names:
                # Check the limits
                if variable_name in variable_limits.keys():
                    lower, upper = variable_limits[variable_name]
                    if upper <= lower:
                        raise ValueError(
                            f"Upper limit must be greater than lower limit for {variable_name}"
                        )
                else:
                    self.log(
                        f"The additional variable {variable_name} has not been set limits. Defaulting to [0, 1]."
                    )
                    lower, upper = [0, 1]

                # Add the new column
                df = df.with_columns(
                    pl.lit(np.random.uniform(lower, upper, data_length)).alias(
                        variable_name
                    )
                )

            # Make the xarray data from the polars dataframe and ship it
            # TODO: Add metadata flexibility

        data = df.to_pandas().to_xarray()
        data["N_PARAM"] = list(data.keys())
        data = data.rename({"index": "N_MEASUREMENTS"})
        self.context["data"] = data
        return self.context
