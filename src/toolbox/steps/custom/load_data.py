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

from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag

import xarray as xr
import pandas as pd
import numpy as np

MIN_YEAR_FILTER = "1990-01-01"


@register_step
class LoadOG1(BaseStep):
    """
    Initialises the LoadOG1 step.

    Parameters
    ----------
    filter_bad_time : bool, optional
        If True, REMOVES all timestamps outside the expected time window.
    data_start : str or np.datetime64, optional
        The minimum valid timestamp for the data. If not provided, the filter defaults to the DEPLOYMENT_TIME found in the dataset, or 1990-01-01T00:00:00 if no deployment time is found.
    data_end : str or np.datetime64, optional
        The maximum valid timestamp for the data. If not provided, it defaults to the current system time when the pipeline is run.

    Example usage in a pipeline configuration:
    steps:
      - name: Load OG1
        parameters:
          file_path: "/path/to/your/dataset.nc"
          filter_bad_time: true
          data_start: "2023-05-01T00:00:00"
          data_end: "2024-05-01T00:00:00"
    """

    step_name = "Load OG1"
    required_variables = []
    provided_variables = ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "CNDC"]

    def __init__(
        self, filter_bad_time=False, data_start=None, data_end=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filter_bad_time = filter_bad_time
        self.data_start = data_start
        self.data_end = data_end

    def run(self):
        # load data from xarray
        self.data = xr.open_dataset(self.file_path)
        self.log(f"Loaded data from {self.file_path}")

        self.log("Loading dataset to RAM...")
        self.data.load()

        if self.filter_bad_time and (
            "TIME" in self.data.variables or "TIME" in self.data.coords
        ):
            orig_len = len(self.data["TIME"])
            time_array = self.data["TIME"]

            start_val = (
                np.datetime64(self.data_start)
                if self.data_start
                else np.datetime64(MIN_YEAR_FILTER)
            )
            end_val = (
                np.datetime64(self.data_end)
                if self.data_end
                else np.datetime64(pd.Timestamp.now())
            )

            valid_mask = time_array >= start_val
            valid_mask &= time_array <= end_val

            if not self.data_start and "DEPLOYMENT_TIME" in self.data.variables:
                deploy_time = pd.to_datetime(self.data["DEPLOYMENT_TIME"].values)
                if isinstance(deploy_time, pd.DatetimeIndex):
                    deploy_time = deploy_time[0]
                valid_mask &= time_array >= np.datetime64(deploy_time)

            time_dim = self.data["TIME"].dims[0]
            self.data = self.data.isel({time_dim: valid_mask.values})
            new_len = len(self.data["TIME"])

            if new_len < orig_len:
                self.log_warn(
                    f"Removed {orig_len - new_len} records containing invalid or pre-deployment timestamps."
                )

        if "TIME" in self.data.coords:
            self.data = self.data.reset_coords("TIME", drop=False)
            self.data = self.data.reset_coords("LATITUDE", drop=False)
            self.data = self.data.reset_coords("LONGITUDE", drop=False)

        if "TIME" not in self.data.data_vars:
            raise ValueError(
                "\n'TIME' could not be found in the dataset. Pipelines cannot be run without this variable.\n"
                "If TIME is listed under another name, please rename it to conform to the OG1 format."
            )

        if np.any(np.isnan(self.data["TIME"])):
            raise ValueError(
                "\n'TIME' has nan values. Pipelines cannot be run without a continuous monotonic time coordinate.\n"
                "Please remove these values (and their concurrent measurements) from the input."
            )

        if not np.all(np.diff(self.data["TIME"]) >= 0):
            self.log_warn(
                "'TIME' is not monotonically increasing. This may cause fatal issues in processing. "
                "Please check the quality of your input data."
            )

        # TODO: Remove QC column resetting when BODC has properly implemented QC outputs
        # Reset all data variable flags. Set unchecked data flags to 0 and missing data flags to 9
        # cols_to_qc = [
        #     var for var in self.data.data_vars
        #     if var.isupper() and (var not in self.data.dims) and ("_QC" not in var)
        # ]
        # data_subset = self.data[cols_to_qc]
        # masks = xr.where(data_subset.isnull(), 9, 0).astype(int)
        # masks = masks.rename({var: f"{var}_QC" for var in cols_to_qc})
        # self.data.update(masks)

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        self.log("Generating diagnostics...")
        diag.generate_info(self.data)
