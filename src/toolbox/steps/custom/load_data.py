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
    step_name = "Load OG1"

    def run(self):
        self.data = xr.open_dataset(self.file_path)
        self.log(f"Loaded data from {self.file_path}")

        self.log("Loading dataset to RAM...")
        self.data.load()

        if "TIME" in self.data.variables or "TIME" in self.data.coords:
            orig_len = len(self.data["TIME"])
            time_array = self.data["TIME"]
            
            valid_mask = time_array >= pd.to_datetime(MIN_YEAR_FILTER)
            valid_mask &= (time_array <= pd.Timestamp.now())
            
            if "DEPLOYMENT_TIME" in self.data.variables:
                deploy_time = pd.to_datetime(self.data["DEPLOYMENT_TIME"].values)
                if isinstance(deploy_time, pd.DatetimeIndex):
                    deploy_time = deploy_time[0]
                valid_mask &= (time_array >= deploy_time)
                
            time_dim = self.data["TIME"].dims[0]
            self.data = self.data.isel({time_dim: valid_mask.values})
            new_len = len(self.data["TIME"])
            
            if new_len < orig_len:
                self.log_warn(f"Removed {orig_len - new_len} records containing invalid or pre-deployment timestamps.")

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

        cols_to_qc = [
            var for var in self.data.data_vars
            if var.isupper() and (var not in self.data.dims) and ("_QC" not in var)
        ]
        
        for var in cols_to_qc:
            qc_array = (self.data[var].isnull().values * 9).astype(np.int8)
            self.data[f"{var}_QC"] = (self.data[var].dims, qc_array)

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        self.log("Generating diagnostics...")
        diag.generate_info(self.data)