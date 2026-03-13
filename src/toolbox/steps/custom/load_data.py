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

"""Class definition for loading data steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import xarray as xr
import pandas as pd
import numpy as np
import warnings


@register_step
class LoadOG1(BaseStep):
    """
    Step for loading OG1 data.

    Derived from Phyto-Phys Repo by Obsidian500

    Available Parameters:
        - file_path: Path to the OG1 data file.
        - add_meta: Boolean flag to indicate whether to add metadata.
        - add_elapsed_time: Boolean flag to indicate whether to add elapsed time.
        - add_dev_cols: Boolean flag to indicate whether to add development columns.
        - diagnostics: Boolean flag to indicate whether to generate diagnostics.
    """

    step_name = "Load OG1"

    def run(self):
        # load data from xarray
        self.data = xr.open_dataset(self.file_path)
        self.log(f"Loaded data from {self.file_path}")

        # Check that the "TIME" variable is monotonic and nanless - then make it a coordinate
        if "TIME" in self.data.coords:  #   Temporary fix for BODC OG1 files where TIME is a coord
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

        # Generate diagnostics if enabled
        if self.diagnostics:
            self.generate_diagnostics()

        # add data to context
        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        self.log(f"Generating diagnostics...")
        # self.log summary stats
        diag.generate_info(self.data)
