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

"""QC test to flag entire profiles that fail length or depth range requirements."""

#### Mandatory imports ####
import numpy as np
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib
import pandas as pd


@register_qc
class ValidProfileQC(BaseQC):
    """
    Target Variable: PROFILE_NUMBER
    Flag Number: 4
    Variables Flagged: PROFILE_NUMBER
    
    Checks each profile to ensure it meets a minimum number of valid data points 
    and spans a required depth range. Profiles failing these criteria are flagged as bad.

    EXAMPLE
    -------
    - name: "Apply QC"
      parameters:
        qc_settings:
            valid profile qc:
              profile_length: 50
              depth_range: [-1000, 0]
      diagnostics: false
    """

    qc_name = "valid profile qc"
    required_variables = ["PROFILE_NUMBER"]
    provided_variables = ["PROFILE_NUMBER_QC"]

    dynamic = True

    parameter_schema = {
        "profile_length": {
            "type": int,
            "default": 50,
            "description": "Minimum number of data points required for a valid profile."
        },
        "depth_range": {
            "type": list,
            "default": [-1000, 0],
            "description": "Required depth range [min, max] that the profile must span."
        },
        "depth_column": {
            "type": str,
            "default": "PRES",
            "description": "Depth or pressure column name. Defaults to PRES."
        }
    }

    def __init__(self, data, **kwargs):
        self.profile_length = kwargs.get("profile_length", 50)
        self.depth_range = kwargs.get("depth_range", [-1000, 0])
        self.depth_col = kwargs.get("depth_column", "PRES")
        
        self.required_variables = ["PROFILE_NUMBER", self.depth_col]
        
        if data is not None:
            self.data = data.copy(deep=True)

        self.flags = None

    def return_qc(self):
        self.data = self.data[self.required_variables]
        
        df = self.data.to_dataframe()
        bad_profiles = []
        
        for pid, group in df.dropna(subset=["PROFILE_NUMBER"]).groupby("PROFILE_NUMBER"):
            is_valid = True
            
            if len(group) < self.profile_length:
                is_valid = False
                
            elif self.depth_range is not None and len(self.depth_range) == 2:
                min_req_depth, max_req_depth = sorted(self.depth_range)
                actual_min = group[self.depth_col].min()
                actual_max = group[self.depth_col].max()
                
                if actual_min > min_req_depth or actual_max < max_req_depth:
                    is_valid = False
                    
            if not is_valid:
                bad_profiles.append(pid)

        self.data["PROFILE_NUMBER_QC"] = xr.where(
            self.data["PROFILE_NUMBER"].isin(bad_profiles),
            4,
            0
        )

        self.flags = self.data[["PROFILE_NUMBER_QC"]]
        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        df = self.data.to_dataframe().reset_index()
        df_profiles = df.dropna(subset=["PROFILE_NUMBER"])
        
        if df_profiles.empty:
            print("No profiles found to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        good_mask = df_profiles["PROFILE_NUMBER_QC"] != 4
        bad_mask = df_profiles["PROFILE_NUMBER_QC"] == 4

        if good_mask.any():
            ax.plot(
                df_profiles.loc[good_mask, "N_MEASUREMENTS"],
                df_profiles.loc[good_mask, self.depth_col],
                c="tab:blue",
                ls="",
                marker="o",
                markersize=2,
                label="Valid Profiles",
                alpha=0.7
            )

        if bad_mask.any():
            ax.plot(
                df_profiles.loc[bad_mask, "N_MEASUREMENTS"],
                df_profiles.loc[bad_mask, self.depth_col],
                c="tab:red",
                ls="",
                marker="x",
                markersize=4,
                label="Failed Profiles",
                alpha=0.9
            )

        if self.depth_range:
            ax.axhline(self.depth_range[0], color="black", linestyle="--", alpha=0.5, label="Depth Limits")
            ax.axhline(self.depth_range[1], color="black", linestyle="--", alpha=0.5)

        ax.set_ylabel(self.depth_col)
        ax.set_xlabel("Measurement Index")
        ax.set_title("Valid Profile QC Diagnostics")
        ax.legend(loc="upper right")
        
        fig.tight_layout()
        plt.show(block=True)