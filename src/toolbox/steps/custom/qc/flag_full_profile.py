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

"""QC test to flag entire glider profiles based on number of bad flags."""

#### Mandatory imports ####
import numpy as np
from toolbox.steps.base_qc import BaseQC, register_qc, flag_cols

#### Custom imports ####
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib


@register_qc
class flag_full_profile(BaseQC):
    """
    Target Variable: Any
    Flag Number: 4
    Variables Flagged: Any
    Checks the number of bad (4) flags per profile. If it
    exceeds the user threshold then all points in the profile
    are flagged.

    EXAMPLE
    -------
    - name: "Apply QC"
      parameters:
        qc_settings: {
            "flag_full_profile": {
              "check_vars": {"PRES": 10, "CHLA": 20},
            }
        }
      diagnostics: true
    """

    qc_name = "flag full profile"
    required_variables = ["PROFILE_NUMBER"]
    provided_variables = []


    # Specify if test target variable is user-defined (if True, __init__ has to be redefined)
    dynamic = True

    def __init__(self, data, **kwargs):
        # Check the necessary kwargs are available
        required_kwargs = {"check_vars"}
        if not required_kwargs.issubset(set(kwargs.keys())):
            raise KeyError(
                f"{required_kwargs - set(kwargs.keys())} are missing from {self.qc_name} settings"
            )

        # Specify the tests paramters from kwargs (config)
        self.expected_parameters = {
            k: v for k, v in kwargs.items() if k in required_kwargs
        }
        self.required_variables = (
            list(self.expected_parameters["check_vars"].keys())
            + [f"{k}_QC" for k in self.expected_parameters["check_vars"].keys()]
            + ["PROFILE_NUMBER"]
        )

        if data is not None:
            self.data = data.copy(deep=True)

        for k, v in self.expected_parameters.items():
            setattr(self, k, v)

        self.flags = None

    def return_qc(self):
        # TODO: Add support for flagging if threshold is a mix of 3 (questionable) and 4 (definitely bad) flags
        # Subset the data
        self.data = self.data[self.required_variables]

        for var, threshold in self.check_vars.items():
            flag_counts = (
                (self.data[f"{var}_QC"] == 4).groupby(self.data["PROFILE_NUMBER"]).sum()
            )  # Default to flag 4 (definitely bad)
            bad_profiles = flag_counts.where(flag_counts >= threshold, drop=True)[
                "PROFILE_NUMBER"
            ]
            self.data[f"{var}_QC"] = xr.where(
                self.data[f"PROFILE_NUMBER"].isin(bad_profiles),
                4,
                self.data[f"{var}_QC"],
            )

        # Select just the flags
        self.flags = self.data[
            [var_qc for var_qc in self.data.data_vars if "_QC" in var_qc]
        ]

        return self.flags

    def plot_diagnostics(self):
        matplotlib.use("tkagg")

        # Plot the QC output
        n_plots = len(self.check_vars.keys())
        fig, axs = plt.subplots(nrows=n_plots, figsize=(8, 4 * n_plots), dpi=200)
        if n_plots == 1:
            axs = [axs]

        for ax, var in zip(axs, self.check_vars.keys()):
            for i in range(10):
                # Plot by flag number
                plot_data = self.data[[var, "N_MEASUREMENTS"]].where(
                    self.data[f"{var}_QC"] == i, drop=True
                )

                if len(plot_data[var]) == 0:
                    continue

                # Plot the data
                ax.plot(
                    plot_data["N_MEASUREMENTS"],
                    plot_data[var],
                    c=flag_cols[i],
                    ls="",
                    marker="o",
                    label=f"{i}",
                )

            ax.set(
                xlabel="Index",
                ylabel=var,
                title=f"{var} Flag Full Profile",
            )

            ax.legend(title="Flags", loc="upper right")

        fig.tight_layout()
        plt.show(block=True)
