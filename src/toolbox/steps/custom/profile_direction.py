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

"""Step for determining glider profile direction."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt


@register_step
class ProfileDirection(BaseStep, QCHandlingMixin):

    """
    Determine whether water profiles are ascending or descending.

    This step calculates the direction of movement through the water column
    by analysing the rate of change of pressure with respect to time. The
    median direction is computed for each profile and stored in a new
    PROFILE_DIRECTION variable.

    Attributes
    ----------
    step_name : str
        The name of this processing step: "Find Profile Direction".
    data : xarray.Dataset
        The dataset being processed, updated with PROFILE_DIRECTION and
        PROFILE_DIRECTION_QC variables.
    context : dict
        Processing context containing metadata and previous step outputs.

    Notes
    -----
    The profile direction is determined by computing the sign of the pressure
    gradient (dPRES/dTIME). Negative gradients indicate ascending profiles
    (pressure decreasing), while positive gradients indicate descending profiles
    (pressure increasing).

    The direction is calculated as::

        direction = -1 * sign(dPRES/dTIME)

    - direction = 1: Ascending profile (pressure decreasing)
    - direction = -1: Descending profile (pressure increasing)

    Examples
    --------
    In a processing configuration file, use this step as follows:

    .. code-block:: yaml

        steps:
          - name: "Find Profile Direction"
            parameters:
            diagnostics: false
    """


    step_name = "Find Profile Direction"
    required_variables = ["PROFILE_NUMBER", "PRES", "TIME"]
    provided_variables = ["PROFILE_DIRECTION"]


    def run(self):

        """
        Execute the profile direction detection algorithm.

        Processes the input data to determine whether each measurement point
        belongs to an ascending or descending profile, then propagates the
        direction classification to all measurements in each profile.

        Returns
        -------
        dict
            The updated context dictionary with the processed data containing
            new PROFILE_DIRECTION and PROFILE_DIRECTION_QC variables.

        Raises
        ------
        ValueError
            If required variables (PROFILE_NUMBER, PRES, TIME) are missing
            from the input dataset.

        Notes
        -----
        This method:

        1. Optionally filters QC flags from the data
        2. Subsets data to remove NaN values in key variables
        3. Computes pressure gradient with respect to time
        4. Calculates median direction per profile
        5. Propagates direction to all measurements
        6. Updates QC flags
        7. Generates diagnostics if enabled
        """

        self.filter_qc()

        # Subsetting the data to remove nans and find pressure rate of change
        is_nan = self.data[["PROFILE_NUMBER", "PRES", "TIME"]].isnull()
        nan_mask = is_nan["PROFILE_NUMBER"] | is_nan["PRES"] | is_nan["TIME"]
        data_subset = self.data[["PROFILE_NUMBER", "PRES", "TIME"]].where(~nan_mask, drop=True)

        # Find the gradient of pressure over time
        data_subset = data_subset.set_coords(["TIME"])
        data_subset["direction"] = -1 * np.sign(data_subset["PRES"].differentiate("TIME", datetime_unit="s"))

        # Find the median direction per profile
        direction_mapping = data_subset.groupby("PROFILE_NUMBER").map(lambda x: x["direction"].median())
        data_subset["PROFILE_DIRECTION"] = direction_mapping.sel(
            PROFILE_NUMBER=data_subset["PROFILE_NUMBER"]
        ).drop(["PROFILE_NUMBER", "TIME"])

        # Map the direction back onto self.data
        self.data["PROFILE_DIRECTION"] = xr.DataArray(
            np.full(len(self.data["N_MEASUREMENTS"]), np.nan),
            dims=["N_MEASUREMENTS"]
        )
        self.data["PROFILE_DIRECTION"][~nan_mask] = data_subset["PROFILE_DIRECTION"]


        self.reconstruct_data()
        self.update_qc()
        self.generate_qc({"PROFILE_DIRECTION_QC": ["PROFILE_NUMBER_QC", "PRES_QC", "TIME_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        """
        Create diagnostic plots of ascending and descending profiles.

        Generates a scatter plot showing all measurements coloured by profile
        direction. Ascending profiles are shown in blue, descending profiles
        in red. Pressure is plotted against time.

        Notes
        -----
        This method uses Tkinter as the matplotlib backend and requires a
        display environment. It blocks continued pipeline execution until the 
        plot window is closed.
        """
        mpl.use("tkagg")
        fig, ax = plt.subplots()

        for direction, col, label in zip([-1, 1], ['r', 'b'], ['Descending', 'Ascending']):
            plot_data = self.data[["TIME", "PRES", "PROFILE_DIRECTION"]].where(
                self.data["PROFILE_DIRECTION"] == direction
            )
            ax.plot(
                plot_data["TIME"],
                plot_data["PRES"],
                c=col,
                ls='',
                marker="o",
                label=label
            )
        ax.set(
            xlabel="TIME",
            ylabel="PRES",
            title="Profile Directions",
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        plt.show(block=True)

