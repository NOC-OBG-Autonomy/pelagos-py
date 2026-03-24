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

"""Example QC test template, using parts of impossible_date_test as a skeleton."""

#### Mandatory imports ####
# from toolbox.steps.base_test import BaseQC, register_qc, flag_cols # Uncomment when implementing
from toolbox.steps.base_qc import BaseQC

#### Custom imports ####
# any additional imports required for the test go here


# @register_qc  # Uncomment when implementing
class blank_qc(BaseQC):
    """
    Example Docstring:
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    qc_name = ""
    expected_parameters = {}
    required_variables = []
    provided_variables = []
    qc_outputs = []

    def return_qc(self):
        # Access the data with self.data
        # self.flags should be an xarray Dataset with data_vars that hold the "{variable}_QC" columns produced by the test
        return self.flags

    def plot_diagnostics(self):
        # plt.show(block=True)
        pass
