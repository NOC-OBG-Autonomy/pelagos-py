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

"""Example step template. Copy and populate this example, which will inherit additional functionality from BaseStep."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####


@register_step
class BlankStep(BaseStep, QCHandlingMixin):

    step_name = "Blank Step"
    required_variables = []
    provided_variables = []


    def run(self):
        self.filter_qc()

        # EXAMPLE: self.data["C"] = self.data["A"] * self.data["B"]

        self.reconstruct_data()
        self.update_qc()

        # If a new variable was added, use self.generate_qc()
        # EXAMPLE: self.generate_qc({"C_QC": ["A_QC", "B_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass
