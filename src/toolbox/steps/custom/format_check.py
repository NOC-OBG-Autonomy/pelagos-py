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

"""Checks the format of the file and returns the results to a written file."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step

#### Custom imports ####
from compliance_checker.runner import ComplianceChecker, CheckSuite
from pathlib import Path


@register_step
class FormatCheck(BaseStep):
    """
    Runs IOOS file format compliance checker and produces results file for reporting purposes.

    Does not run directly on a dataset, follows own import/loading routines.

    Parameters
    ----------
    src : path or str
        The path to the source file.
    standards : list of str
        A list of the standards to be tested. For example, ['cf', 'og']
    """
    step_name = "Format Checker"

    def run(self):

        check_suite = CheckSuite()
        check_suite.load_all_available_checkers()

        src = self.parameters.get("src")
        cnames = self.parameters.get("standards")

        #   If ran after loading data, the filename stem should be saved in the global pipeline params
        fname = self.context.get("global_parameters", {}).get("filename_core") or Path(src.strip("*.nc")).stem
        f_out = self.context["global_parameters"]["out_directory"] + fname + "_check.rst"

        #   return_value: True if passes specified checks. errors: True if there were any errors running the CC.
        return_value, errors = ComplianceChecker.run_checker(
            src,
            checker_names = cnames,
            verbose = True,
            criteria = "lenient",
            output_filename = f_out,
            output_format = "text",
        )

        if return_value == False:
            self.log_warn(
                f"File {fname} did not pass the file format compliance checker. See the outfile ({f_out})."
            )
            if self.parameters.get("proceed_on_fail") == False:
                self.log("Halting pipeline, 'proceed on fail' parameter from config is set to False.")
                raise RuntimeError("Compliance check step failed. Check input files and log for details.")

        if errors:
            self.log_warn(
                f"Errors occurred when running the file format compliance checker. See the outfile ({f_out}).")

        return self.context
