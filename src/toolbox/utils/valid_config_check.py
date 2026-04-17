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

from toolbox.steps import STEP_CLASSES

STANDARD_VARIABLES = {
    "TIME", 
    "LATITUDE", 
    "LONGITUDE", 
    "PRES", 
    "TEMP", 
    "CNDC"
}

def check_pipeline_variables(steps_list, logger, available_vars=None):
    if available_vars is None:
        logger.info("Checking pipeline variable requirements...")
        available_vars = set(STANDARD_VARIABLES)

    for step_config in steps_list:
        step_name = step_config["name"]
        
        step_class = STEP_CLASSES.get(step_name)
        if not step_class:
            continue
            
        parameters = step_config.get("parameters", {})

        # Check for missing config parameters
        expected_params = getattr(step_class, "expected_parameters", [])
        if isinstance(expected_params, dict):
            expected_params = list(expected_params.keys())
            
        missing_params = [p for p in expected_params if p not in parameters]
        if missing_params:
            missing_str = ", ".join(missing_params)
            logger.error("Validation Failed: '%s' is missing required config parameters: %s.", step_name, missing_str)
            raise ValueError(f"Missing config parameters for '{step_name}': {missing_str}.")
        
        req_vars = list(getattr(step_class, "required_variables", []))
        provided_vars = getattr(step_class, "provided_variables", []) + getattr(step_class, "qc_outputs", [])
        
        if step_name == "Find Profiles":
            depth_col = parameters.get("depth_column", "DEPTH")
            if depth_col not in req_vars:
                req_vars.append(depth_col)

        missing_vars = [req for req in req_vars if req not in available_vars]
        
        if missing_vars:
            missing_str = ", ".join(missing_vars)
            logger.error("Validation Failed: '%s' requires %s but they are not provided.", step_name, missing_str)
            raise ValueError(f"Missing variables for '{step_name}': {missing_str}. Please add suitable steps beforehand.")
        
        available_vars.update(provided_vars)
        available_vars.update(parameters.get("to_derive", []))
        available_vars.update(parameters.get("qc_outputs", []))
            
    if steps_list:
        logger.info("Pipeline variable check successful.")
        
    return True