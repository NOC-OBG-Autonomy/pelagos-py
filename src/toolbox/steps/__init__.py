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

"""
toolbox.steps
~~~~~~~~~~~~~~~~~~
This module contains the logic to dynamically discover and register step implementations,
and to instantiate them in a config-aware way.
"""

import os
import importlib
import pathlib
import yaml
from .base_step import REGISTERED_STEPS
from .base_qc import REGISTERED_QC

# Global registries
STEP_CLASSES = {}
"""Dictionary mapping step names to their implementing classes."""
QC_CLASSES = {}
"""Dictionary mapping QC test names to their implementing classes."""


def discover_steps():
    """
    Dynamically discover and import step modules from the custom directory.
    This populates the global STEP_CLASSES and QC_CLASSES registries for use elsewhere.
    """
    base_dir = pathlib.Path(__file__).parent.resolve()
    custom_dir = base_dir / "custom"
    print(f"[Discovery] Scanning for step modules in {custom_dir}")

    for py_file in custom_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Convert file path to module path
        relative_path = py_file.resolve().relative_to(base_dir)
        module_name = ".".join(("toolbox.steps",) + relative_path.with_suffix("").parts)

        try:
            print(f"[Discovery] Importing step module: {module_name}")
            importlib.import_module(module_name)
        except Exception as e:
            print(f"[Discovery] Failed to import {module_name}: {e}")

    # Populate global step class map
    STEP_CLASSES.update(REGISTERED_STEPS)
    for step_name in STEP_CLASSES:
        print(f"[Discovery] Registered step: {step_name}")

    QC_CLASSES.update(REGISTERED_QC)
    for qc_name in QC_CLASSES:
        print(f"[Discovery] Registered QC test: {qc_name}")


# Auto-discover steps when toolbox.steps is imported
discover_steps()


def create_step(step_config, context=None):
    """
    Factory to create a Step instance from a dictionary or YAML file.

    Parameters
    ----------
    step_config : dict | str
        - If dict: must contain at least {'name': <step_name>}.
        - If str: path to a YAML file describing the step.
    context : dict | None
        Shared context passed through the pipeline.

    Returns
    -------
    BaseStep
        An instantiated, config-aware step.
    """
    # --- If user passed a YAML path instead of a dict ---
    if isinstance(step_config, str):
        if not os.path.exists(step_config):
            raise FileNotFoundError(f"Step config file not found: {step_config}")
        with open(step_config, "r") as f:
            step_config = yaml.safe_load(f) or {}
        if "name" not in step_config:
            raise ValueError(f"Invalid step YAML: missing 'name' key → {step_config}")

    # --- Validate and resolve step class ---
    step_name = step_config.get("name")
    if not step_name:
        raise ValueError("Step config missing required 'name' field.")

    step_class = STEP_CLASSES.get(step_name)
    if not step_class:
        raise ValueError(
            f"Step '{step_name}' not recognized or missing @register_step. "
            f"Available: {list(STEP_CLASSES.keys())}"
        )

    # --- Instantiate the step ---
    parameters = step_config.get("parameters", {}) or {}
    diagnostics = bool(step_config.get("diagnostics", False))
    step = step_class(
        name=step_name,
        parameters=parameters,
        diagnostics=diagnostics,
        context=context,
    )

    # --- Synchronize config-mirroring store (if step supports it) ---
    # BaseStep inherits ConfigMirrorMixin
    if hasattr(step, "_parameters") and hasattr(step, "_reset_parameter_bridge"):
        step._parameters.update(step_config)
        # ensure mirrored keys are recognized
        step._reset_parameter_bridge(mirror_keys=["parameters", "diagnostics"])

    return step