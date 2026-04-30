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
"""This module defines the base class for pipeline steps and configurations."""

from toolbox.utils.config_mirror import ConfigMirrorMixin
import warnings
import logging
import os

REGISTERED_STEPS = {}
"""Registry of explicitly registered step classes."""


def register_step(cls):
    """Decorator to mark a step class for inclusion in the pipeline."""
    step_name = getattr(cls, "step_name", None)
    if step_name is None:
        raise ValueError(
            f"Class {cls.__name__} is missing required 'step_name' attribute."
        )
    REGISTERED_STEPS[step_name] = cls
    return cls


class BaseStep(ConfigMirrorMixin):
    """
    Base class for pipeline steps with config-mirroring support.
    Every concrete subclass (registered via @register_step) inherits this.
    """

    def __init__(self, name, parameters=None, diagnostics=False, context=None):
        # === Core behaviour (same as before) ===
        self.name = name
        self.parameters = parameters or {}
        self.diagnostics = diagnostics
        self.context = context or {}

        # Get child logger initialized in pipeline.py
        self.logger = logging.getLogger(f"toolbox.pipeline.step.{self.name}")

        # === Initialise config mirror system ===
        self._init_config_mirror()
        # canonical parameters go in private store
        self._parameters = {
            "name": self.name,
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
        }
        # mirror parameters & diagnostics as attributes
        self._reset_parameter_bridge(mirror_keys=["parameters", "diagnostics"])

        # expose param keys as attributes (for user convenience)
        for key, value in self.parameters.items():
            setattr(self, key, value)

        # Continue method resolution order
        super().__init__()

    def run(self):
        """To be implemented by subclasses."""
        raise NotImplementedError(f"Step '{self.name}' must implement a run() method.")
        return self.context

    def generate_diagnostics(self):
        """Hook for diagnostics (optional)."""
        pass

    def log(self, message):
        """Log an info-level message with step name prefix."""
        self.logger.info("[%s] %s", self.name, message)

    def log_warn(self, message, warning_type=UserWarning):
        """Log a warning-level message with step name prefix."""
        self.logger.warning("[%s] %s", self.name, message)

    def check_data(self):
        """Check for data in context for transformer steps."""
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")

    # ----------- Config Handling -----------

    def update_parameters(self, **kwargs):
        """
        Update parameter values both in attributes and in private store.
        Example:
            self.update_parameters(file_path='newfile.nc', add_meta=False)
        """
        for k, v in kwargs.items():
            self.parameters[k] = v
            setattr(self, k, v)
        self._parameters["parameters"] = self.parameters

    def generate_config(self):
        """Return this step's config dict (suitable for saving to YAML)."""
        self._sync_attributes_to_parameters()
        return dict(self._parameters)

    def save_config(self, path: str | None = None):
        """Save this step's config to YAML (for standalone debugging)."""
        import yaml, os

        cfg = self.generate_config()
        if path is None:
            safe_name = self.name.replace(" ", "_").lower()
            path = f"{safe_name}_step.yaml"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[{self.name}] Step config saved → {path}")
        return cfg
