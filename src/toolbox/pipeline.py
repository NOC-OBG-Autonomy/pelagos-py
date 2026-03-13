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

"""Pipeline class definition to handle configuration and step execution."""

import yaml
import pandas as pd
import numpy as np
import xarray as xr
import os
import logging
import datetime as _dt
from graphviz import Digraph

from toolbox.utils.config_mirror import ConfigMirrorMixin

from toolbox.steps import (
    create_step,
    STEP_CLASSES,
    STEP_DEPENDENCIES
)

_PIPELINE_LOGGER_NAME = "toolbox.pipeline"
"""Global logger name for the pipeline. Used to create child loggers for steps."""

def _setup_logging(out_dir=None, log_file=None, level=logging.INFO):
    """
    Set up logging for the entire pipeline.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file. If provided, logs will be written to this file.
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(_PIPELINE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger  # already configured

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if specified
    if log_file:
        log_file = os.path.abspath(os.path.join(out_dir or ".", log_file))        # absolute path
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logging to file: %s", log_file)

    return logger

class Pipeline(ConfigMirrorMixin):
    """
    Pipeline that manages a sequence of processing steps.
    
    Config-aware pipeline that can:
      - Load config YAML into private self._parameters
      - Keep global_parameters mirrored to _parameters['pipeline']
      - Build, run, and export steps as before

    Parameters
    ----------
    ConfigMirrorMixin : Class
        Class to handle configuration
    
    """

    def __init__(self, config_path=None):
        """
        Initialize pipeline with optional config file.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file.
        """
        self.steps = []  # hierarchical step configs
        self.graph = Digraph("Pipeline", format="png", graph_attr={"rankdir": "TB"})
        self.global_parameters = {}  # mirrors _parameters["pipeline"]
        self._context = None

        # initialise config mirror system
        self._init_config_mirror()

        if config_path:
            self.load_config_from_file(config_path, mirror_keys=["pipeline"])
            # set convenience alias for user-facing access
            self.global_parameters = self._parameters.get("pipeline", {})
            # build steps from loaded config
            self.logger = _setup_logging(self.global_parameters.get("out_directory"),
                                         self.global_parameters.get("log_file"))
            self.build_steps(self._parameters.get("steps", []))
            self.logger.info("Pipeline initialised")

    def build_steps(self, steps_config, parent_name=None):
        """
        Recursively build steps from configuration.

        Individual steps, including parameters and diagnostics, are saved to self.steps using add_step() for other functions.
        If "substeps" is a part of the defined step, it recursively builds that child step as a field of the parent.

        Parameters
        ----------
        steps_config : list of dict
            List of step configurations.
        parent_name : str, optional
            Name of the parent step, if any.
        """
        self.logger.info("Assembling steps to run from config.")
        for step in steps_config:
            REQUIRED_STEPS = STEP_DEPENDENCIES.get(step["name"], [])
            for required_step in REQUIRED_STEPS:
                if required_step not in STEP_CLASSES:
                    raise ValueError(
                        f"Required step '{required_step}' for '{step['name']}' is not found."
                    )
            self.add_step(
                step_name=step["name"],
                parameters=step.get("parameters", {}),
                diagnostics=step.get("diagnostics", False),
                parent_name=parent_name,
                run_immediately=False,
            )
            if "substeps" in step:
                self.build_steps(step["substeps"], parent_name=step["name"])

    def add_step(
        self,
        step_name,
        parameters=None,
        diagnostics=False,
        parent_name=None,
        run_immediately=False,
    ):
        """
        Dynamically adds a step and optionally runs it immediately.

        Parameters
        ----------
        step_name : str
            Name of the step to add.
        parameters : dict, optional
            Parameters for the step.
        diagnostics : bool, optional
            Whether to enable diagnostics for this step.
        parent_name : str, optional
            Name of the parent step, if any.
        run_immediately : bool, optional
            Whether to run the step immediately after adding it.

        Raises
        ------
        ValueError
            If the step name is not recognized or a specified parent step is not found.
        """
        if step_name not in STEP_CLASSES:
            raise ValueError(
                f"Step '{step_name}' is not recognized or missing @register_step."
            )

        step_config = {
            "name": step_name,
            "parameters": parameters or {},
            "diagnostics": diagnostics,
            "substeps": [],
        }

        if parent_name:
            parent = self._find_step(self.steps, parent_name)
            if parent:
                parent["substeps"].append(step_config)
            else:
                raise ValueError(f"Parent step '{parent_name}' not found.")
        else:
            self.steps.append(step_config)

        self.logger.info(f"Step '{step_name}' added successfully!")

        if run_immediately:
            self.logger.info(f"Running step '{step_name}' immediately.")
            self._context = self.execute_step(step_config, self._context)

    def _find_step(self, steps_list, step_name):
        """
        Recursively find a step by name.

        Parameters
        ----------
        steps_list : list of dict
            List of step configurations.
        step_name : str
            Name of the step to find.
        """
        for step in steps_list:
            if step["name"] == step_name:
                return step
            found = self._find_step(step.get("substeps", []), step_name)
            if found:
                return found
        return None

    def execute_step(self, step_config, _context):
        """
        Executes a single step.

        Parameters
        ----------
        step_config : dict
            Configuration for the step to execute.
        _context : dict
            Current context to pass to the step.
        """
        step_context = _context.copy() if _context else {}
        step_context["global_parameters"] = self.global_parameters
        step = create_step(step_config, step_context)
        self.logger.info(f"Executing: {step.name}")
        return step.run()

    def run_last_step(self):
        """
        Runs only the most recently added step based on the index in self.steps.
        """
        if not self.steps:
            self.logger.info("No steps to run.")
            return
        last_step = self.steps[-1]
        self.logger.info(f"Running last step: {last_step['name']}")
        self._context = self.execute_step(last_step, self._context)

    def run(self):
        """
        Runs the entire pipeline.

        If visualisation is specified in the configuration parameters, a visualisation
        of the pipeline execution will be generated.
        """
        for step in self.steps:
            self._context = self.execute_step(step, self._context)

        if self.global_parameters.get("visualisation", False):
            self.visualise_pipeline()

    def visualise_pipeline(self):
        """
        Generates a visualisation of the pipeline execution.
        """
        self.graph.clear()

        def add_to_graph(step_config, parent_name=None, step_order=None):
            """Add a step to the graph, intended for recursive use."""
            step_name = step_config["name"]
            diagnostics = step_config.get("diagnostics", False)
            color = "red" if diagnostics else "black"
            self.graph.node(
                step_name,
                step_name,
                color=color,
                style="filled",
                fillcolor="lightblue" if diagnostics else "white",
            )
            if parent_name:
                self.graph.edge(parent_name, step_name)
            if step_order and len(step_order) > 1:
                for i in range(len(step_order) - 1):
                    self.graph.edge(step_order[i], step_order[i + 1])
            substep_order = []
            for substep in step_config.get("substeps", []):
                substep_order.append(substep["name"])
                add_to_graph(substep, parent_name=step_name, step_order=substep_order)

        for step in self.steps:
            step_order = [step["name"]]
            add_to_graph(step, step_order=step_order)
        self.graph.render("pipeline_visualisation", view=True)

    def generate_config(self):
        """
        Generate a configuration dictionary from the current pipeline setup.
        
        returns
        -------
        dict
            Configuration dictionary of the current pipeline.
        """
        cfg = {
            "pipeline": self.global_parameters,
            "steps": self.steps,
        }
        # Keep private config in sync
        self._parameters.update(cfg)
        return cfg

    def export_config(self, output_path="generated_pipeline.yaml"):
        """
        Write current config to file (respects private _parameters)
        
        parameters
        ----------
        output_path : str
            Path to save the exported configuration YAML file.
        
        returns
        -------
        dict
            Configuration dictionary of the current pipeline.
        """
        cfg = self.generate_config()
        with open(output_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.logger.info(f"Pipeline config exported → {output_path}")
        return cfg

    def save_config(self, path="pipeline_config.yaml"):
        """
        Save the canonical private config (same as manager.save_config).

        parameters
        ----------
        path : str
            Path to save the exported configuration YAML file.
        """
        # ensure _parameters is up to date
        self._parameters.update(self.generate_config())
        super().save_config(path)