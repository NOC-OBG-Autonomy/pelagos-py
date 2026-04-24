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
"""This module defines the base class for QC tests and a registry for QC test classes."""

import time
import os
import logging
from functools import wraps

REGISTERED_QC = {}
"""Registry of explicitly registered QC test classes."""

flag_cols = {
    0: "gray",
    1: "blue",
    2: "lightblue",
    3: "orange",
    4: "red",
    5: "gray",
    6: "gray",
    7: "gray",
    8: "cyan",
    9: "black",
}
"""Map of QC flag values to colors for diagnostics plotting."""


def register_qc(cls):
    """Decorator to mark QC tests that can be accessed by the ApplyQC step."""
    qc_name = getattr(cls, "qc_name", None)
    if qc_name is None:
        raise ValueError(
            f"QC test {cls.__name__} is missing required 'qc_name' attribute."
        )
    REGISTERED_QC[qc_name] = cls
    return cls


class BaseQC:
    """
    Initialises a base class for quality control, to be further tweaked when inherited.

    Follow the docstring format below when creating new QC tests.
    
    Target Variable: "Any" or a specific variable names (see impossible_location_test.py)
    Flag Number: "Any" or a specific ARGO flag number
    Variables Flagged: "Any" or specific variable names, possibly external to the target variable (see valid_profile_test.py)
    Your description follows here.

    Target Variable:
    Flag Number:
    Variables Flagged:
    
    """

    qc_name = None
    expected_parameters = {}
    required_variables = []
    qc_outputs = []

    def __init_subclass__(cls, **kwargs):
        """Automatically wraps the return_qc method of all subclasses to include telemetry."""
        super().__init_subclass__(**kwargs)
        
        if not getattr(cls.return_qc, "_is_telemetry_wrapped", False):
            original_return_qc = cls.return_qc
            
            @wraps(original_return_qc)
            def return_qc_with_telemetry(self, *args, **kwargs):
                start_time = time.time()
                
                result = original_return_qc(self, *args, **kwargs)
                
                if getattr(self, "diagnostics", True):
                    duration = time.time() - start_time
                    self.log(f"Execution time: {duration:.2f} seconds.")
                    
                    try:
                        import psutil
                        process = psutil.Process(os.getpid())
                        mem_info = process.memory_info()
                        self.log(f"Current memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")
                    except ImportError:
                        pass
                        
                return result
            
            return_qc_with_telemetry._is_telemetry_wrapped = True
            cls.return_qc = return_qc_with_telemetry

    def __init__(self, data, **kwargs):
        self.data = data.copy(deep=True) if data is not None else None
        self.logger = logging.getLogger(f"toolbox.qc.{self.qc_name}")

        invalid_params = set(kwargs.keys()) - set(self.expected_parameters.keys())
        if invalid_params:
            raise KeyError(
                f"Unexpected parameters for {self.qc_name}: {invalid_params}"
            )

        for k, v in kwargs.items():
            self.expected_parameters[k] = v

        for k, v in self.expected_parameters.items():
            setattr(self, k, v)

        self.flags = None

    def log(self, message):
        """Log an info-level message with QC test name prefix."""
        self.logger.info("[%s] %s", self.qc_name, message)

    def return_qc(self):
        """Representative of QC processing, to be overridden by subclasses.

        Returns
        -------
        flags : array-like
            Output QC flags for the data specific to the test.
        """
        self.flags = None  # replace with processing of some kind
        return self.flags

    def plot_diagnostics(self):
        """Representative of diagnostic plotting (optional)."""
        # Any relevant diagnostic is generated or written out here
        pass