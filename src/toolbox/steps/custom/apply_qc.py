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

"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
from toolbox.steps import QC_CLASSES

#### Custom imports ####
import xarray as xr
import numpy as np
import pandas as pd
import json


@register_step
class ApplyQC(BaseStep):
    """
    Step to apply quality control tests to the dataset.

    Inherits properties from BaseStep (see base_step.py).
    """

    step_name = "Apply QC"

    def organise_flags(self, new_flags):
        """
        Method for taking in new flags (new_flags) and cross checking against existing flags (self.flag_store), 
        including upgrading flags when necessary, following ARGO flagging standards.
        """

        # Define combinatrix for handling flag upgrade behaviour
        qc_combinatrix = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 1, 2, 3, 4, 5, 1, 1, 8, 9],
                [2, 2, 2, 3, 4, 5, 2, 2, 8, 9],
                [3, 3, 3, 3, 4, 3, 3, 3, 3, 9],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 9],
                [5, 5, 5, 3, 4, 5, 5, 5, 8, 9],
                [6, 1, 2, 3, 4, 5, 6, 6, 8, 9],
                [7, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [8, 8, 8, 3, 4, 8, 8, 8, 8, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            ],
            dtype=np.int8
        )

        # Update existing flag columns
        flag_columns_to_update = set(new_flags.data_vars) & set(
            self.flag_store.data_vars
        )
        for column_name in flag_columns_to_update:
            self.flag_store[column_name][:] = qc_combinatrix[
                self.flag_store[column_name], new_flags[column_name]
            ]

        # Add new QC flag columns if they dont already exist
        flag_columns_to_add = set(new_flags.data_vars) - set(self.flag_store.data_vars)
        if len(flag_columns_to_add) > 0:
            for column_name in flag_columns_to_add:
                self.flag_store[column_name] = new_flags[column_name]

    def _sanitise_flags(self, flag_array):
        """
        Converts an array of mixed/invalid flags to valid 0-9 integers using vectorized operations.
        
        This method replaces Python loops with fast NumPy/Pandas operations to handle
        large datasets (millions of rows) efficiently.
        """
        # Convert everything to numeric (coercing strings/errors to NaN)
        numeric_flags = pd.to_numeric(flag_array, errors='coerce')
        
        # Identify values that are NaN or outside the valid ARGO 0-9 range
        # np.mod(x, 1) != 0 identifies non-integers like 1.5
        invalid_mask = (
            np.isnan(numeric_flags) | 
            (numeric_flags < 0) | 
            (numeric_flags > 9) | 
            (np.mod(numeric_flags, 1) != 0)
        )
        
        errors_found = invalid_mask.any()
        
        # Create a clean integer array, defaulting invalid values to 0
        clean_vals = np.where(invalid_mask, 0, numeric_flags).astype(np.int8)
        
        return clean_vals, errors_found
    

    def run(self):
        """
        Run the Apply QC step.
       """
        if len(self.qc_settings.keys()) == 0:
            raise KeyError("[Apply QC] No QC operations were specified.")
        else:
            invalid_requests = set(self.qc_settings.keys()) - set(QC_CLASSES.keys())
            if invalid_requests:
                raise KeyError(f"[Apply QC] Requested QC tests not found: {invalid_requests}")
                
        queued_qc = [QC_CLASSES.get(key) for key in self.qc_settings.keys()]

        self.check_data()
        data = self.context["data"].copy(deep=True)
        qc_history = self.context.setdefault("qc_history", {})

        all_required_variables = set({})
        test_qc_outputs_cols = set({})
        for test in queued_qc:
            if hasattr(test, "dynamic"):
                test_instance = test(None, **self.qc_settings[test.qc_name])
                all_required_variables.update(test_instance.required_variables)
                test_qc_outputs_cols.update(test_instance.qc_outputs)
            else:
                all_required_variables.update(test.required_variables)
                test_qc_outputs_cols.update(test.qc_outputs)
            
            if not set(all_required_variables).issubset(set(data.keys())):
                raise KeyError(f"[Apply QC] Missing variables: {set(all_required_variables) - set(data.keys())}")
        
        # Only fetch flags that exist and have the correct measurement dimension
        existing_flags = [
            flag_col for flag_col in data.data_vars 
            if flag_col in test_qc_outputs_cols and "N_MEASUREMENTS" in data[flag_col].dims
        ]
        
        self.flag_store = xr.Dataset(coords={"N_MEASUREMENTS": data["N_MEASUREMENTS"]})
        sanitised_summary = []

        if len(existing_flags) > 0:
            self.log(f"Found existing flags columns {set(existing_flags)} in data.")
            for flag_col in existing_flags:
                clean_flags, errors = self._sanitise_flags(data[flag_col].values)
                if errors:
                    sanitised_summary.append(flag_col)
                self.flag_store[flag_col] = (("N_MEASUREMENTS",), clean_flags)
        
        # Sanitise other QC columns that aren't subject to the current tests
        other_existing_qc = set([
            var for var in data.data_vars 
            if var.endswith("_QC") and "N_MEASUREMENTS" in data[var].dims
        ]) - set(test_qc_outputs_cols)
        
        if any(other_existing_qc):
            for flag_col in other_existing_qc:
                clean_flags, errors = self._sanitise_flags(data[flag_col].values)
                if errors:
                    sanitised_summary.append(flag_col)
                data[flag_col] = (("N_MEASUREMENTS",), clean_flags)

        if sanitised_summary:
            self.log_warn(f"Sanitised invalid flags to 0 in {len(sanitised_summary)} existing _QC variables.")

        # Initialise missing _QC columns
        measurement_vars = [v for v in data.data_vars if "N_MEASUREMENTS" in data[v].dims and not v.endswith("_QC")]
        for var in measurement_vars:
            qc_col = f"{var}_QC"
            if qc_col not in data.data_vars and qc_col not in test_qc_outputs_cols:
                clean_flags = xr.where(data[var].isnull(), 9, 0).astype(np.int8).values
                data[qc_col] = (("N_MEASUREMENTS",), clean_flags)

        # Build placeholders for variables about to be tested
        mia_qc = test_qc_outputs_cols - set(data.data_vars)
        base = [var[:-3] for var in mia_qc]
        if not set(base).issubset(set(data.keys())):
            raise KeyError(f"[Apply QC] Data missing: {set(base) - set(data.keys())}")
        
        data_subset = data[base]
        masks = xr.where(data_subset.isnull(), 9, 0).astype(np.int8)
        masks = masks.rename({var: f"{var}_QC" for var in base})
        self.flag_store.update(masks)

        for qc_qc_name, qc_test_params in self.qc_settings.items():
            self.log(f"Applying: {qc_qc_name}")
            qc_test_instance = QC_CLASSES[qc_qc_name](data, **qc_test_params)
            returned_flags = qc_test_instance.return_qc().astype(np.int8) 
            self.organise_flags(returned_flags)

            for flagged_var in returned_flags.data_vars:
                var_flags = returned_flags[flagged_var]
                percent_flagged = (var_flags.to_numpy() != 0).sum() / len(var_flags)
                
                qc_history.setdefault(flagged_var, []).append((qc_qc_name, percent_flagged))

                parent_attrs = data[flagged_var[:-3]].attrs                
                attrs = self.flag_store[flagged_var].attrs
                base_long_name = parent_attrs.get('long_name', flagged_var[:-3])
                base_standard_name = parent_attrs.get('standard_name', flagged_var[:-3].lower())

                attrs["quality_control_conventions"] = "Argo standard flags"
                attrs["valid_min"] = 0
                attrs["valid_max"] = 9
                attrs["flag_values"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                attrs["flag_meanings"] = "NO_QC, GOOD, PROB_GOOD, PROB_BAD, BAD, VALUE_CHANGED, NOT_USED, NOT_USED, ESTIMATED, MISSING"
                attrs["long_name"] = f"{base_long_name} quality flag"
                attrs["standard_name"] = f"{base_standard_name}_flag"
                
                attr_test = qc_qc_name.replace(" ", "_").lower()
                # Fast counting for attributes
                counts = {i: int(np.sum(var_flags.to_numpy() == i)) for i in range(10)}
                attrs[f"{attr_test}_flag_cts"] = json.dumps(counts)

            if self.diagnostics:
                qc_test_instance.plot_diagnostics()

        for flag_column in self.flag_store.data_vars:
            data[flag_column] = (("N_MEASUREMENTS",), self.flag_store[flag_column].to_numpy())
            data[flag_column].attrs = self.flag_store[flag_column].attrs.copy()
            
        self.context["data"] = data
        self.context["qc_history"] = qc_history

        return self.context