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

"""Class definition for exporting data steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
import json


@register_step
class ExportStep(BaseStep):
    """
    Step to export data in various formats.
    """
    step_name = "Data Export"

    parameter_schema = {
        "export_format": {
            "type": str, 
            "default": "netcdf", 
            "description": "Format to export data (csv, netcdf, hdf5, parquet)"
        },
        "output_path": {
            "type": str, 
            "default": "./pipeline_output/exported_data.nc", 
            "description": "Path to save the exported data"
        },
        "compression_level": {
            "type": int,
            "default": 6,
            "description": "Zlib compression level from 0 (off), 1 (fastest) to 9 (smallest)"
        }
    }

    def run(self):
        self.log(
            f"Exporting data in {self.export_format} format to {self.output_path}"
        )

        # Check if the data is in the context
        self.check_data()
        data = self.context["data"]
        
        # Add exiting notes on QC history if available TODO: Move earlier to individual QC steps on each data variable attribute
        if "qc_history" in self.context:
            self.log("QC history found in context.")
            data.attrs["delayed_qc_history"] = json.dumps(self.context["qc_history"])

        # Validate the export format
        if self.export_format not in ["csv", "netcdf", "hdf5", "parquet"]:
            raise ValueError(
                f"Unsupported export format: {self.export_format}. Supported formats are: csv, netcdf, hdf5, parquet."
            )
            
        if not self.output_path:
            raise ValueError("Output path must be specified for data export.")
            
        # Ensure the output path is a string
        if not isinstance(self.output_path, str):
            raise ValueError("Output path must be a string.")

        # Export data based on the specified format
        if self.export_format == "csv":
            data.to_dataframe().to_csv(self.output_path)
        elif self.export_format == "netcdf":
            comp_level = getattr(self, "compression_level", 6)
            
            if comp_level > 0:
                self.log(f"Applying lossless NetCDF compression at level {comp_level}.")
                encoding_dict = {
                    var_name: {"zlib": True, "complevel": comp_level}
                    for var_name in data.variables
                }
                data.to_netcdf(self.output_path, engine="netcdf4", encoding=encoding_dict)
            else:
                self.log("Exporting NetCDF without compression.")
                data.to_netcdf(self.output_path, engine="netcdf4")
                
        elif self.export_format == "hdf5":
            data.to_netcdf(self.output_path, engine="h5netcdf")
        elif self.export_format == "parquet":
            data.to_dataframe().to_parquet(self.output_path)
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")
            
        self.log(f"Data exported successfully to {self.output_path}")
        
        if getattr(self, "diagnostics", False):
            self.generate_diagnostics()
            
        return self.context

    def generate_diagnostics(self):
        """
        Generate diagnostics for the export step natively.
        """
        self.log(f"Generating diagnostics for {self.step_name}")
        diag.generate_diagnostics(self.context, self.step_name)
        self.log("Diagnostics generated successfully.")