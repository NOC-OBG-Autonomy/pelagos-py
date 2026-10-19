# This file is part of pelagos_py.
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
from pelagos_py.steps.base_step import BaseStep, register_step
from pelagos_py.utils.processing_utils import small_netcdf_chunk_cache
import json
import os
import threading
import time
import numpy as np
import xarray as xr


@register_step
class ExportStep(BaseStep):
    """
    Exports the the data output by the previous step.

    Parameters
    ----------
    output_path : str
        Path and file name to output to. The file extension should be included.
    export_format : str
        Either "netcdf", "csv", "hdf5" or "parquet". Defaults to "netcdf".
    compression : int
        zlib compression level for netcdf/hdf5 output: 0 turns compression off,
        1 (fastest) to 9 (smallest). Defaults to 2. Ignored for csv/parquet.
    include_intermediates : bool
        Also write the working variables steps made along the way (and their
        ``_QC``). Defaults to False.

    Every exported variable gets a ``_QC`` companion: one missing it is given
    0 (no QC) flags, 9 where the value is missing.

    Examples
    --------
    Example usage in a pipeline configuration:

    .. code-block:: yaml

        steps:
          - name: Data Export
            parameters:
                output_path: "save/my/data/here.nc"
                export_format: "netcdf"
                compression: 2
                include_intermediates: false
    """

    step_name = "Data Export"

    parameter_schema = {
        "export_format": {
            "type": str,
            "default": "netcdf",
            "options": ["csv", "netcdf", "hdf5", "parquet"],
            "description": "Export format compatible with xarray (csv/netcdf/hdf5/parquet).",
        },
        "output_path": {
            "type": str,
            "required": True,
            "description": "Path the dataset will be exported to.",
        },
        "compression": {
            "type": int,
            "default": 2,
            "min": 0,
            "max": 9,
            "description": "zlib level for netcdf/hdf5 (0 = off, 1-9). Ignored for csv/parquet.",
        },
        "include_intermediates": {
            "type": bool,
            "default": False,
            "description": "Also write the working variables steps made along the way (and their _QC).",
        },
    }

    def run(self):
        self.log(
            f"Exporting {self.parameters['export_format']} to {self.parameters['output_path']}"
        )

        # Check if the data is in the context
        self.check_data()
        data = self.context["data"]
        # Add exiting notes on QC history if available TODO: Move earlier to individual QC steps on each data variable attribute
        if "qc_history" in self.context:
            self.log("QC history found in context.", console=False)
            data.attrs["delayed_qc_history"] = json.dumps(self.context["qc_history"])

        if not self.include_intermediates:
            data = self._drop_intermediates(data)
        data = self._add_missing_qc(data)

        export_format = self.parameters["export_format"]
        output_path = self.parameters["output_path"]
        compression = self.parameters["compression"]

        # Validate the export format  # TODO: have all of these file types been tested?
        if export_format not in ["csv", "netcdf", "hdf5", "parquet"]:
            raise ValueError(
                f"Unsupported export format: {export_format}. Supported formats are: csv, netcdf, hdf5, parquet."
            )
        if not output_path:
            raise ValueError("Output path must be specified for data export.")
        # Ensure the output path is a string
        if not isinstance(output_path, str):
            raise ValueError("Output path must be a string.")

        # Build zlib encoding for netcdf/hdf5 (compression=0 disables it)
        if compression:
            if compression > 9 or compression < 0:
                raise ValueError(
                    f"Unsupported compression level: {compression}. Please specify compression from 0-9."
                )
            # Input chunks can be tiny (slow, bigger file); 1-D only, N-D could pass HDF5's 4 GB chunk limit.
            encoding = {}
            for var in data.data_vars:
                encoding[var] = {"zlib": True, "complevel": compression}
                if data[var].ndim == 1:
                    encoding[var]["chunksizes"] = (
                        max(1, min(data[var].size, 1_000_000)),
                    )
        else:
            encoding = None

        if export_format == "netcdf":
            small_netcdf_chunk_cache()

        def write():
            if export_format == "csv":
                data.to_dataframe().to_csv(output_path, index=False)
            elif export_format == "netcdf":
                data.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
            elif export_format == "hdf5":
                data.to_netcdf(output_path, engine="h5netcdf", encoding=encoding)
            elif export_format == "parquet":
                data.to_dataframe().to_parquet(output_path, index=False)

        started = time.time()
        self._write_with_progress(write, output_path, data.nbytes)
        size_mb = os.path.getsize(output_path) / 1024**2
        self.log(
            f"Exported {size_mb:.1f} MB to {output_path} in {time.time() - started:.1f}s"
        )
        return self.context

    def _drop_intermediates(self, data):
        names = self.context.get("intermediate_variables", [])
        dropped = [
            name
            for var in names
            for name in (var, f"{var}_QC")
            if name in data.data_vars
        ]
        if not dropped:
            return data
        self.log(
            f"Leaving out {len(dropped)} intermediate variable(s); "
            "set include_intermediates to keep them."
        )
        self.log(f"Left out: {', '.join(dropped)}", console=False)
        return data.drop_vars(dropped)

    def _add_missing_qc(self, data):
        missing = [
            var
            for var in data.data_vars
            if var.isupper()
            and not var.endswith("_QC")
            and var not in data.dims
            and f"{var}_QC" not in data.data_vars
        ]
        if not missing:
            return data
        self.log(
            f"{len(missing)} variable(s) have no QC; writing unchecked (0) flags for them."
        )
        self.log(f"Unchecked QC added for: {', '.join(missing)}", console=False)
        flags = xr.where(data[missing].isnull(), 9, 0).astype(np.int8)
        return data.assign(flags.rename({var: f"{var}_QC" for var in missing}))

    def _write_with_progress(self, write, output_path, nbytes):
        # writers report no progress, so watch the file grow against the in-memory size
        total_mb = max(1, int(nbytes / 1024**2))
        bar = self.log_progress(total=total_mb, desc="Writing", unit="MB")
        done = threading.Event()

        def watch():
            while not done.wait(0.25):
                try:
                    mb = int(os.path.getsize(output_path) / 1024**2)
                except OSError:
                    continue
                if mb > bar.n:
                    bar.update(min(mb, total_mb - 1) - bar.n)

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            write()
            done.set()
            watcher.join()
            bar.update(total_mb - bar.n)
        finally:
            done.set()
            bar.close()

    def generate_diagnostics(self):
        # nothing to plot for an export
        return
