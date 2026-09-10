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

"""Cheap metadata probe of an OG1 NetCDF file, shared by the config validator,
the config builder and the dashboard: variable names, units, which float
variables are entirely NaN, and a median for a few variables whose *values*
decide how they must be treated (e.g. CNDC unit mislabelling)."""

import json
import os
import subprocess
import sys
from pathlib import Path

# Variables whose median is needed to tell a unit/naming problem from real data.
MEDIAN_VARIABLES = ("CNDC", "BBP700", "BETA_BACKSCATTERING700")

# Runs in a subprocess: some netCDF4/HDF5 + h5py combinations segfault the whole
# interpreter on open, which try/except can't catch. netCDF4 rather than xarray
# for import time; slices so a variable with real data settles on its first one.
_SCRIPT = r"""
import sys, json
import numpy as np
import netCDF4
median_vars = json.loads(sys.argv[2])
CHUNK = 500_000
def chunks(v):
    if v.ndim == 0:
        yield np.ma.filled(np.ma.asarray(v[...]).astype(float), np.nan).ravel()
        return
    for start in range(0, v.shape[0], CHUNK):
        yield np.ma.filled(np.ma.asarray(v[start:start + CHUNK]).astype(float), np.nan).ravel()
out = {}
with netCDF4.Dataset(sys.argv[1]) as ds:
    for name, v in ds.variables.items():
        kind = getattr(v.dtype, "kind", "")
        info = {"units": str(getattr(v, "units", "")), "numeric": kind in "fiu", "all_nan": False}
        if kind == "f" or (kind in "iu" and any(hasattr(v, a) for a in
                ("scale_factor", "add_offset", "_FillValue", "missing_value"))):
            info["all_nan"] = True
            for x in chunks(v):
                finite = x[np.isfinite(x)]
                if finite.size:
                    info["all_nan"] = False
                    if name in median_vars:
                        info["median"] = float(np.median(finite))
                    break
        out[name] = info
print(json.dumps(out))
"""

_cache = {}  # (path, mtime) -> probe; one entry, the file being edited/run


def probe_file(file_path, logger=None, timeout=30):
    """``{variable: {"units", "numeric", "all_nan", "median"?}}`` for every
    variable in ``file_path``, or ``None`` if it can't be read. Cached on
    (path, mtime) so repeated validation of the same file is free."""
    try:
        key = (str(file_path), Path(file_path).stat().st_mtime)
    except OSError as exc:
        if logger:
            logger.info("Could not read '%s': %s", file_path, exc)
        return None
    if key in _cache:
        return _cache[key]

    probe = None
    try:
        # HDF5 file locking can transiently fail (Errno -101) if the load step
        # reopens the file right after this subprocess closes it.
        env = dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE")
        cmd = [sys.executable, "-c", _SCRIPT, str(file_path), json.dumps(MEDIAN_VARIABLES)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode == 0:
            probe = json.loads(result.stdout)
        elif logger:
            err = (result.stderr or "").strip().splitlines()[-1:] or "unknown error"
            logger.info("Could not read '%s' to inspect its variables: %s", file_path, err)
    except Exception as exc:
        if logger:
            logger.info("Could not read '%s' to inspect its variables: %s", file_path, exc)

    _cache.clear()
    _cache[key] = probe
    return probe


def present(probe, name):
    """Whether ``name`` is in the file with real (not all-NaN) data."""
    info = (probe or {}).get(name)
    return bool(info) and not info["all_nan"]
