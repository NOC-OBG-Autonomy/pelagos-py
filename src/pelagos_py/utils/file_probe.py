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

"""Cheap metadata probe of an OG1 NetCDF file for the config builder: variable
names, units, which float variables are entirely NaN, and a median for a few
variables whose *values* decide how they must be treated (e.g. CNDC unit
mislabelling)."""

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Variables whose median is needed to tell a unit/naming problem from real data.
MEDIAN_VARIABLES = ("CNDC", "BBP700", "BETA_BACKSCATTERING700")

TIMEOUT = 30
CHUNK = 500_000
PACKED_ATTRS = ("scale_factor", "add_offset", "_FillValue", "missing_value")


def probe_file(file_path, logger=None):
    """``{variable: {"units", "numeric", "all_nan", "median"?}}`` for every
    variable in ``file_path``, or ``None`` if it can't be read. Cached on
    (path, mtime) so repeated validation of the same file is free."""
    try:
        mtime = Path(file_path).stat().st_mtime
    except OSError as exc:
        if logger:
            logger.info("Could not read '%s': %s", file_path, exc)
        return None
    probe, error = _probe(str(file_path), mtime)
    if error and logger:
        logger.info("Could not read '%s' to inspect its variables: %s", file_path, error)
    return probe


@functools.lru_cache(maxsize=1)
def _probe(file_path, mtime):
    # `mtime` is only part of the cache key, so an edited file is probed again.
    # Runs this file in a subprocess: some netCDF4/HDF5 + h5py combinations segfault
    # the whole interpreter on open, which try/except can't catch.
    # HDF5 file locking can transiently fail (Errno -101) if the load step
    # reopens the file right after this subprocess closes it.
    env = dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE")
    cmd = [sys.executable, __file__, file_path, json.dumps(MEDIAN_VARIABLES)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env)
    except Exception as exc:
        return None, str(exc)
    if result.returncode != 0:
        stderr_lines = (result.stderr or "").strip().splitlines()
        return None, stderr_lines[-1] if stderr_lines else "unknown error"
    return json.loads(result.stdout), None


def present(probe, name):
    """Whether ``name`` is in the file with real (not all-NaN) data."""
    info = (probe or {}).get(name)
    return bool(info) and not info["all_nan"]


# The subprocess side, run as `python file_probe.py <path> <median_vars>`.
def _chunks(v):
    # Read in slices so a variable with real data settles on its first one.
    if v.ndim == 0:
        yield np.ma.filled(np.ma.asarray(v[...]).astype(float), np.nan).ravel()
        return
    for start in range(0, v.shape[0], CHUNK):
        yield np.ma.filled(np.ma.asarray(v[start:start + CHUNK]).astype(float), np.nan).ravel()


def _summarise(v, want_median):
    kind = getattr(v.dtype, "kind", "")
    info = {"units": str(getattr(v, "units", "")), "numeric": kind in "fiu", "all_nan": False}
    can_be_nan = kind == "f" or (kind in "iu" and any(hasattr(v, a) for a in PACKED_ATTRS))
    if not can_be_nan:
        return info
    info["all_nan"] = True
    finite_parts = []
    for x in _chunks(v):
        finite = x[np.isfinite(x)]
        if finite.size:
            info["all_nan"] = False
            if not want_median:
                break
            finite_parts.append(finite)
    # Median of the whole variable, matching what Prepare OG1 computes at run time.
    if finite_parts:
        info["median"] = float(np.median(np.concatenate(finite_parts)))
    return info


def _summarise_file(file_path, median_vars):
    # Imported here so the parent process never loads netCDF4.
    import netCDF4

    with netCDF4.Dataset(file_path) as ds:
        return {name: _summarise(v, name in median_vars) for name, v in ds.variables.items()}


if __name__ == "__main__":
    print(json.dumps(_summarise_file(sys.argv[1], json.loads(sys.argv[2]))))
