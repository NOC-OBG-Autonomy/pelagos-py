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
mislabelling), plus the depth of each dive for the deep correction threshold."""

import functools
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

# Variables whose median is needed to tell a unit/naming problem from real data.
MEDIAN_VARIABLES = ("CNDC", "BBP700", "BETA_BACKSCATTERING700")
DIVE_VARIABLE = "PRES"
# A dive bottom must sit this many dbar below the shallowest point either side of it.
DIVE_PROMINENCE = 50
DIVE_BLOCKS = 100_000
# Reading every point of a big file takes too long, so at most this many are read.
DIVE_READ_LIMIT = 4_000_000
DIVE_WINDOWS = 20

TIMEOUT = 30
CHUNK = 500_000
PACKED_ATTRS = ("scale_factor", "add_offset", "_FillValue", "missing_value")


def probe_file(file_path, logger=None):
    """``{variable: {"units", "numeric", "all_nan", "median"?}}`` for every
    variable in ``file_path`` (PRES also gets ``"dive_depths"``), or ``None``
    if it can't be read. Cached on (path, mtime) so repeated validation of the
    same file is free."""
    try:
        mtime = Path(file_path).stat().st_mtime
    except OSError as exc:
        if logger:
            logger.info("Could not read '%s': %s", file_path, exc)
        return None
    probe, error = _probe(str(file_path), mtime)
    if error and logger:
        logger.info(
            "Could not read '%s' to inspect its variables: %s", file_path, error
        )
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
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env
        )
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
        yield np.ma.filled(
            np.ma.asarray(v[start : start + CHUNK]).astype(float), np.nan
        ).ravel()


def _summarise(v, want_median):
    kind = getattr(v.dtype, "kind", "")
    info = {
        "units": str(getattr(v, "units", "")),
        "numeric": kind in "fiu",
        "all_nan": False,
    }
    can_be_nan = kind == "f" or (
        kind in "iu" and any(hasattr(v, a) for a in PACKED_ATTRS)
    )
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


def _read_pres(v, start, stop):
    # Unmasked reads are ~5x faster on big files, so fill values are NaN'd by hand.
    import netCDF4

    v.set_auto_mask(False)
    pres = v[start:stop].ravel()
    if pres.dtype.kind != "f":
        pres = pres.astype(float)
    fills = [getattr(v, a) for a in ("_FillValue", "missing_value") if hasattr(v, a)]
    for fill in fills or [netCDF4.default_fillvals.get(v.dtype.str[1:])]:
        pres[pres == fill] = np.nan
    return pres


def _bottoms(pres, blocks):
    # Block maxima keep every bottom but cap the walk below at `blocks` steps.
    size = -(-pres.size // blocks)
    padded = np.full(size * blocks, np.nan, dtype=pres.dtype)
    padded[: pres.size] = pres
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
        block_max = np.nanmax(padded.reshape(-1, size), axis=1)
    block_max = block_max[np.isfinite(block_max)].tolist()

    bottoms = []
    deepest = None
    shallowest = block_max[0] if block_max else 0.0
    for p in block_max:
        if deepest is None:
            if p >= shallowest + DIVE_PROMINENCE:
                deepest = p
            else:
                shallowest = min(shallowest, p)
        elif p > deepest:
            deepest = p
        elif p <= deepest - DIVE_PROMINENCE:
            bottoms.append(round(deepest, 1))
            deepest = None
            shallowest = p
    return bottoms


def _dive_depths(v):
    # Bottom of every dive and yo; files over DIVE_READ_LIMIT are sampled in evenly spaced slices.
    n = v.shape[0]
    windows = 1 if n <= DIVE_READ_LIMIT else DIVE_WINDOWS
    length = min(n, DIVE_READ_LIMIT // windows)
    bottoms = []
    for start in np.linspace(0, n - length, windows).astype(int):
        bottoms += _bottoms(
            _read_pres(v, start, start + length), DIVE_BLOCKS // windows
        )
    return bottoms


def _summarise_file(file_path, median_vars):
    # Imported here so the parent process never loads netCDF4.
    import netCDF4

    with netCDF4.Dataset(file_path) as ds:
        probe = {
            name: _summarise(v, name in median_vars) for name, v in ds.variables.items()
        }
        if DIVE_VARIABLE in probe and not probe[DIVE_VARIABLE]["all_nan"]:
            probe[DIVE_VARIABLE]["dive_depths"] = _dive_depths(
                ds.variables[DIVE_VARIABLE]
            )
        return probe


if __name__ == "__main__":
    print(json.dumps(_summarise_file(sys.argv[1], json.loads(sys.argv[2]))))
