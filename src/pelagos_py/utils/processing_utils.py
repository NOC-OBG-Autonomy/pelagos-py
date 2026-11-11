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

import numpy as np

# Largest gap between two samples that a value is interpolated across (Interpolate Data's default).
MAX_FILL_GAP_SECONDS = 5 * 60


# ------------------------------- Units -----------------------------------
def cndc_scale_factor(units):
    """Factor converting conductivity to the mS/cm gsw expects; assumes S/m when units are unset."""
    u = (units or "").lower().replace(" ", "").replace("/", "")
    return 1.0 if u.startswith("mscm") else 10.0


# ----------------------------- NaN Handling ------------------------------
def profile_indices(profile_number):
    """Map each finite profile number to its sorted sample indices."""
    pn = np.asarray(profile_number, dtype=float)
    order = np.flatnonzero(np.isfinite(pn))
    order = order[np.argsort(pn[order], kind="stable")]
    keys, starts = np.unique(pn[order], return_index=True)
    ends = np.append(starts[1:], order.size)
    return {k: order[s:e] for k, s, e in zip(keys, starts, ends)}


def interpolate_by_time(values, time):
    """Linearly fill non-finite ``values`` against ``time``; ends stay NaN (no extrapolation)."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return v
    t = np.asarray(time).astype("datetime64[ns]").astype("int64")
    out = v.copy()
    out[~ok] = np.interp(t[~ok], t[ok], v[ok], left=np.nan, right=np.nan)
    return out


def interpolate_at(time, values, query_time, max_gap_seconds=MAX_FILL_GAP_SECONDS):
    """Linearly interpolate the finite ``values`` (at ``time``, in seconds) at each ``query_time``.

    NaN before the first and after the last value, and between two values more than
    ``max_gap_seconds`` apart. Also returns the indices of the values either side of
    each query (equal for an exact match, -1 where the result is NaN).
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    query_time = np.asarray(query_time, dtype=float)
    known = np.flatnonzero(np.isfinite(values) & np.isfinite(time))
    known = known[np.argsort(time[known], kind="stable")]
    result = np.full(query_time.shape, np.nan)
    before = np.full(query_time.shape, -1)
    after = np.full(query_time.shape, -1)
    if known.size == 0:
        return result, before, after

    known_time = time[known]
    right = np.searchsorted(known_time, query_time, side="left")
    right_clipped = np.minimum(right, known.size - 1)
    exact = np.isfinite(query_time) & (known_time[right_clipped] == query_time)
    left = right - 1
    between = ~exact & (left >= 0) & (right < known.size)
    between[between] = (
        known_time[right[between]] - known_time[left[between]] <= max_gap_seconds
    )

    before[exact] = after[exact] = known[right_clipped[exact]]
    before[between] = known[left[between]]
    after[between] = known[right[between]]
    found = exact | between
    result[found] = np.interp(query_time[found], known_time, values[known])
    return result, before, after


def find_nans(data: np.ndarray):
    """
    Handles generation of masks and location indices of nans.
    Intended for 1D arrays.

    Parameters
    ----------
    data : np.ndarray
        numpy array with nans

    Returns
    -------
    nan_mask : np.ndarray
        mask where incices with nans are True
    nan_indices : np.ndarray
        indices of locations where there are nans
    non_nan_indices : np.ndarray
        indices of locations where there are values
    """
    nan_mask = np.isnan(data)
    nan_indices = np.nonzero(nan_mask)[0]
    non_nan_indices = np.nonzero(~nan_mask)[0]
    return nan_mask, nan_indices, non_nan_indices


def interpolate_nans(data, coords):
    """
    Fills nan values in y using interpolation over x.
    x and y must have the same dimensions.

    Parameters
    ----------
    data : np.ndarray
        1D array of size N to interpolate
    coords : np.ndarray
        1D array of size N which the data will be interpolated over

    Returns
    -------
    filled_data : np.ndarray
        data with nans filled using linear interpolation
    """

    # Convert datetimes to floats
    args = [np.array(data), np.array(coords)]
    for i, array in enumerate(args):
        if np.issubdtype(array.dtype, np.datetime64):
            elapsed_time = (args[i] - args[i][0]) / np.timedelta64(1, "s")
            args[i] = elapsed_time

    data, coords = args
    non_nan_mask = ~np.isnan(data)
    filled_data = np.interp(coords, coords[non_nan_mask], data[non_nan_mask])
    return filled_data


def small_netcdf_chunk_cache():
    """Shrink netCDF-C's 64 MB-per-variable chunk cache, which whole-variable reads don't
    use but which stays allocated until the file closes."""
    import netCDF4

    netCDF4.set_chunk_cache(1_000_000, *netCDF4.get_chunk_cache()[1:])


# ----------------------------- Filtering ---------------------------------
def remove_outliers(data):
    """
    Removes outliers (including NaNs) from data. Exclusion is based on
    inter-quartile range.

    Parameters
    ----------
    data : np.ndarray | list
        1D array of size N to interpolate

    Returns
    -------
    filtered_data : np.ndarray
        data with outliers removed

    """
    data = np.array(data)
    data = data[np.isfinite(data)]  # remove NaNs
    if len(data) == 0:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered_data = data[(data >= lower) & (data <= upper)]
    return filtered_data
