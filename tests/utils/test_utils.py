import xarray as xr
import numpy as np


def create_mock_dataset(lats=None, lons=None, times=None):
    def to_array(x):
        if x is None:
            return None
        if np.ndim(x) == 0:
            x = [x]
        return np.array(x)

    lats = to_array(lats)
    lons = to_array(lons)
    times = to_array(times)

    arrays = [a for a in (lats, lons, times) if a is not None]
    length = max((len(a) for a in arrays), default=0)

    def tile(a):
        if a is None or length == 0:
            return None
        reps = -(-length // len(a))
        return np.tile(a, reps)[:length]

    lats = tile(lats)
    lons = tile(lons)
    times = tile(times)

    data_vars = {}
    if lats is not None:
        data_vars["LATITUDE"] = ("N_MEASUREMENTS", lats)
    if lons is not None:
        data_vars["LONGITUDE"] = ("N_MEASUREMENTS", lons)
    if times is not None:
        data_vars["TIME"] = ("N_MEASUREMENTS", times)

    return xr.Dataset(
        data_vars,
        coords={"N_MEASUREMENTS": range(length)},
    )
