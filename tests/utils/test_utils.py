import xarray as xr

def create_mock_dataset(lats=None, lons=None, times=None):
    data_vars = {}
    
    if lats is not None:
        data_vars["LATITUDE"] = ("N_MEASUREMENTS", lats)
    if lons is not None:
        data_vars["LONGITUDE"] = ("N_MEASUREMENTS", lons)
    if times is not None:
        data_vars["TIME"] = ("N_MEASUREMENTS", times)
        
    lengths = [len(v[1]) for v in data_vars.values()]
    length = lengths[0] if lengths else 0

    return xr.Dataset(
        data_vars,
        coords={"N_MEASUREMENTS": range(length)},
    )