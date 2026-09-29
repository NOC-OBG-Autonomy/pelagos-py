"""Tests the NetCDF metadata probe (src/pelagos_py/utils/file_probe.py)."""

import numpy as np
import xarray as xr

from pelagos_py.utils import file_probe


def test_probe_reports_units_all_nan_and_median(tmp_path):
    path = tmp_path / "g.nc"
    ds = xr.Dataset({
        "CNDC": ("N", np.array([3.0, 4.0, np.nan]), {"units": "S/m"}),
        "EMPTY": ("N", np.full(3, np.nan)),
        "COUNT": ("N", np.array([1, 2, 3], dtype=np.int32)),
    })
    ds.to_netcdf(path)

    probe = file_probe.probe_file(path)

    assert probe["CNDC"] == {"units": "S/m", "numeric": True, "all_nan": False, "median": 3.5}
    assert probe["EMPTY"]["all_nan"] and "median" not in probe["EMPTY"]
    assert file_probe.present(probe, "CNDC")
    assert not file_probe.present(probe, "EMPTY")
    assert not file_probe.present(probe, "MISSING")


def test_unreadable_file_returns_none(tmp_path):
    assert file_probe.probe_file(tmp_path / "missing.nc") is None
    bad = tmp_path / "bad.nc"
    bad.write_text("not a netcdf file")
    assert file_probe.probe_file(bad) is None
