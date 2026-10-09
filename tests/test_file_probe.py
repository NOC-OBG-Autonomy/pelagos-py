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


def test_median_covers_every_chunk(tmp_path, monkeypatch):
    # First chunk in air (~0), the rest in seawater (mS/cm): the median must see it all.
    monkeypatch.setattr(file_probe, "CHUNK", 2)
    path = tmp_path / "g.nc"
    cndc = np.array([0.0, 0.0, 36.0, 36.0, 36.0, 36.0])
    xr.Dataset({"CNDC": ("N", cndc, {"units": "S/m"})}).to_netcdf(path)
    assert file_probe._summarise_file(str(path), ["CNDC"])["CNDC"]["median"] == 36.0



def test_dive_depths_finds_each_bottom_including_yos(tmp_path):
    path = tmp_path / "g.nc"
    # Surface -> 1000, yo up to 600 and back to 990, surface -> 790; 2 dbar wiggles are noise.
    pres = np.concatenate([
        np.linspace(0, 1000, 50), np.linspace(1000, 600, 20), np.linspace(600, 990, 20),
        np.linspace(990, 0, 50), [2, 0, 2, 0], np.linspace(0, 790, 40), np.linspace(790, 0, 40),
    ])
    xr.Dataset({"PRES": ("N", pres)}).to_netcdf(path)
    assert file_probe._summarise_file(str(path), [])["PRES"]["dive_depths"] == [1000.0, 990.0, 790.0]
